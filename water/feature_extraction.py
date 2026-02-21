"""
feature_extraction.py — Per-beat and windowed feature extraction.

Extracts hydration-relevant features from pre-processed PPG data:
  • Perfusion Index (PI)
  • Systolic Amplitude
  • Pulse Width
  • VPG / APG derivatives and b/a ratio
  • Heart Rate and HRV statistics (SDNN, RMSSD)
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

import config as cfg


# ─── Per-Beat Feature Computation ────────────────────────────────────

def _per_beat_features(corrected: np.ndarray, raw_ppg: np.ndarray,
                       peaks: np.ndarray, valleys: np.ndarray,
                       sqi: np.ndarray, fs: int) -> pd.DataFrame:
    """
    Compute features for each individual cardiac cycle.

    Only beats flagged as good quality (sqi=True) are retained.
    """
    records = []

    for i, pk in enumerate(peaks):
        if i >= len(sqi) or not sqi[i]:
            continue

        # Find the nearest preceding and following valleys
        pre = valleys[valleys < pk]
        post = valleys[valleys > pk]
        if len(pre) == 0 or len(post) == 0:
            continue

        v_start = pre[-1]
        v_end = post[0]

        # Basic morphology
        ac_amplitude = corrected[pk] - corrected[v_start]
        dc_component = np.mean(raw_ppg[v_start:v_end + 1])
        perfusion_index = (ac_amplitude / (dc_component + 1e-9)) * 100

        # Filter extreme artifacts: PI > 500% is physically impossible
        if perfusion_index > 500.0:
            continue

        # Clamp PI at 20.0 to prevent trend spikes from artifacts
        perfusion_index = min(perfusion_index, 20.0)
            
        pulse_width_s = (v_end - v_start) / fs
        time_s = pk / fs

        # IBI / HR
        if i > 0 and i - 1 < len(peaks):
            prev_good = None
            for j in range(i - 1, -1, -1):
                if j < len(sqi) and sqi[j]:
                    prev_good = j
                    break
            if prev_good is not None:
                ibi_s = (pk - peaks[prev_good]) / fs
                hr_bpm = 60.0 / ibi_s if ibi_s > 0 else np.nan
            else:
                ibi_s = np.nan
                hr_bpm = np.nan
        else:
            ibi_s = np.nan
            hr_bpm = np.nan

        # APG features (2nd derivative of the beat segment)
        beat_seg = corrected[v_start:v_end + 1].copy()
        if len(beat_seg) >= 7:
            # Smooth lightly before differentiation
            vpg = np.gradient(beat_seg, 1.0 / fs)
            apg = np.gradient(vpg, 1.0 / fs)

            # a-wave: first positive max, b-wave: first negative min after a
            a_idx = np.argmax(apg[:len(apg) // 2])
            a_wave = apg[a_idx]

            search_start = a_idx + 1
            if search_start < len(apg) // 2:
                b_idx = search_start + np.argmin(apg[search_start:len(apg) // 2])
                b_wave = apg[b_idx]
            else:
                b_wave = np.nan

            ba_ratio = b_wave / (a_wave + 1e-9) if not np.isnan(b_wave) else np.nan

            # VPG max slope
            vpg_max = np.max(vpg)
        else:
            ba_ratio = np.nan
            vpg_max = np.nan

        records.append({
            "time_s": time_s,
            "peak_idx": pk,
            "ac_amplitude": ac_amplitude,
            "dc_component": dc_component,
            "perfusion_index": perfusion_index,
            "pulse_width_s": pulse_width_s,
            "hr_bpm": hr_bpm,
            "ibi_s": ibi_s,
            "ba_ratio": ba_ratio,
            "vpg_max_slope": vpg_max,
        })

    return pd.DataFrame(records)


# ─── Windowed Aggregation ────────────────────────────────────────────

def _windowed_features(beat_df: pd.DataFrame, total_duration_s: float) -> pd.DataFrame:
    """
    Aggregate per-beat features into overlapping time windows.
    """
    win = cfg.FEATURE_WINDOW_SECONDS
    step = win * (1 - cfg.FEATURE_WINDOW_OVERLAP)
    records = []

    t = 0.0
    while t + win <= total_duration_s:
        mask = (beat_df["time_s"] >= t) & (beat_df["time_s"] < t + win)
        window = beat_df.loc[mask]

        if len(window) < cfg.MIN_PEAKS_PER_WINDOW:
            t += step
            continue

        ibis = window["ibi_s"].dropna()
        hrv_sdnn = ibis.std() if len(ibis) > 1 else np.nan
        hrv_rmssd = (np.sqrt(np.mean(np.diff(ibis) ** 2))
                     if len(ibis) > 2 else np.nan)

        records.append({
            "window_center_s": t + win / 2,
            "perfusion_index": window["perfusion_index"].median(),
            "systolic_amplitude": window["ac_amplitude"].median(),
            "pulse_width_s": window["pulse_width_s"].median(),
            "hr_bpm": window["hr_bpm"].median(),
            "ba_ratio": window["ba_ratio"].median(),
            "vpg_max_slope": window["vpg_max_slope"].median(),
            "hrv_sdnn": hrv_sdnn,
            "hrv_rmssd": hrv_rmssd,
            "n_beats": len(window),
        })
        t += step

    return pd.DataFrame(records)


# ─── Public API ──────────────────────────────────────────────────────

def extract_features(corrected: np.ndarray, raw_ppg: np.ndarray,
                     peaks: np.ndarray, valleys: np.ndarray,
                     sqi: np.ndarray, fs: int) -> pd.DataFrame:
    """
    Full feature-extraction pipeline.

    Returns a DataFrame with one row per time window containing:
      window_center_s, perfusion_index, systolic_amplitude,
      pulse_width_s, hr_bpm, ba_ratio, vpg_max_slope,
      hrv_sdnn, hrv_rmssd, n_beats
    """
    total_duration_s = len(corrected) / fs

    beat_df = _per_beat_features(corrected, raw_ppg, peaks, valleys, sqi, fs)
    if beat_df.empty:
        return pd.DataFrame()

    windowed = _windowed_features(beat_df, total_duration_s)
    return windowed
