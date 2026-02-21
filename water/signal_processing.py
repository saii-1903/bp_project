"""
signal_processing.py — Filtering, baseline correction, and peak detection.

Implements the pre-processing pipeline described in the research document:
  1.  5th-order zero-phase elliptic bandpass (0.5 – 5.0 Hz)
  2.  Cubic-spline baseline correction through detected valleys
  3.  Systolic peak and diastolic valley detection
  4.  Signal-quality index (SQI) per cardiac cycle
"""

import numpy as np
from scipy.signal import ellip, sosfiltfilt, find_peaks
from scipy.interpolate import CubicSpline

import config as cfg


# ─── Bandpass Filter ─────────────────────────────────────────────────
def _design_bandpass(fs: int) -> np.ndarray:
    """Design the elliptic bandpass filter (second-order sections)."""
    nyquist = fs / 2.0
    low = cfg.BANDPASS_LOW_HZ / nyquist
    high = cfg.BANDPASS_HIGH_HZ / nyquist
    sos = ellip(
        cfg.BANDPASS_ORDER,
        cfg.BANDPASS_RIPPLE_DB,
        cfg.BANDPASS_ATTENUATION_DB,
        [low, high],
        btype="bandpass",
        output="sos",
    )
    return sos


def bandpass_filter(signal: np.ndarray, fs: int) -> np.ndarray:
    """Apply zero-phase elliptic bandpass filtering."""
    sos = _design_bandpass(fs)
    return sosfiltfilt(sos, signal)


# ─── Peak & Valley Detection ────────────────────────────────────────
def detect_peaks_valleys(filtered: np.ndarray, fs: int):
    """
    Detect systolic peaks and diastolic valleys.

    Returns
    -------
    peaks   : 1-D int array of peak sample indices
    valleys : 1-D int array of valley sample indices
    """
    min_distance = int(fs * 0.4)  # At least 0.4 s between peaks (~150 bpm max)

    peaks, _ = find_peaks(filtered, distance=min_distance,
                          prominence=0.01 * np.ptp(filtered))
    valleys, _ = find_peaks(-filtered, distance=min_distance,
                            prominence=0.01 * np.ptp(filtered))
    return peaks, valleys


# ─── Baseline Correction (Cubic Spline) ─────────────────────────────
def baseline_correction(signal: np.ndarray, valleys: np.ndarray) -> tuple:
    """
    Remove baseline wander via cubic spline interpolation through valleys.

    Returns
    -------
    corrected : baseline-removed AC waveform
    baseline  : the fitted baseline curve
    """
    if len(valleys) < 4:
        # Not enough anchor points — return signal as-is
        baseline = np.zeros_like(signal)
        return signal, baseline

    cs = CubicSpline(valleys, signal[valleys], extrapolate=True)
    baseline = cs(np.arange(len(signal)))
    corrected = signal - baseline
    return corrected, baseline


# ─── Signal Quality Index ───────────────────────────────────────────
def signal_quality_index(corrected: np.ndarray, peaks: np.ndarray,
                         valleys: np.ndarray) -> np.ndarray:
    """
    Per-beat SQI based on amplitude consistency.

    A beat is flagged as low-quality if its amplitude deviates more than
    2× the median absolute deviation from the median amplitude.

    Returns
    -------
    sqi : boolean array aligned with *peaks* (True = good quality)
    """
    if len(peaks) == 0:
        return np.array([], dtype=bool)

    # Find nearest preceding valley for each peak
    amplitudes = []
    for pk in peaks:
        preceding = valleys[valleys < pk]
        if len(preceding) == 0:
            amplitudes.append(0.0)
        else:
            vl = preceding[-1]
            amplitudes.append(corrected[pk] - corrected[vl])

    amplitudes = np.array(amplitudes)
    med = np.median(amplitudes)
    mad = np.median(np.abs(amplitudes - med)) + 1e-9
    sqi = np.abs(amplitudes - med) < 2.0 * mad
    return sqi


# ─── Full Pre-processing Pipeline ───────────────────────────────────
def preprocess(raw_ppg: np.ndarray, fs: int) -> dict:
    """
    Run the complete pre-processing chain.

    Returns
    -------
    dict with keys:
        "filtered"  – bandpass-filtered signal
        "corrected" – baseline-corrected AC waveform
        "baseline"  – estimated DC baseline
        "peaks"     – systolic peak indices
        "valleys"   – diastolic valley indices
        "sqi"       – per-beat quality flags
    """
    filtered = bandpass_filter(raw_ppg, fs)
    peaks, valleys = detect_peaks_valleys(filtered, fs)
    corrected, baseline = baseline_correction(filtered, valleys)

    # Re-detect peaks on the corrected signal for downstream accuracy
    peaks_c, valleys_c = detect_peaks_valleys(corrected, fs)
    sqi = signal_quality_index(corrected, peaks_c, valleys_c)

    return {
        "filtered": filtered,
        "corrected": corrected,
        "baseline": baseline,
        "peaks": peaks_c,
        "valleys": valleys_c,
        "sqi": sqi,
    }
