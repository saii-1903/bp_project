"""
ppg_simulator.py — Realistic synthetic PPG signal generator.

Produces a 60 Hz photoplethysmogram that passes through four hydration
phases (baseline → dehydration → water-intake → rehydration), complete
with respiratory modulation, Gaussian noise, and motion-artifact bursts.
"""

import numpy as np
import config as cfg


def _cardiac_template(phase: float, amplitude: float, notch_ratio: float,
                      diastolic_ratio: float) -> float:
    """
    Single-sample value of a stylised cardiac pulse at *phase* ∈ [0, 1).

    The template is built from three Gaussian-like bumps:
      • systolic peak   at  phase ≈ 0.15
      • dicrotic notch  at  phase ≈ 0.40  (negative dip)
      • diastolic peak  at  phase ≈ 0.55
    """
    systolic = amplitude * np.exp(-((phase - 0.15) ** 2) / (2 * 0.01))
    notch = -notch_ratio * amplitude * np.exp(-((phase - 0.40) ** 2) / (2 * 0.005))
    diastolic = diastolic_ratio * amplitude * np.exp(-((phase - 0.55) ** 2) / (2 * 0.015))
    return systolic + notch + diastolic


def _hydration_profile(n_samples: int) -> np.ndarray:
    """
    Returns a per-sample hydration factor in [0, 1].

    1.0 = fully hydrated (baseline)
    0.0 = maximally dehydrated

    The profile encodes four consecutive phases whose durations are
    defined in config.py.
    """
    fs = cfg.SAMPLING_RATE_HZ
    s_baseline = int(cfg.PHASE_BASELINE_MIN * 60 * fs)
    s_dehydr = int(cfg.PHASE_DEHYDRATION_MIN * 60 * fs)
    s_intake = int(cfg.PHASE_WATER_INTAKE_MIN * 60 * fs)
    s_rehydr = int(cfg.PHASE_REHYDRATION_MIN * 60 * fs)

    profile = np.ones(n_samples)

    # Phase 1 – baseline (already 1.0)
    idx = s_baseline

    # Phase 2 – linear dehydration ramp to 0.35
    end = min(idx + s_dehydr, n_samples)
    ramp = np.linspace(1.0, 0.35, end - idx)
    profile[idx:end] = ramp
    idx = end

    # Phase 3 – water intake (rapid rise from 0.35 → 0.75)
    end = min(idx + s_intake, n_samples)
    ramp = np.linspace(0.35, 0.75, end - idx)
    profile[idx:end] = ramp
    idx = end

    # Phase 4 – slow rehydration recovery (0.75 → 0.95)
    end = min(idx + s_rehydr, n_samples)
    ramp = np.linspace(0.75, 0.95, end - idx)
    profile[idx:end] = ramp
    idx = end

    # Any remaining samples stay at last value
    if idx < n_samples:
        profile[idx:] = profile[idx - 1]

    return profile


def generate_ppg(duration_minutes: float | None = None,
                 seed: int = 42) -> dict:
    """
    Generate a complete synthetic PPG record.

    Parameters
    ----------
    duration_minutes : float, optional
        Override the simulation length (default from config).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        "time_s"            – 1-D array of time stamps (seconds)
        "raw_ppg"           – 1-D array of the raw PPG signal
        "hydration_factor"  – 1-D ground-truth hydration profile [0–1]
        "fs"                – sampling rate in Hz
    """
    rng = np.random.default_rng(seed)
    dur = duration_minutes or cfg.SIMULATION_DURATION_MINUTES
    fs = cfg.SAMPLING_RATE_HZ
    n_samples = int(dur * 60 * fs)

    time_s = np.arange(n_samples) / fs
    hydration = _hydration_profile(n_samples)

    # ── Build the PPG sample-by-sample ──────────────────────────────
    ppg = np.zeros(n_samples)

    # Instantaneous heart rate modulated by hydration
    hr_bpm = (cfg.HEART_RATE_BPM_BASELINE +
              (cfg.HEART_RATE_BPM_DEHYDRATED - cfg.HEART_RATE_BPM_BASELINE)
              * (1.0 - hydration))
    hr_hz = hr_bpm / 60.0

    # Cumulative cardiac phase
    phase_acc = np.cumsum(hr_hz / fs) % 1.0

    # Systolic amplitude varies with hydration
    amp = cfg.SYSTOLIC_AMPLITUDE_BASELINE * (0.5 + 0.5 * hydration)

    for i in range(n_samples):
        ppg[i] = _cardiac_template(
            phase_acc[i],
            amplitude=amp[i],
            notch_ratio=cfg.DICROTIC_NOTCH_RATIO,
            diastolic_ratio=cfg.DIASTOLIC_AMPLITUDE_RATIO,
        )

    # ── Add DC baseline (~1.0) ──────────────────────────────────────
    ppg += 1.0

    # ── Respiratory modulation (RIIV + RIAV) ────────────────────────
    resp = np.sin(2 * np.pi * cfg.RESP_RATE_HZ * time_s)
    ppg += cfg.RIIV_AMPLITUDE * resp                         # baseline shift
    ppg *= (1.0 + cfg.RIAV_AMPLITUDE * resp)                 # amplitude mod

    # ── Gaussian noise ──────────────────────────────────────────────
    ppg += rng.normal(0, cfg.NOISE_STD, n_samples)

    # ── Sporadic motion artefacts ───────────────────────────────────
    artifact_mask = rng.random(n_samples) < cfg.MOTION_ARTIFACT_PROBABILITY
    ppg[artifact_mask] += rng.uniform(
        -cfg.MOTION_ARTIFACT_AMPLITUDE,
        cfg.MOTION_ARTIFACT_AMPLITUDE,
        artifact_mask.sum(),
    )

    return {
        "time_s": time_s,
        "raw_ppg": ppg,
        "hydration_factor": hydration,
        "fs": fs,
    }
