"""
hydration_engine.py — Hydration Index computation and trend detection.

Implements:
  • Personal baseline establishment
  • Hydration Index (HI) as relative change from baseline
  • EWMA temporal smoothing
  • Moving-slope trend estimation
  • Seasonal Kendall test for monotonic trend significance
  • 4-level hydration classification
"""

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

import config as cfg


# ─── Personal Baseline ──────────────────────────────────────────────

def _establish_baseline(features: pd.DataFrame) -> dict:
    """
    Use the first BASELINE_WINDOW_MINUTES of data to build a personal
    baseline for each feature.
    """
    cutoff_s = cfg.BASELINE_WINDOW_MINUTES * 60
    baseline_data = features[features["window_center_s"] <= cutoff_s]

    if baseline_data.empty:
        baseline_data = features.head(3)

    baseline = {}
    for col in ["perfusion_index", "systolic_amplitude", "ba_ratio",
                "hr_bpm", "pulse_width_s", "vpg_max_slope",
                "hrv_sdnn", "hrv_rmssd"]:
        val = baseline_data[col].median()
        baseline[col] = val if not np.isnan(val) else 1e-9

    return baseline


# ─── Hydration Index ────────────────────────────────────────────────

def _compute_hydration_index(features: pd.DataFrame,
                             baseline: dict) -> pd.Series:
    """
    Multi-Feature Hydration Index (HI) Fusion.
    
    Combines:
    1. Perfusion Index (PI) - Primary volume/flow marker (50% weight)
    2. Systolic Amplitude - Secondary volume marker (30% weight)
    3. Heart Rate - Compensatory marker (20% weight - direction inverted)
    
    Each is computed as % change from personal baseline.
    """
    # 1. PI Component (Increases with hydration)
    pi_b = baseline["perfusion_index"]
    pi_change = ((features["perfusion_index"] - pi_b) / (abs(pi_b) + 1e-9)) * 100
    
    # 2. Systolic Amplitude Component (Increases with hydration)
    amp_b = baseline["systolic_amplitude"]
    amp_change = ((features["systolic_amplitude"] - amp_b) / (abs(amp_b) + 1e-9)) * 100
    
    # 3. Heart Rate Component (HR DECREASES with hydration generally)
    hr_b = baseline["hr_bpm"]
    hr_change = ((features["hr_bpm"] - hr_b) / (abs(hr_b) + 1e-9)) * 100
    
    # Fused Index
    # Note: Heart rate goes UP when dehydrated, so we subtract its positive change
    hi = (0.50 * pi_change) + (0.30 * amp_change) - (0.20 * hr_change)
    
    return hi


# ─── EWMA Smoothing ────────────────────────────────────────────────

def _smooth_ewma(series: pd.Series, span_minutes: float,
                 window_step_s: float) -> pd.Series:
    """Exponentially weighted moving average with span in real time."""
    span_points = max(1, int((span_minutes * 60) / window_step_s))
    return series.ewm(span=span_points, adjust=False).mean()


# ─── Moving Slope ───────────────────────────────────────────────────

def _moving_slope(series: pd.Series, window: int) -> pd.Series:
    """
    Compute the slope (per point) over a rolling window using
    simple linear regression.
    """
    slopes = pd.Series(np.nan, index=series.index)
    values = series.values
    for i in range(window, len(values)):
        y = values[i - window:i]
        if np.any(np.isnan(y)):
            continue
        x = np.arange(window, dtype=float)
        slope = np.polyfit(x, y, 1)[0]
        slopes.iloc[i] = slope
    return slopes


# ─── Seasonal Kendall Trend Test ────────────────────────────────────

def _kendall_trend(series: pd.Series) -> dict:
    """
    Non-parametric Kendall test for monotonic trend.

    Returns dict with tau, p_value, and trend direction string.
    """
    clean = series.dropna()
    if len(clean) < 4:
        return {"tau": 0, "p_value": 1.0, "trend": "Insufficient data"}

    x = np.arange(len(clean))
    tau, p_value = kendalltau(x, clean.values)

    if p_value > cfg.KENDALL_SIGNIFICANCE:
        direction = "No significant trend"
    elif tau > 0:
        direction = "Increasing Hydration ↑"
    else:
        direction = "Decreasing Hydration ↓"

    return {"tau": round(tau, 4), "p_value": round(p_value, 6),
            "trend": direction}


# ─── 4-Level Classification ────────────────────────────────────────

def _classify_hydration(hi_value: float) -> str:
    """Map a single HI value to a descriptive level."""
    if hi_value >= cfg.HI_FULLY_HYDRATED:
        return "Optimal (Full)"
    elif hi_value >= cfg.HI_MILD_DEHYDRATION:
        return "Fair (Stable)"
    elif hi_value >= cfg.HI_MODERATE_DEHYDRATION:
        return "Low (Drink Water)"
    else:
        return "Very Low (Dehydrated)"


def _classify_color(hi_value: float) -> str:
    """Map HI to a zone colour key (from config.COLORS)."""
    if hi_value >= cfg.HI_FULLY_HYDRATED:
        return cfg.COLORS["hydration_green"]
    elif hi_value >= cfg.HI_MILD_DEHYDRATION:
        return cfg.COLORS["hydration_yellow"]
    elif hi_value >= cfg.HI_MODERATE_DEHYDRATION:
        return cfg.COLORS["hydration_orange"]
    else:
        return cfg.COLORS["hydration_red"]


# ─── Public API ──────────────────────────────────────────────────────

def analyse_hydration(features: pd.DataFrame) -> dict:
    """
    Full hydration-analysis pipeline.

    Parameters
    ----------
    features : DataFrame from feature_extraction.extract_features()

    Returns
    -------
    dict with keys:
        "results"          – DataFrame with HI, smoothed HI, slope,
                             classification, and zone colors
        "baseline"         – dict of personal baseline values
        "overall_kendall"  – Kendall trend result for the full session
        "phase_trends"     – list of Kendall results per phase segment
    """
    baseline = _establish_baseline(features)

    results = features[["window_center_s"]].copy()
    results["hydration_index"] = _compute_hydration_index(features, baseline).values

    # Smoothing
    window_step_s = cfg.FEATURE_WINDOW_SECONDS * (1 - cfg.FEATURE_WINDOW_OVERLAP)
    results["hi_smoothed"] = _smooth_ewma(
        results["hydration_index"], cfg.EWMA_SPAN_MINUTES, window_step_s
    ).values

    # Moving slope
    results["slope"] = _moving_slope(
        results["hi_smoothed"], cfg.TREND_WINDOW_POINTS
    ).values

    # Classification
    results["hydration_level"] = results["hi_smoothed"].apply(_classify_hydration)
    results["zone_color"] = results["hi_smoothed"].apply(_classify_color)

    # Trend direction per point
    results["trend_direction"] = results["slope"].apply(
        lambda s: "↑ Increasing" if s > 0.1
        else ("↓ Decreasing" if s < -0.1 else "→ Stable")
        if not np.isnan(s) else ""
    )

    # Overall Kendall trend
    overall_kendall = _kendall_trend(results["hi_smoothed"])

    # Phase-segmented trends (split into thirds for summary)
    n = len(results)
    third = max(1, n // 3)
    phase_trends = []
    for label, sl in [("Early", slice(0, third)),
                      ("Mid", slice(third, 2 * third)),
                      ("Late", slice(2 * third, n))]:
        kt = _kendall_trend(results["hi_smoothed"].iloc[sl])
        kt["phase"] = label
        phase_trends.append(kt)

    # Copy over raw features for plotting
    for col in ["perfusion_index", "systolic_amplitude", "ba_ratio",
                "hr_bpm", "hrv_sdnn"]:
        if col in features.columns:
            results[col] = features[col].values

    return {
        "results": results,
        "baseline": baseline,
        "overall_kendall": overall_kendall,
        "phase_trends": phase_trends,
    }
