"""
visualization.py — Multi-panel dark-themed graphical output.

Generates a comprehensive figure with:
  Panel 1: Raw vs. Filtered PPG (zoomed 10-second window)
  Panel 2: Extracted Features Over Time (PI, amplitude, b/a ratio)
  Panel 3: Hydration Index Trend with color-coded zones
  Panel 4: Heart Rate & HRV Over Time
  Panel 5: Summary Statistics
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection

import config as cfg


def _apply_dark_theme(fig, axes):
    """Apply the dark colour palette to all axes."""
    fig.patch.set_facecolor(cfg.COLORS["background"])
    for ax in axes:
        ax.set_facecolor(cfg.COLORS["panel_bg"])
        ax.tick_params(colors=cfg.COLORS["text"], labelsize=9)
        ax.xaxis.label.set_color(cfg.COLORS["text"])
        ax.yaxis.label.set_color(cfg.COLORS["text"])
        ax.title.set_color(cfg.COLORS["text"])
        for spine in ax.spines.values():
            spine.set_color(cfg.COLORS["grid"])
        ax.grid(True, color=cfg.COLORS["grid"], linewidth=0.5, alpha=0.5)


def _colored_line(ax, x, y, colors, linewidth=2.0):
    """Draw a line whose colour changes per segment."""
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, colors=colors[:-1], linewidths=linewidth)
    ax.add_collection(lc)
    ax.set_xlim(x.min(), x.max())
    pad = (y.max() - y.min()) * 0.1 + 0.01
    ax.set_ylim(y.min() - pad, y.max() + pad)


def generate_all(sim_data: dict, proc_data: dict,
                 features: pd.DataFrame, hydration: dict):
    """
    Create and save the full visualisation suite.

    Parameters
    ----------
    sim_data   : dict from ppg_simulator.generate_ppg()
    proc_data  : dict from signal_processing.preprocess()
    features   : DataFrame from feature_extraction.extract_features()
    hydration  : dict from hydration_engine.analyse_hydration()
    """
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    results = hydration["results"]

    fig, axes = plt.subplots(5, 1, figsize=(cfg.FIGURE_WIDTH, cfg.FIGURE_HEIGHT),
                             gridspec_kw={"height_ratios": [1, 1.2, 1.5, 1, 0.6]})
    _apply_dark_theme(fig, axes)

    fig.suptitle("PPG-Based Hydration Trend Discovery",
                 fontsize=20, fontweight="bold",
                 color=cfg.COLORS["accent"], y=0.98)

    time_s = sim_data["time_s"]
    fs = sim_data["fs"]

    # ─── Panel 1: Raw vs Filtered PPG (10-s zoom) ───────────────────
    ax = axes[0]
    zoom_start = int(15 * 60 * fs)  # 15 minutes in (during dehydration)
    zoom_end = zoom_start + int(10 * fs)
    zoom_end = min(zoom_end, len(time_s))

    t_zoom = time_s[zoom_start:zoom_end]
    ax.plot(t_zoom, sim_data["raw_ppg"][zoom_start:zoom_end],
            color=cfg.COLORS["raw_signal"], alpha=0.5, linewidth=0.8,
            label="Raw PPG")
    ax.plot(t_zoom, proc_data["filtered"][zoom_start:zoom_end],
            color=cfg.COLORS["filtered_signal"], linewidth=1.5,
            label="Filtered PPG")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("① Raw vs. Filtered PPG Signal (10-second window during dehydration phase)",
                 fontsize=12, fontweight="bold", pad=10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.3,
              labelcolor=cfg.COLORS["text"])

    # ─── Panel 2: Extracted Features Over Time ─────────────────────
    ax = axes[1]
    t_min = results["window_center_s"] / 60  # convert to minutes

    ax2_twin = ax.twinx()

    ln1 = ax.plot(t_min, results["perfusion_index"],
                  color=cfg.COLORS["pi_line"], linewidth=1.8,
                  label="Perfusion Index", marker=".", markersize=3)
    ln2 = ax.plot(t_min, results["systolic_amplitude"],
                  color=cfg.COLORS["amplitude_line"], linewidth=1.8,
                  label="Systolic Amplitude", marker=".", markersize=3)
    ln3 = ax2_twin.plot(t_min, results["ba_ratio"],
                        color=cfg.COLORS["ba_ratio_line"], linewidth=1.8,
                        label="b/a Ratio", marker=".", markersize=3, linestyle="--")

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("PI / Amplitude", color=cfg.COLORS["text"])
    ax2_twin.set_ylabel("b/a Ratio", color=cfg.COLORS["ba_ratio_line"])
    ax2_twin.tick_params(axis="y", colors=cfg.COLORS["ba_ratio_line"])
    ax2_twin.spines["right"].set_color(cfg.COLORS["ba_ratio_line"])

    lns = ln1 + ln2 + ln3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc="upper right", fontsize=9, framealpha=0.3,
              labelcolor=cfg.COLORS["text"])
    ax.set_title("② Extracted PPG Features Over Time",
                 fontsize=12, fontweight="bold", pad=10)

    # ─── Panel 3: Hydration Index Trend ────────────────────────────
    ax = axes[2]
    hi_smooth = results["hi_smoothed"].values
    zone_colors = results["zone_color"].values

    # Color-coded hydration line
    _colored_line(ax, t_min.values, hi_smooth, zone_colors, linewidth=2.5)

    # Raw HI as faint background
    ax.plot(t_min, results["hydration_index"],
            color=cfg.COLORS["text"], alpha=0.15, linewidth=0.8)

    # Zero reference line
    ax.axhline(0, color=cfg.COLORS["text"], alpha=0.3, linestyle="--", linewidth=0.8)

    # Hydration zone bands
    ax.axhspan(0, hi_smooth.max() + 10, color=cfg.COLORS["hydration_green"], alpha=0.08)
    ax.axhspan(cfg.HI_MILD_DEHYDRATION, 0,
               color=cfg.COLORS["hydration_yellow"], alpha=0.08)
    ax.axhspan(cfg.HI_MODERATE_DEHYDRATION, cfg.HI_MILD_DEHYDRATION,
               color=cfg.COLORS["hydration_orange"], alpha=0.08)
    ax.axhspan(hi_smooth.min() - 10, cfg.HI_MODERATE_DEHYDRATION,
               color=cfg.COLORS["hydration_red"], alpha=0.08)

    # Trend arrows at key points
    slopes = results["slope"].values
    n_pts = len(slopes)
    arrow_positions = [n_pts // 4, n_pts // 2, 3 * n_pts // 4]
    for pos in arrow_positions:
        if pos < n_pts and not np.isnan(slopes[pos]):
            direction = results["trend_direction"].iloc[pos]
            ax.annotate(direction,
                        xy=(t_min.iloc[pos], hi_smooth[pos]),
                        xytext=(t_min.iloc[pos], hi_smooth[pos] + 8),
                        fontsize=9, fontweight="bold",
                        color=cfg.COLORS["trend_arrow"],
                        ha="center",
                        arrowprops=dict(arrowstyle="->",
                                        color=cfg.COLORS["trend_arrow"],
                                        lw=1.5))

    # Legend for zones
    legend_patches = [
        mpatches.Patch(color=cfg.COLORS["hydration_green"], alpha=0.5,
                       label="Fully Hydrated (HI ≥ 0%)"),
        mpatches.Patch(color=cfg.COLORS["hydration_yellow"], alpha=0.5,
                       label="Mild Dehydration (-10% ≤ HI < 0%)"),
        mpatches.Patch(color=cfg.COLORS["hydration_orange"], alpha=0.5,
                       label="Moderate Dehydration (-25% ≤ HI < -10%)"),
        mpatches.Patch(color=cfg.COLORS["hydration_red"], alpha=0.5,
                       label="Extreme Dehydration (HI < -25%)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8,
              framealpha=0.4, labelcolor=cfg.COLORS["text"])

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Hydration Index (%)")
    ax.set_title("③ Hydration Index Trend — Smoothed with EWMA",
                 fontsize=12, fontweight="bold", pad=10)

    # ─── Panel 4: Heart Rate & HRV ────────────────────────────────
    ax = axes[3]
    ax4_twin = ax.twinx()

    ax.plot(t_min, results["hr_bpm"],
            color=cfg.COLORS["hydration_red"], linewidth=1.5,
            label="Heart Rate (bpm)", alpha=0.9)
    if "hrv_sdnn" in results.columns:
        ax4_twin.plot(t_min, results["hrv_sdnn"],
                      color=cfg.COLORS["hydration_green"], linewidth=1.5,
                      label="HRV (SDNN)", linestyle="--", alpha=0.9)

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Heart Rate (bpm)", color=cfg.COLORS["hydration_red"])
    ax4_twin.set_ylabel("HRV SDNN (s)", color=cfg.COLORS["hydration_green"])
    ax4_twin.tick_params(axis="y", colors=cfg.COLORS["hydration_green"])
    ax.set_title("④ Heart Rate & Heart Rate Variability",
                 fontsize=12, fontweight="bold", pad=10)

    lns_hr = ax.get_lines() + ax4_twin.get_lines()
    labs_hr = [l.get_label() for l in lns_hr]
    ax.legend(lns_hr, labs_hr, loc="upper right", fontsize=9,
              framealpha=0.3, labelcolor=cfg.COLORS["text"])

    # ─── Panel 5: Summary Statistics ───────────────────────────────
    ax = axes[4]
    ax.axis("off")

    kendall = hydration["overall_kendall"]
    baseline = hydration["baseline"]
    phase_trends = hydration["phase_trends"]

    summary_lines = [
        f"Overall Trend:  {kendall['trend']}   (Kendall τ = {kendall['tau']},  p = {kendall['p_value']})",
        f"Baseline PI: {baseline['perfusion_index']:.2f}%    |    "
        f"Baseline HR: {baseline['hr_bpm']:.1f} bpm    |    "
        f"Baseline b/a: {baseline['ba_ratio']:.3f}",
        "Phase Trends:  " + "  •  ".join(
            [f"{pt['phase']}: {pt['trend']}" for pt in phase_trends]
        ),
        f"Final Hydration Level: {results['hydration_level'].iloc[-1]}    |    "
        f"Final HI: {results['hi_smoothed'].iloc[-1]:.1f}%",
    ]

    for i, line in enumerate(summary_lines):
        ax.text(0.02, 0.85 - i * 0.25, line,
                transform=ax.transAxes, fontsize=11,
                color=cfg.COLORS["text"], fontfamily="monospace",
                verticalalignment="top")

    ax.set_title("⑤ Session Summary",
                 fontsize=12, fontweight="bold", pad=10,
                 color=cfg.COLORS["text"])

    # ─── Save ──────────────────────────────────────────────────────
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = os.path.join(cfg.OUTPUT_DIR, "hydration_trend_report.png")
    fig.savefig(out_path, dpi=cfg.FIGURE_DPI, facecolor=cfg.COLORS["background"],
                bbox_inches="tight")
    plt.close(fig)

    print(f"\n{'═' * 60}")
    print(f"  ✅  Report saved  →  {os.path.abspath(out_path)}")
    print(f"{'═' * 60}\n")

    return os.path.abspath(out_path)
