"""
main.py -- PPG Hydration Trend Discovery Pipeline Orchestrator.

Wires together every module to run the complete analysis:
  1.  Generate (or load) raw PPG data
  2.  Apply signal processing (filter + baseline correction)
  3.  Extract hydration-relevant features
  4.  Compute Hydration Index and detect trends
  5.  Generate publication-quality graphs
  6.  Print a console summary
"""

import sys
import os
import time
import numpy as np
import pandas as pd

# Force UTF-8 output on Windows
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import config as cfg
from ppg_simulator import generate_ppg
from signal_processing import preprocess
from feature_extraction import extract_features
from hydration_engine import analyse_hydration
from visualization import generate_all


def _banner():
    print("")
    print("  +==================================================================+")
    print("  |        PPG-Based Hydration Trend Discovery Pipeline              |")
    print("  |   Non-Invasive - 60 Hz Photoplethysmography - Trend Analysis     |")
    print("  +==================================================================+")
    print("")


def _print_section(title: str):
    print(f"\n  > {title}")
    print(f"    {'-' * (len(title) + 2)}")


def main():
    _banner()
    t0 = time.time()

    # -- Step 1: Generate synthetic PPG data -------------------------
    _print_section("Step 1 / 5 -- Generating synthetic PPG signal")
    sim = generate_ppg(duration_minutes=cfg.SIMULATION_DURATION_MINUTES, seed=42)
    n_samples = len(sim["raw_ppg"])
    duration_min = n_samples / sim["fs"] / 60
    print(f"    Sampling rate   : {sim['fs']} Hz")
    print(f"    Total samples   : {n_samples:,}")
    print(f"    Duration        : {duration_min:.1f} minutes")
    print(f"    Hydration range : {sim['hydration_factor'].min():.2f} - "
          f"{sim['hydration_factor'].max():.2f}")

    # -- Step 2: Signal processing -----------------------------------
    _print_section("Step 2 / 5 -- Pre-processing (filter + baseline correction)")
    proc = preprocess(sim["raw_ppg"], sim["fs"])
    n_peaks = len(proc["peaks"])
    good_beats = proc["sqi"].sum() if len(proc["sqi"]) > 0 else 0
    print(f"    Peaks detected  : {n_peaks}")
    print(f"    Good-quality    : {good_beats} / {n_peaks} "
          f"({100 * good_beats / max(n_peaks, 1):.1f}%)")

    # -- Step 3: Feature extraction ----------------------------------
    _print_section("Step 3 / 5 -- Extracting hydration-relevant features")
    features = extract_features(
        corrected=proc["corrected"],
        raw_ppg=sim["raw_ppg"],
        peaks=proc["peaks"],
        valleys=proc["valleys"],
        sqi=proc["sqi"],
        fs=sim["fs"],
    )
    if features.empty:
        print("    [!] No valid feature windows -- aborting.")
        sys.exit(1)

    print(f"    Feature windows : {len(features)}")
    print(f"    Window size     : {cfg.FEATURE_WINDOW_SECONDS}s  "
          f"({cfg.FEATURE_WINDOW_OVERLAP * 100:.0f}% overlap)")
    print(f"    Median PI       : {features['perfusion_index'].median():.2f}%")
    print(f"    Median HR       : {features['hr_bpm'].median():.1f} bpm")
    print(f"    Median b/a      : {features['ba_ratio'].median():.3f}")

    # -- Step 4: Hydration analysis ----------------------------------
    _print_section("Step 4 / 5 -- Computing Hydration Index & detecting trends")
    hydration = analyse_hydration(features)
    res = hydration["results"]
    kendall = hydration["overall_kendall"]

    print(f"    Baseline PI     : {hydration['baseline']['perfusion_index']:.2f}%")
    print(f"    HI range        : {res['hi_smoothed'].min():.1f}% -> "
          f"{res['hi_smoothed'].max():.1f}%")
    print(f"    Overall trend   : {kendall['trend']}")
    print(f"    Kendall tau     : {kendall['tau']}")
    print(f"    Kendall p-value : {kendall['p_value']}")
    print(f"    Final status    : {res['hydration_level'].iloc[-1]} "
          f"(HI = {res['hi_smoothed'].iloc[-1]:.1f}%)")

    # Phase-by-phase breakdown
    print("\n    Phase breakdown:")
    for pt in hydration["phase_trends"]:
        print(f"      {pt['phase']:>5s}:  {pt['trend']}")

    # -- Step 5: Visualisation ---------------------------------------
    _print_section("Step 5 / 5 -- Generating graphical report")
    report_path = generate_all(sim, proc, features, hydration)

    elapsed = time.time() - t0
    print(f"\n  [OK] Pipeline completed in {elapsed:.2f} seconds.")
    print(f"  [OK] Open the report: {report_path}\n")


if __name__ == "__main__":
    main()

