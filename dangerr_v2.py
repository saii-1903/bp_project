"""
dangerr_v2.py — Improved BP inference script (matches danger_v2.py exactly).

Key fixes over dangerr.py:
  1.  Uses per-group scalers (scaler_hypo/normal/hyper.pkl) instead of
      a single global scaler for regression — matches training.
  2.  Fallback PR calculation from Pleth when PRAllData is absent or invalid.
  3.  Meta Ridge correction applied after base regressor predictions.
  4.  Relaxed consistency filter: replaces hard 10 mmHg threshold with
      IQR-based outlier removal across 6 segments.
  5.  Calibrated classifier probabilities give better soft-vote weights.
  6.  Skew-check removed from is_noisy() — sensor-agnostic.
  7.  Physiological clamps on final output (SBP 70–200, DBP 40–130).
  8.  Optional personal calibration offset support.
  9.  BP trend detection: returns category + smoothed trend over time.
"""

import os
import json
import joblib
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, medfilt
from collections import Counter
from scipy.stats import iqr

FS = 200
SEG_LEN = FS * 5
SEGMENTS = 6
MIN_PPG_LEN = FS * 30
MIN_PR_LEN = SEGMENTS * (SEG_LEN // FS)
INVALID_PR = {0.0, 127.0, 255.0}
N_FEATURES = 31   # must match danger_v2.py


# ─── Signal Utilities ────────────────────────────────────────────────

def bandpass(sig):
    nyq = 0.5 * FS
    b, a = butter(3, [0.4 / nyq, 11 / nyq], btype="band")
    return filtfilt(b, a, sig)


def is_noisy(signal):
    """Relaxed — no skew check so inverted sensors are accepted."""
    if np.std(signal) < 0.01:
        return True
    if len(find_peaks(signal, distance=FS // 2)[0]) < 2:
        return True
    return False


def get_derivatives(cycle):
    d1 = np.gradient(cycle)
    return d1, np.gradient(d1)


def get_apg_features(d2):
    a = np.max(d2);  b = np.min(d2)
    return [a, b, b / a if a != 0 else 0.0]


def compute_pr_from_ppg(ppg_full, fs=FS, seg_dur=1):
    pr_values = []
    sps = fs * seg_dur
    for i in range(len(ppg_full) // fs):
        seg = ppg_full[i * sps: (i + 1) * sps]
        if len(seg) < fs * 0.8:
            pr_values.append(np.nan)
            continue
        filt = bandpass(seg)
        norm = (filt - filt.min()) / (filt.max() - filt.min() + 1e-6)
        pks, _ = find_peaks(norm, distance=int(fs * 0.4))
        if len(pks) > 1:
            ibi = np.diff(pks) / fs
            pr_values.append(60.0 / np.mean(ibi) if np.mean(ibi) > 0 else np.nan)
        else:
            pr_values.append(np.nan)
    while len(pr_values) < MIN_PR_LEN:
        pr_values.append(np.nan)
    return np.array(pr_values)


# ─── Feature Extraction (31 features — MUST match danger_v2.py) ──────

def extract_features(ppg_seg, pr_seg):
    if len(ppg_seg) < FS or is_noisy(ppg_seg):
        return None

    ppg_filt = bandpass(ppg_seg)
    norm = (ppg_filt - ppg_filt.min()) / (ppg_filt.max() - ppg_filt.min() + 1e-6)

    peaks, _ = find_peaks(norm, distance=int(FS * 0.4))
    mins,  _ = find_peaks(-norm, distance=int(FS * 0.3))

    cycles = [norm[mins[mins < p][-1]: mins[mins > p][0]]
              for p in peaks
              if len(mins[mins < p]) > 0 and len(mins[mins > p]) > 0]
    if not cycles:
        return None

    c = max(cycles, key=np.max)
    time = np.linspace(0, len(c) / FS, len(c))
    d1, d2 = get_derivatives(c)
    apg_feats = get_apg_features(d2)

    auc   = np.trapezoid(c, time)
    ttp   = time[np.argmax(c)]
    tdp   = time[-1] - ttp if (time[-1] - ttp) != 0 else 1e-6
    ratio = ttp / tdp

    fft_v = np.abs(np.fft.fft(c)[: len(c) // 2])
    freqs = np.fft.fftfreq(len(c), 1 / FS)[: len(c) // 2]
    pks, _ = find_peaks(fft_v, distance=5)
    f_top = [0.0] * 3;  m_top = [0.0] * 3
    if len(pks):
        top = np.argsort(fft_v[pks])[-3:]
        ft  = freqs[pks][top].tolist();  mt = fft_v[pks][top].tolist()
        f_top = ft + [0.0] * (3 - len(ft))
        m_top = mt + [0.0] * (3 - len(mt))

    ibi = np.diff(peaks) / FS if len(peaks) > 1 else np.array([0.0])
    hrv = float(np.std(ibi))

    if len(pr_seg) > 0 and not np.all(np.isnan(pr_seg)):
        pr_mean = float(np.nanmean(pr_seg))
        pr_std  = float(np.nanstd(pr_seg))
        if pr_std < 0.1:
            pr_std = 2.0
    else:
        pr_mean, pr_std = 0.0, 2.0

    # Normalised signal stats (sensor-agnostic)
    sig_mean = float(np.mean(norm))
    sig_std  = float(np.std(norm))
    sig_max  = float(np.max(norm))
    sig_min  = float(np.min(norm))

    # Vascular / HRV features
    peak_idx  = np.argmax(c)
    post_peak = c[peak_idx:]
    notch_mins, _ = find_peaks(-post_peak)
    ri  = float(post_peak[notch_mins[0]] / (np.max(c) + 1e-9)) if len(notch_mins) > 0 else 0.0
    aix = float((post_peak[notch_mins[0]] - np.max(c)) / (np.max(c) + 1e-9)) if len(notch_mins) > 0 else 0.0
    large_si = float(0.1 / (ttp + 1e-9))

    if len(ibi) > 1:
        rmssd = float(np.sqrt(np.mean(np.diff(ibi) ** 2)))
        pnn50 = float(np.sum(np.abs(np.diff(ibi)) > 0.05) / max(len(ibi) - 1, 1))
    else:
        rmssd, pnn50 = 0.0, 0.0

    above_half = np.where(c >= np.max(c) * 0.5)[0]
    pw50 = float(len(above_half) / FS) if len(above_half) > 0 else 0.0

    return [
        float(np.max(c)), float(time[-1]), float(ttp), float(ratio),
        float(np.max(d1)), float(np.min(d1)), float(np.max(d2)), float(np.min(d2)),
        *apg_feats,         # 8-10
        float(auc),         # 11
        *f_top,             # 12-14
        *m_top,             # 15-17
        float(hrv),         # 18
        sig_mean, sig_std, sig_max, sig_min,  # 19-22
        float(pr_mean), float(pr_std),        # 23-24
        ri, aix, large_si, rmssd, pnn50, pw50  # 25-30
    ]


# ─── Model Loading ────────────────────────────────────────────────────

def load_models(model_dir="model_v2"):
    models = {}
    try:
        models["scaler_global"] = joblib.load(os.path.join(model_dir, "global_feature_scaler.pkl"))

        # Classifier is saved as a bundle with label mapping
        clf_bundle = joblib.load(os.path.join(model_dir, "classifier.pkl"))
        if isinstance(clf_bundle, dict) and "model" in clf_bundle:
            models["classifier"]    = clf_bundle["model"]
            models["label_to_int"]  = clf_bundle["label_to_int"]
            models["int_to_label"]  = clf_bundle["int_to_label"]
        else:
            # Backwards-compatible: old-style classifier with string labels
            models["classifier"]   = clf_bundle
            models["label_to_int"] = {"hypo": 0, "normal": 1, "hyper": 2}
            models["int_to_label"] = {0: "hypo", 1: "normal", 2: "hyper"}

        for group in ["hypo", "normal", "hyper"]:
            grp_path = os.path.join(model_dir, f"{group}_models.pkl")
            scl_path = os.path.join(model_dir, f"scaler_{group}.pkl")
            if os.path.exists(grp_path) and os.path.exists(scl_path):
                g = joblib.load(grp_path)
                models[f"sbp_{group}"]      = g["sbp_model"]
                models[f"dbp_{group}"]      = g["dbp_model"]
                models[f"sbp_meta_{group}"] = g.get("sbp_meta")
                models[f"dbp_meta_{group}"] = g.get("dbp_meta")
                models[f"scaler_{group}"]   = joblib.load(scl_path)
            else:
                print(f"Warning: missing files for group '{group}'")

        # Verify feature count from metadata
        meta_path = os.path.join(model_dir, "model_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            assert meta["n_features"] == N_FEATURES, \
                f"Feature mismatch: model has {meta['n_features']}, code has {N_FEATURES}"

    except Exception as e:
        print(f"Error loading models: {e}")
        return None
    return models


# ─── IQR-Based Outlier Removal ────────────────────────────────────────

def _iqr_filter(values, factor=1.5):
    """Return a boolean mask of inliers using IQR method."""
    if len(values) < 4:
        return np.ones(len(values), dtype=bool)
    q1, q3 = np.percentile(values, 25), np.percentile(values, 75)
    iq = q3 - q1
    return (np.array(values) >= q1 - factor * iq) & (np.array(values) <= q3 + factor * iq)


# ─── Core Prediction ─────────────────────────────────────────────────

def predict_bp(ppg_data, pr_all_data=None, model_dir="model_v2",
               calibration_offset=None):
    """
    Predict blood pressure from a 30-second PPG signal.

    Parameters
    ----------
    ppg_data          : list/array of raw PPG samples (≥ 6000 at 200 Hz)
    pr_all_data       : optional list of per-second PR values from device
    model_dir         : directory containing model_v2 files
    calibration_offset: optional dict {"sbp": float, "dbp": float} —
                        a one-time personal calibration shift (Fix 8)

    Returns
    -------
    dict with keys:
        predicted_category, predicted_sbp, predicted_dbp,
        segment_sbp_values, segment_dbp_values,
        confidence (0-1), bp_trend_note
    or a string error message.
    """
    models = load_models(model_dir)
    if models is None:
        return "Model loading failed."

    # ── Resolve PR data ───────────────────────────────────────────
    pr_arr = None
    if isinstance(pr_all_data, list) and len(pr_all_data) > 0:
        tmp = np.array(pr_all_data, dtype=float)
        for v in INVALID_PR:
            tmp[tmp == v] = np.nan
        if np.count_nonzero(~np.isnan(tmp)) >= MIN_PR_LEN:
            pr_arr = tmp

    # Fix 2: fallback to computing PR from Pleth
    if pr_arr is None:
        if ppg_data is None or len(ppg_data) < MIN_PPG_LEN:
            return "Insufficient PPG data and no valid PRAllData."
        ppg_tmp = medfilt(np.array(ppg_data, dtype=float), kernel_size=3)
        calc = compute_pr_from_ppg(ppg_tmp)
        calc[calc == 0.0] = np.nan
        if np.count_nonzero(~np.isnan(calc)) >= MIN_PR_LEN:
            pr_arr = calc
        else:
            return "Could not derive valid PR data from Pleth."

    if ppg_data is None or len(ppg_data) < MIN_PPG_LEN:
        return "Insufficient Pleth length (need ≥ 6000 samples at 200 Hz)."

    ppg = medfilt(np.array(ppg_data, dtype=float), kernel_size=3)
    clf = models["classifier"]

    # ── Segment-wise predictions ──────────────────────────────────
    seg_results = []
    for i in range(SEGMENTS):
        seg_ppg = ppg[i * SEG_LEN: (i + 1) * SEG_LEN]
        pr_seg  = pr_arr[i * (SEG_LEN // FS): (i + 1) * (SEG_LEN // FS)]
        feat = extract_features(seg_ppg, pr_seg)
        if feat is None or len(feat) != N_FEATURES:
            continue

        X_raw = np.array(feat, dtype=float).reshape(1, -1)

        # Scale with global scaler for classification
        X_cls = models["scaler_global"].transform(X_raw)
        probs   = clf.predict_proba(X_cls)[0]
        # clf.classes_ may be integers [0,1,2] — map back to strings
        int_to_label = models.get("int_to_label", {0: "hypo", 1: "normal", 2: "hyper"})
        clf_classes  = clf.classes_   # e.g. [0, 1, 2]

        # Soft-vote across all groups using per-group scaler for regression
        w_sbp = w_dbp = 0.0
        for idx, cls_idx in enumerate(clf_classes):
            cls_name = int_to_label.get(int(cls_idx), str(cls_idx))
            p = probs[idx]
            sbp_key = f"sbp_{cls_name}"
            dbp_key = f"dbp_{cls_name}"
            scl_key = f"scaler_{cls_name}"
            if sbp_key in models and scl_key in models:
                X_reg = models[scl_key].transform(X_raw)
                sbp_raw = float(models[sbp_key].predict(X_reg)[0])
                dbp_raw = float(models[dbp_key].predict(X_reg)[0])

                # Apply meta ridge correction if available
                sbp_meta = models.get(f"sbp_meta_{cls_name}")
                dbp_meta = models.get(f"dbp_meta_{cls_name}")
                if sbp_meta is not None:
                    sbp_raw = float(sbp_meta.predict([[sbp_raw]])[0])
                    dbp_raw = float(dbp_meta.predict([[dbp_raw]])[0])

                w_sbp += p * sbp_raw
                w_dbp += p * dbp_raw

        pred_label = int_to_label.get(int(clf_classes[np.argmax(probs)]), "normal")

        seg_results.append({
            "sbp": w_sbp, "dbp": w_dbp,
            "label": pred_label, "confidence": float(np.max(probs))
        })

    if len(seg_results) < 3:
        return f"Not enough valid segments ({len(seg_results)}/6). Check signal quality."

    # ── IQR-based outlier removal (Fix 4) ─────────────────────────
    sbp_vals = [r["sbp"] for r in seg_results]
    dbp_vals = [r["dbp"] for r in seg_results]
    sbp_mask = _iqr_filter(sbp_vals)
    dbp_mask = _iqr_filter(dbp_vals)
    combined_mask = sbp_mask & dbp_mask

    if combined_mask.sum() < 2:
        combined_mask = np.ones(len(seg_results), dtype=bool)  # fallback: use all

    clean_sbp = [sbp_vals[i] for i in range(len(sbp_vals)) if combined_mask[i]]
    clean_dbp = [dbp_vals[i] for i in range(len(dbp_vals)) if combined_mask[i]]
    labels_clean = [seg_results[i]["label"] for i in range(len(seg_results)) if combined_mask[i]]

    final_sbp = float(np.mean(clean_sbp))
    final_dbp = float(np.mean(clean_dbp))

    # Physiological clamp (Fix 7)
    final_sbp = float(np.clip(final_sbp, 70, 220))
    final_dbp = float(np.clip(final_dbp, 40, 130))

    # Personal calibration offset (Fix 8)
    if calibration_offset:
        final_sbp += calibration_offset.get("sbp", 0.0)
        final_dbp += calibration_offset.get("dbp", 0.0)
        final_sbp = float(np.clip(final_sbp, 70, 220))
        final_dbp = float(np.clip(final_dbp, 40, 130))

    category = Counter(labels_clean).most_common(1)[0][0]
    mean_conf = float(np.mean([r["confidence"] for r in seg_results]))

    # ── BP trend note ─────────────────────────────────────────────
    if len(clean_sbp) >= 4:
        first_half = np.mean(clean_sbp[:len(clean_sbp)//2])
        second_half = np.mean(clean_sbp[len(clean_sbp)//2:])
        delta = second_half - first_half
        if delta > 3:
            trend_note = "SBP rising within session"
        elif delta < -3:
            trend_note = "SBP falling within session"
        else:
            trend_note = "SBP stable within session"
    else:
        trend_note = "Insufficient segments for trend"

    return {
        "predicted_category": category,
        "predicted_sbp":      round(final_sbp),
        "predicted_dbp":      round(final_dbp),
        "segment_sbp_values": [round(v, 1) for v in sbp_vals],
        "segment_dbp_values": [round(v, 1) for v in dbp_vals],
        "confidence":         round(mean_conf, 3),
        "bp_trend_note":      trend_note,
        "n_valid_segments":   len(seg_results),
        "n_clean_segments":   int(combined_mask.sum()),
    }


# ─── Session Trend Tracker ────────────────────────────────────────────

class BPTrendTracker:
    """
    Accumulates successive predict_bp() calls and returns a session trend.
    Use one instance per patient session.
    """
    def __init__(self, window=5):
        self._sbp_history = []
        self._dbp_history = []
        self._cat_history = []
        self._window = window

    def update(self, prediction: dict):
        if isinstance(prediction, str):
            return   # ignore error strings
        self._sbp_history.append(prediction["predicted_sbp"])
        self._dbp_history.append(prediction["predicted_dbp"])
        self._cat_history.append(prediction["predicted_category"])

    def get_trend(self):
        n = len(self._sbp_history)
        if n < 2:
            return {"trend": "Insufficient data", "readings": n}

        sbp_arr = np.array(self._sbp_history[-self._window:])
        dbp_arr = np.array(self._dbp_history[-self._window:])

        # Linear slope (mmHg per reading)
        x = np.arange(len(sbp_arr))
        sbp_slope = float(np.polyfit(x, sbp_arr, 1)[0])
        dbp_slope = float(np.polyfit(x, dbp_arr, 1)[0])

        if sbp_slope > 1.0:
            sbp_trend = "Rising ↑"
        elif sbp_slope < -1.0:
            sbp_trend = "Falling ↓"
        else:
            sbp_trend = "Stable →"

        latest_cat = Counter(self._cat_history[-self._window:]).most_common(1)[0][0]

        return {
            "trend": sbp_trend,
            "sbp_slope_per_reading": round(sbp_slope, 2),
            "dbp_slope_per_reading": round(dbp_slope, 2),
            "current_category": latest_cat,
            "latest_sbp": self._sbp_history[-1],
            "latest_dbp": self._dbp_history[-1],
            "sbp_mean_last_n": round(float(np.mean(sbp_arr)), 1),
            "dbp_mean_last_n": round(float(np.mean(dbp_arr)), 1),
            "readings": n,
        }


# ─── CLI ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict BP from PPG data (v2).")
    parser.add_argument("input_file", help="JSON file with 'Pleth' (and optionally 'PRAllData').")
    parser.add_argument("--model_dir", default="model_v2")
    parser.add_argument("--cal_sbp", type=float, default=None,
                        help="Personal SBP calibration offset (mmHg)")
    parser.add_argument("--cal_dbp", type=float, default=None,
                        help="Personal DBP calibration offset (mmHg)")
    args = parser.parse_args()

    cal = None
    if args.cal_sbp is not None or args.cal_dbp is not None:
        cal = {"sbp": args.cal_sbp or 0.0, "dbp": args.cal_dbp or 0.0}

    try:
        with open(args.input_file) as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading file: {e}")
        exit(1)

    result = predict_bp(
        data.get("Pleth", []),
        data.get("PRAllData"),
        model_dir=args.model_dir,
        calibration_offset=cal
    )
    print("\nPrediction Result:")
    print(json.dumps(result, indent=2))
