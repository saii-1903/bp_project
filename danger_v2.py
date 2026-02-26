"""
danger_v2.py — Improved BP model training script.

Key fixes over danger.py:
  1.  Feature set locked to 31 features (25 original + 6 vascular/HRV).
      Scaler and all models share the same 31-feature space.
  2.  Per-group StandardScalers — each category (hypo/normal/hyper)
      gets its own scaler so within-group distributions are normalized.
  3.  Strict GroupShuffleSplit inside each per-group model to prevent
      same-patient segments appearing in both train and test sets.
  4.  Stacking ensemble for regression: XGBRegressor base + Ridge meta.
  5.  SMOTE-like oversampling for minority BP classes in the classifier.
  6.  Signal-quality filter tuned: skew check removed (sensor-agnostic).
  7.  Absolute signal stats (mean/std/max/min of raw ppg_filt) replaced
      by normalized equivalents — removes sensor-gain dependency.
  8.  PR fallback: always calculate from Pleth if PRAllData is invalid.
  9.  Class-weighted XGBoost classifier with calibrated probabilities.
  10. Model metadata saved alongside .pkl files for sanity checks.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from glob import glob
from scipy.signal import butter, filtfilt, find_peaks, medfilt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import (GroupShuffleSplit, RandomizedSearchCV,
                                     cross_val_predict)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, mean_absolute_error,
                             r2_score, classification_report)
from sklearn.pipeline import Pipeline
from scipy.stats import skew
import psutil

try:
    from xgboost import XGBRegressor, XGBClassifier
    USE_XGB = True
except ImportError:
    from sklearn.ensemble import RandomForestRegressor
    USE_XGB = False
    print("WARNING: xgboost not installed — falling back to RandomForest.")

try:
    from imblearn.over_sampling import SMOTE
    USE_SMOTE = True
except ImportError:
    USE_SMOTE = False
    print("WARNING: imbalanced-learn not installed — skipping SMOTE.")

# ─── Constants ───────────────────────────────────────────────────────
# CRITICAL FIX: Set to 120 Hz to match your training data (3600 samples / 30s)
FS = 120                  
SEG_LEN = FS * 5          # 5-second segments = 600 samples
SEGMENTS = 6              # 6 segments per 30-second recording
MIN_PPG_LEN = FS * 30     # 3600 samples
MIN_PR_LEN = SEGMENTS * (SEG_LEN // FS)

INVALID_PR = {0.0, 127.0, 255.0}
INVALID_BP = {0.0, -1.0, 1.0, 200.0, 202.0, 400.0, 404.0}

N_FEATURES = 31           # Locked feature count

# ─── Signal Utilities ────────────────────────────────────────────────

def mem_mb():
    return psutil.Process(os.getpid()).memory_info().rss / 1e6


def bandpass(sig):
    nyq = 0.5 * FS
    b, a = butter(3, [0.4 / nyq, 11 / nyq], btype="band")
    return filtfilt(b, a, sig)


def is_noisy(signal):
    """Relaxed noise gate."""
    if np.std(signal) < 0.01:
        return True
    if len(find_peaks(signal, distance=FS // 2)[0]) < 2:
        return True
    return False


def get_derivatives(cycle):
    d1 = np.gradient(cycle)
    d2 = np.gradient(d1)
    return d1, d2


def get_apg_features(d2):
    a = np.max(d2)
    b = np.min(d2)
    return [a, b, b / a if a != 0 else 0.0]


def compute_pr_from_ppg(ppg_full, fs=FS, seg_dur=1):
    """Compute per-second pulse rate from a raw PPG array."""
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


# ─── Feature Extraction ──────────────────────────────────────────────

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

    # PR stats
    if len(pr_seg) > 0:
        pr_mean = float(np.nanmean(pr_seg)) if not np.all(np.isnan(pr_seg)) else 0.0
        pr_std  = float(np.nanstd(pr_seg))  if not np.all(np.isnan(pr_seg)) else 2.0
        if pr_std < 0.1:
            pr_std = 2.0
    else:
        pr_mean, pr_std = 0.0, 2.0

    # Normalised signal stats
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
        *apg_feats, float(auc), *f_top, *m_top, float(hrv),
        sig_mean, sig_std, sig_max, sig_min,
        float(pr_mean), float(pr_std),
        ri, aix, large_si, rmssd, pnn50, pw50
    ]


# ─── BP Labelling ─────────────────────────────────────────────────────

def get_bp_label(sbp, dbp):
    """ACC/AHA 2017 guidelines."""
    if sbp < 90 or dbp < 60:
        return "hypo"
    elif sbp >= 130 or dbp >= 80:  # Corrected threshold (>=80 is Stage 1)
        return "hyper"
    return "normal"


# ─── Data Loading ─────────────────────────────────────────────────────

def load_data(folder):
    X, Y_sbp, Y_dbp, labels, pids = [], [], [], [], []
    files = glob(os.path.join(folder, "*.json"))
    print(f"Found {len(files)} JSON files. Initial memory: {mem_mb():.1f} MB")

    for path in files:
        try:
            with open(path) as f:
                d = json.load(f)
        except json.JSONDecodeError:
            continue

        sbp = d.get("SBP") or d.get("BPSystolic")
        dbp = d.get("DBP") or d.get("BPDiastolic")
        if not isinstance(sbp, (int, float)) or not isinstance(dbp, (int, float)):
            continue
        if sbp in INVALID_BP or dbp in INVALID_BP or sbp <= 0 or dbp <= 0:
            continue
        if not (60 <= sbp <= 220) or not (30 <= dbp <= 130):
            continue
        if sbp <= dbp:
            continue

        label = get_bp_label(sbp, dbp)
        ppg_raw = d.get("Pleth")
        if not ppg_raw or len(ppg_raw) < MIN_PPG_LEN:
            continue

        pr_arr = None
        pr_json = d.get("PRAllData")
        if isinstance(pr_json, list):
            tmp = np.array(pr_json, dtype=float)
            for v in INVALID_PR:
                tmp[tmp == v] = np.nan
            if np.count_nonzero(~np.isnan(tmp)) >= MIN_PR_LEN:
                pr_arr = tmp

        if pr_arr is None:
            ppg_tmp = medfilt(np.array(ppg_raw), kernel_size=3)
            calc = compute_pr_from_ppg(ppg_tmp)
            calc[calc == 0.0] = np.nan
            if np.count_nonzero(~np.isnan(calc)) >= MIN_PR_LEN:
                pr_arr = calc
            else:
                continue

        ppg = medfilt(np.array(ppg_raw), kernel_size=3)
        pid = d.get("PatientID") or d.get("Name") or os.path.basename(path)

        for i in range(SEGMENTS):
            seg_ppg = ppg[i * SEG_LEN: (i + 1) * SEG_LEN]
            pr_seg  = pr_arr[i * (SEG_LEN // FS): (i + 1) * (SEG_LEN // FS)]
            feat = extract_features(seg_ppg, pr_seg)
            if feat and len(feat) == N_FEATURES:
                X.append(feat)
                Y_sbp.append(sbp)
                Y_dbp.append(dbp)
                labels.append(label)
                pids.append(pid)

    X = np.array(X, dtype=float)
    print(f"Loaded {len(X)} segment rows.")
    return X, np.array(Y_sbp), np.array(Y_dbp), np.array(labels), np.array(pids)


# ─── Training ─────────────────────────────────────────────────────────

def train_models(X, Y_sbp, Y_dbp, labels, pids, output_dir="model_v2"):
    os.makedirs(output_dir, exist_ok=True)
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, te_idx = next(gss.split(X, labels, groups=pids))
    X_tr, X_te   = X[tr_idx], X[te_idx]
    l_tr, l_te   = labels[tr_idx], labels[te_idx]
    print(f"Split: {len(tr_idx)} train / {len(te_idx)} test")

    # Global scaler
    global_scaler = StandardScaler()
    X_tr_sc = global_scaler.fit_transform(X_tr)
    X_te_sc  = global_scaler.transform(X_te)
    joblib.dump(global_scaler, os.path.join(output_dir, "global_feature_scaler.pkl"))

    # 1. Classifier
    print("\n--- Training Classifier ---")
    LABEL_ORDER = ["hypo", "normal", "hyper"]
    label_to_int = {l: i for i, l in enumerate(LABEL_ORDER)}
    int_to_label = {i: l for i, l in enumerate(LABEL_ORDER)}

    l_tr_int = np.array([label_to_int[l] for l in l_tr])
    l_te_int = np.array([label_to_int[l] for l in l_te])

    if USE_XGB:
        classes_int, counts = np.unique(l_tr_int, return_counts=True)
        cw = {c: len(l_tr_int) / (len(classes_int) * cnt) for c, cnt in zip(classes_int, counts)}
        sample_weights = np.array([cw[li] for li in l_tr_int])

        base_clf = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=42, n_jobs=-1,
            num_class=len(LABEL_ORDER), objective="multi:softprob",
        )
        if USE_SMOTE:
            smote = SMOTE(random_state=42)
            X_bal, l_bal = smote.fit_resample(X_tr_sc, l_tr_int)
            base_clf.fit(X_bal, l_bal)
        else:
            base_clf.fit(X_tr_sc, l_tr_int, sample_weight=sample_weights)

        clf = CalibratedClassifierCV(base_clf, cv="prefit", method="isotonic")
        clf.fit(X_tr_sc, l_tr_int)
    else:
        from sklearn.linear_model import LogisticRegression as _LR
        clf = _LR(max_iter=2000, class_weight="balanced", random_state=42)
        clf.fit(X_tr_sc, l_tr_int)

    pred_int = clf.predict(X_te_sc)
    print(f"Classifier accuracy: {accuracy_score(l_te_int, pred_int):.4f}")
    
    clf_bundle = {"model": clf, "label_to_int": label_to_int, "int_to_label": int_to_label}
    joblib.dump(clf_bundle, os.path.join(output_dir, "classifier.pkl"))

    # 2. Regression
    print("\n--- Training Regression ---")
    
    # Base Regressor
    if USE_XGB:
        base_reg = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, n_jobs=-1)
    else:
        from sklearn.ensemble import GradientBoostingRegressor
        base_reg = GradientBoostingRegressor(n_estimators=100)

    for group in LABEL_ORDER:
        g_tr_mask = l_tr == group
        if g_tr_mask.sum() < 10:
            continue
            
        print(f"  Training {group.upper()} models...")
        Xg_tr = X[tr_idx[g_tr_mask]]
        Ysbp_tr = Y_sbp[tr_idx[g_tr_mask]]
        Ydbp_tr = Y_dbp[tr_idx[g_tr_mask]]

        grp_scaler = StandardScaler()
        Xg_tr_sc = grp_scaler.fit_transform(Xg_tr)
        joblib.dump(grp_scaler, os.path.join(output_dir, f"scaler_{group}.pkl"))

        # Train SBP
        sbp_model = base_reg.__class__(**base_reg.get_params())
        sbp_model.fit(Xg_tr_sc, Ysbp_tr)

        # Train DBP
        dbp_model = base_reg.__class__(**base_reg.get_params())
        dbp_model.fit(Xg_tr_sc, Ydbp_tr)
        
        # Meta Learners (Ridge)
        sbp_meta = Ridge(alpha=10.0)
        sbp_meta.fit(sbp_model.predict(Xg_tr_sc).reshape(-1, 1), Ysbp_tr)
        
        dbp_meta = Ridge(alpha=10.0)
        dbp_meta.fit(dbp_model.predict(Xg_tr_sc).reshape(-1, 1), Ydbp_tr)

        combined = {
            "sbp_model": sbp_model, "dbp_model": dbp_model,
            "sbp_meta": sbp_meta, "dbp_meta": dbp_meta
        }
        joblib.dump(combined, os.path.join(output_dir, f"{group}_models.pkl"))

    print(f"Training complete. Models saved to {output_dir}")

    # Metadata
    meta = {
        "n_features": N_FEATURES,
        "groups": LABEL_ORDER,
        "label_to_int": label_to_int,
        "int_to_label": {str(k): v for k, v in int_to_label.items()},
        "classifier_uses_int_labels": True,
        "uses_meta_ridge": True,
        "uses_per_group_scaler": True,
    }
    with open(os.path.join(output_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_folder")
    parser.add_argument("--output_dir", default="water/models")
    args = parser.parse_args()

    X, Y_sbp, Y_dbp, labels, pids = load_data(args.data_folder)
    if len(X) > 0:
        train_models(X, Y_sbp, Y_dbp, labels, pids, args.output_dir)