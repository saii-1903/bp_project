# train_model.py
import os
import json
import joblib
import numpy as np
import pandas as pd
from glob import glob
from scipy.signal import butter, filtfilt, find_peaks, medfilt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GroupShuffleSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
import psutil
from scipy.stats import skew
try:
    from xgboost import XGBRegressor
    USE_XGB = True
except ImportError:
    from sklearn.ensemble import RandomForestRegressor
    USE_XGB = False
    print("WARNING: xgboost not installed — falling back to RandomForest. Run: pip install xgboost")

FS = 200 
SEG_LEN = FS * 5 
SEGMENTS = 6 
MIN_PPG_LEN = FS * 30 
MIN_PR_ALL_DATA_LEN = SEGMENTS * (SEG_LEN // FS) 
INVALID_PR_VALUES = {0.0, 127.0, 255.0} 
INVALID_BP_VALUES = {0.0, -1.0, 1.0, 200.0, 202.0, 400.0, 404.0} 

# --- Utility Functions ---

def get_current_memory_usage():
    """Returns current process's Resident Set Size (RSS) memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024) # Convert bytes to MB

def bandpass(sig):
    """
    Applies a Butterworth bandpass filter to the signal.
    Filters between 0.4 Hz and 11 Hz.
    """
    nyq = 0.5 * FS
    b, a = butter(3, [0.4 / nyq, 11 / nyq], btype='band')
    return filtfilt(b, a, sig)

def is_noisy(signal):
    """
    Checks if a signal segment is noisy based on standard deviation,
    insufficient peaks, or negative skewness (inverted signal).
    """
    if np.std(signal) < 0.01: return True
    if len(find_peaks(signal, distance=FS // 2)[0]) < 2: return True
    if skew(signal) < 0: return True # Reject inverted signals
    return False

def get_derivatives(cycle):
    """
    Returns 1st (VPG) and 2nd (APG) derivatives of the cycle.
    """
    d1 = np.gradient(cycle)
    d2 = np.gradient(d1)
    return d1, d2

def get_apg_features(d2):
    """
    Extracts a, b, c, d, e waves from APG (2nd derivative) and ratios.
    Simplified: Uses max/min for a, b, d, e approximations.
    """
    a = np.max(d2)
    b = np.min(d2)
    # Simple approximation for ratios
    b_a_ratio = b / a if a != 0 else 0
    
    # d and e are often subsequent local maxima/minima, simplifying to meaningful stats
    # for robustness if specific fiducial points are hard to detect reliably in noise
    d_a_ratio = 0 # Placeholder if complex peak detection fails
    
    return [a, b, b_a_ratio]

def calculate_pr_from_ppg(ppg_full, sampling_rate=FS, segment_duration=1):
    """
    Calculates pulse rate (PR) for each specified segment_duration (e.g., 1 second)
    from the full PPG signal. Invalid PR values are replaced with NaN.
    """
    pr_values = []
    samples_per_segment = sampling_rate * segment_duration
    total_duration_seconds = len(ppg_full) // sampling_rate

    for i in range(total_duration_seconds):
        start_idx = i * samples_per_segment
        end_idx = start_idx + samples_per_segment
        ppg_segment = ppg_full[start_idx:end_idx]

        if len(ppg_segment) < sampling_rate * 0.8:
            pr_values.append(np.nan) # Use NaN for insufficient data
            continue
        
        filtered_segment = bandpass(ppg_segment)
        norm_segment = (filtered_segment - np.min(filtered_segment)) / (np.max(filtered_segment) - np.min(filtered_segment) + 1e-6)
        
        peaks, _ = find_peaks(norm_segment, distance=int(sampling_rate * 0.4)) 
        
        if len(peaks) > 1:
            ibi_seconds = np.diff(peaks) / sampling_rate
            avg_ibi_seconds = np.mean(ibi_seconds)
            pr_bpm = 60 / avg_ibi_seconds if avg_ibi_seconds > 0 else np.nan
            pr_values.append(pr_bpm)
        else:
            pr_values.append(np.nan) # Use NaN if no or too few peaks

    # Pad with NaN if the original PPG was less than 30 seconds
    while len(pr_values) < MIN_PR_ALL_DATA_LEN:
        pr_values.append(np.nan)

    return np.array(pr_values)

def extract_features(ppg_seg, pr_data_for_segment):
    """
    Extracts time-domain, frequency-domain, morphological, and HRV features
    from a PPG segment. Returns 31 features (25 original + 6 new).
    """
    if len(ppg_seg) < FS or is_noisy(ppg_seg):
        return None

    ppg_filt = bandpass(ppg_seg)
    norm = (ppg_filt - np.min(ppg_filt)) / (np.max(ppg_filt) - np.min(ppg_filt) + 1e-6)
    peaks, _ = find_peaks(norm, distance=int(FS * 0.4))
    mins,  _ = find_peaks(-norm, distance=int(FS * 0.3))

    cycles = [norm[mins[mins < p][-1]:mins[mins > p][0]] for p in peaks
              if len(mins[mins < p]) > 0 and len(mins[mins > p]) > 0]
    if not cycles:
        return None

    c = max(cycles, key=np.max)          # best-quality pulse cycle
    time = np.linspace(0, len(c) / FS, len(c))
    d1, d2 = get_derivatives(c)
    apg_feats = get_apg_features(d2)     # [a, b, b/a]

    auc   = np.trapezoid(c, time)
    ttp   = time[np.argmax(c)]
    tdp   = time[-1] - ttp if (time[-1] - ttp) != 0 else 1e-6
    ratio = ttp / tdp

    fft_vals = np.abs(np.fft.fft(c)[:len(c) // 2])
    freqs    = np.fft.fftfreq(len(c), 1 / FS)[:len(c) // 2]
    pks, _   = find_peaks(fft_vals, distance=5)
    f_top    = [0.0] * 3
    m_top    = [0.0] * 3
    if len(pks) > 0:
        top_idx  = np.argsort(fft_vals[pks])[-3:]
        f_top_t  = freqs[pks][top_idx].tolist()
        m_top_t  = fft_vals[pks][top_idx].tolist()
        f_top    = f_top_t  + [0.0] * (3 - len(f_top_t))
        m_top    = m_top_t  + [0.0] * (3 - len(m_top_t))

    ibi = np.diff(peaks) / FS if len(peaks) > 1 else np.array([0.0])
    hrv = float(np.std(ibi))

    # PR stats
    if len(pr_data_for_segment) > 0:
        pr_mean = float(np.nanmean(pr_data_for_segment))
        pr_std  = float(np.nanstd(pr_data_for_segment))
        if np.isnan(pr_mean): pr_mean = 0.0
        if np.isnan(pr_std) or pr_std < 0.1: pr_std = 2.0
    else:
        pr_mean, pr_std = 0.0, 0.0

    # ── NEW: vascular / HRV features (Fix 4) ─────────────────────────────
    # Reflection Index: amplitude at dicrotic notch / systolic peak
    # Approximate dicrotic notch as the first local min AFTER systolic peak
    peak_idx   = np.argmax(c)
    post_peak  = c[peak_idx:]
    notch_mins, _ = find_peaks(-post_peak)
    ri  = float(post_peak[notch_mins[0]] / (np.max(c) + 1e-9)) if len(notch_mins) > 0 else 0.0

    # Augmentation Index (AIx): (c_notch - c_peak) / c_peak
    aix = float((post_peak[notch_mins[0]] - np.max(c)) / (np.max(c) + 1e-9)) if len(notch_mins) > 0 else 0.0

    # Large Artery Stiffness Index (simplified): 0.1 / time-to-peak
    large_si = float(0.1 / (ttp + 1e-9))

    # RMSSD and pNN50 (beat-to-beat HRV)
    if len(ibi) > 1:
        rmssd = float(np.sqrt(np.mean(np.diff(ibi) ** 2)))
        pnn50 = float(np.sum(np.abs(np.diff(ibi)) > 0.05) / max(len(ibi) - 1, 1))
    else:
        rmssd, pnn50 = 0.0, 0.0

    # Pulse Width at 50% amplitude
    half_max   = np.max(c) * 0.5
    above_half = np.where(c >= half_max)[0]
    pw50 = float(len(above_half) / FS) if len(above_half) > 0 else 0.0
    # ─────────────────────────────────────────────────────────────────────

    return [
        float(np.max(c)),          # 0  systolic amplitude
        float(time[-1]),           # 1  pulse duration
        float(ttp),                # 2  time-to-peak
        float(ratio),              # 3  ttp/tdp ratio
        float(np.max(d1)),         # 4  max VPG
        float(np.min(d1)),         # 5  min VPG
        float(np.max(d2)),         # 6  max APG
        float(np.min(d2)),         # 7  min APG
        *apg_feats,                # 8-10 APG a, b, b/a
        float(auc),                # 11 area under curve
        *f_top,                    # 12-14 top FFT freqs
        *m_top,                    # 15-17 top FFT magnitudes
        float(hrv),                # 18 HRV std
        float(np.mean(ppg_filt)),  # 19 signal mean
        float(np.std(ppg_filt)),   # 20 signal std
        float(np.max(ppg_filt)),   # 21 signal max
        float(np.min(ppg_filt)),   # 22 signal min
        float(pr_mean),            # 23 PR mean
        float(pr_std),             # 24 PR std
        ri,                        # 25 Reflection Index
        aix,                       # 26 Augmentation Index
        large_si,                  # 27 Large Artery Stiffness Index
        rmssd,                     # 28 RMSSD
        pnn50,                     # 29 pNN50
        pw50,                      # 30 Pulse Width at 50%
    ]

def get_bp_label(sbp, dbp):
    """
    Categorizes BP into 'hypo', 'normal', or 'hyper'.
    These thresholds are illustrative and should be chosen based on medical guidelines.
    """
    if sbp < 90 or dbp < 60:
        return "hypo"
    elif sbp > 130 or dbp > 80:
        return "hyper"
    else:
        return "normal"

def load_data(folder):
    """
    Loads data from JSON files, processes PPG to derive PR, extracts features,
    and assigns BP labels. Prioritizes 'PRAllData' from JSON if available and valid.
    Filters out invalid PR values.
    """
    X, Y_sbp, Y_dbp, labels, patient_ids = [], [], [], [], []
    files = glob(os.path.join(folder, "*.json"))
    
    initial_mem = get_current_memory_usage()
    print(f"Found {len(files)} JSON files in '{folder}'. Processing...")
    print(f"Initial Memory Usage: {initial_mem:.2f} MB")

    for path in files:
        with open(path) as f:
            try:
                d = json.load(f)
            except json.JSONDecodeError:
                print(f"Skipping malformed JSON file: {path}")
                continue

        # Get SBP/DBP labels from JSON (checking for common key names)
        sbp = d.get("SBP") or d.get("BPSystolic")
        dbp = d.get("DBP") or d.get("BPDiastolic")

        # Validate BP values
        if sbp is None or dbp is None:
            continue
        if not isinstance(sbp, (int, float)) or not isinstance(dbp, (int, float)):
            continue
        
        # Check for explicitly defined invalid BP values
        if sbp in INVALID_BP_VALUES or dbp in INVALID_BP_VALUES:
            # print(f"Skipping {path}: Contains explicitly invalid BP value (e.g., 0, -1, 999). SBP={sbp}, DBP={dbp}") # Uncomment for debugging
            continue

        # Original basic validity check for BP values
        if sbp <= 0 or dbp <= 0: 
            # print(f"Skipping {path}: SBP or DBP is non-positive after initial checks. SBP={sbp}, DBP={dbp}") # Uncomment for debugging
            continue

        label = get_bp_label(sbp, dbp)
        
        # Get PPG data
        ppg_raw = d.get("Pleth")
        
        # --- Determine PR source: PRAllData from JSON or calculated from Pleth ---
        pr_derived_array = None
        pr_all_data_from_json = d.get("PRAllData")

        if isinstance(pr_all_data_from_json, list):
            temp_pr_array = np.array(pr_all_data_from_json, dtype=float)
            # Replace invalid PR values (0, 127, 255) with NaN
            for val in INVALID_PR_VALUES:
                temp_pr_array[temp_pr_array == val] = np.nan
            
            if np.count_nonzero(~np.isnan(temp_pr_array)) >= MIN_PR_ALL_DATA_LEN: # Check for enough *valid* data
                pr_derived_array = temp_pr_array
        
        # Fallback to calculating from PPG if PRAllData is not valid or not present
        if pr_derived_array is None:
            if not ppg_raw or not isinstance(ppg_raw, list) or len(ppg_raw) < MIN_PPG_LEN:
                continue # Skip if no valid PRAllData AND no valid Pleth
            else:
                ppg = medfilt(np.array(ppg_raw), kernel_size=3)
                calculated_pr = calculate_pr_from_ppg(ppg)
                # Replace any 0.0s (from calculation failures) with NaN for consistency
                calculated_pr[calculated_pr == 0.0] = np.nan 
                if np.count_nonzero(~np.isnan(calculated_pr)) >= MIN_PR_ALL_DATA_LEN:
                    pr_derived_array = calculated_pr
                else:
                    continue # Skip if calculated PR also insufficient

        if pr_derived_array is None: 
            continue
            
        # Ensure PPG data is available for morphological feature extraction
        if not ppg_raw or not isinstance(ppg_raw, list) or len(ppg_raw) < MIN_PPG_LEN:
            print(f"Warning: Skipping {os.path.basename(path)}. PR data available but Pleth data is missing/insufficient for other feature extraction.")
            continue
        
        ppg = medfilt(np.array(ppg_raw), kernel_size=3)

        # Fix 2: ONE ROW PER SEGMENT — do NOT average across segments.
        # This gives 6× more training rows and matches inference behavior exactly.
        patient_id = d.get("PatientID") or d.get("Name") or os.path.basename(path)
        for i in range(SEGMENTS):
            seg_ppg = ppg[i * SEG_LEN : (i + 1) * SEG_LEN]
            pr_seg  = pr_derived_array[i * (SEG_LEN // FS) : (i + 1) * (SEG_LEN // FS)]
            feat = extract_features(seg_ppg, pr_seg)
            if feat:
                X.append(feat)
                Y_sbp.append(sbp)
                Y_dbp.append(dbp)
                labels.append(label)
                patient_ids.append(patient_id)

    print(f"Loaded {len(X)} segment rows from files (Fix 2: per-segment training).")
    print(f"Memory Usage After Data Loading: {get_current_memory_usage():.2f} MB")
    return np.array(X), np.array(Y_sbp), np.array(Y_dbp), np.array(labels), np.array(patient_ids)

def train_models(X, Y_sbp, Y_dbp, labels, patient_ids, output_dir="model"):
    """
    Trains classification and regression models and saves them.
    Fix 5: Uses XGBoost regressors with RandomizedSearchCV.
    Fix 6: GroupShuffleSplit by patient_id for honest subject-independent evaluation.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Memory Before Training: {get_current_memory_usage():.2f} MB")

    # ── Fix 6: Subject-independent split ─────────────────────────────────
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, labels, groups=patient_ids))
    X_train, X_test   = X[train_idx], X[test_idx]
    l_train, l_test   = labels[train_idx], labels[test_idx]
    print(f"Subject-independent split: {len(train_idx)} train rows, {len(test_idx)} test rows")
    print(f"Unique patients in test:  {len(set(patient_ids[test_idx]))}")

    # ── Global Scaler ─────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(output_dir, "global_feature_scaler.pkl"))
    print(f"Saved global_feature_scaler.pkl  ({X_train.shape[1]} features)")

    # ── 1. Classifier (Logistic Regression) ──────────────────────────────
    print("\n--- Training Classifier ---")
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
    clf.fit(X_train_sc, l_train)
    print(f"Classifier Accuracy (subject-independent test): "
          f"{accuracy_score(l_test, clf.predict(X_test_sc)):.4f}")
    clf_path = os.path.join(output_dir, "classifier.pkl")
    joblib.dump(clf, clf_path)
    print(f"Saved classifier.pkl")

    # ── XGBoost hyperparameter space (Fix 5) ─────────────────────────────
    if USE_XGB:
        base_model  = XGBRegressor(random_state=42, n_jobs=-1, verbosity=0)
        param_space = {
            'n_estimators':     [100, 200, 300],
            'max_depth':        [3, 4, 5, 6],
            'learning_rate':    [0.01, 0.05, 0.1],
            'subsample':        [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 1.0],
            'reg_alpha':        [0, 0.1, 0.5],
            'reg_lambda':       [1, 2, 5],
        }
        print("Using XGBoost regressors.")
    else:
        from sklearn.ensemble import RandomForestRegressor
        base_model  = RandomForestRegressor(random_state=42)
        param_space = {
            'n_estimators':     [50, 100, 200],
            'max_depth':        [None, 10, 20],
            'min_samples_split':[2, 5],
        }
        print("Using RandomForest regressors (xgboost not available).")

    # ── 2. Group Regression Models ────────────────────────────────────────
    print("\n--- Training Regression Models per Category ---")
    total_size = 0
    for group in ["hypo", "normal", "hyper"]:
        g_train = train_idx[l_train == group]
        g_test  = test_idx [l_test  == group]
        if len(g_train) < 10:
            print(f"  Skipping '{group}': only {len(g_train)} train rows")
            continue
        print(f"\n  {group.upper()}  — {len(g_train)} train / {len(g_test)} test rows")

        Xg_tr = scaler.transform(X[g_train])
        Xg_te = scaler.transform(X[g_test])
        Ysbp_tr, Ydbp_tr = Y_sbp[g_train], Y_dbp[g_train]
        Ysbp_te, Ydbp_te = Y_sbp[g_test],  Y_dbp[g_test]

        # SBP
        sbp_search = RandomizedSearchCV(
            base_model, param_space, n_iter=30, cv=3,
            scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42
        )
        sbp_search.fit(Xg_tr, Ysbp_tr)
        sbp_model = sbp_search.best_estimator_

        # DBP
        dbp_search = RandomizedSearchCV(
            base_model, param_space, n_iter=30, cv=3,
            scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42
        )
        dbp_search.fit(Xg_tr, Ydbp_tr)
        dbp_model = dbp_search.best_estimator_

        # Evaluate
        if len(Xg_te) > 0:
            mae_sbp = mean_absolute_error(Ysbp_te, sbp_model.predict(Xg_te))
            mae_dbp = mean_absolute_error(Ydbp_te, dbp_model.predict(Xg_te))
            r2_sbp  = r2_score(Ysbp_te, sbp_model.predict(Xg_te))
            r2_dbp  = r2_score(Ydbp_te, dbp_model.predict(Xg_te))
            print(f"  SBP MAE={mae_sbp:.2f} R2={r2_sbp:.2f} | DBP MAE={mae_dbp:.2f} R2={r2_dbp:.2f}")
        else:
            print(f"  No test rows for {group} — skipping evaluation")

        combined = {'sbp_model': sbp_model, 'dbp_model': dbp_model}
        out_path = os.path.join(output_dir, f"{group}_models.pkl")
        joblib.dump(combined, out_path)
        sz = os.path.getsize(out_path)
        total_size += sz
        print(f"  Saved {group}_models.pkl  ({sz/1024:.1f} KB)")

    print(f"\nDone. Models in: {os.path.abspath(output_dir)}")
    print(f"Total model size: {total_size/1024:.1f} KB")
    print(f"Memory After Training: {get_current_memory_usage():.2f} MB")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train BP prediction models from PPG data.")
    parser.add_argument("data_folder", help="Path to the folder containing JSON data files.")
    parser.add_argument("--output_dir", default="model",
                        help="Path to the directory to save trained models (default: 'model').")
    args = parser.parse_args()

    X, Y_sbp, Y_dbp, labels, patient_ids = load_data(args.data_folder)

    if len(X) == 0:
        print("No valid data loaded. Cannot train models.")
    else:
        train_models(X, Y_sbp, Y_dbp, labels, patient_ids, args.output_dir)