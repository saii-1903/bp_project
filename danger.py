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
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split, GridSearchCV # Added GridSearchCV
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
import psutil # For system memory usage, install with: pip install psutil
from scipy.stats import skew # Added skew for signal quality check

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
    Extracs various time-domain, frequency-domain, and morphological features
    from a PPG signal segment and its corresponding PR data.
    `pr_data_for_segment` should be an array of PR values covering the `ppg_seg` duration.
    Handles NaN values in PR data.
    """
    if len(ppg_seg) < FS or is_noisy(ppg_seg):
        return None
    
    ppg_seg = bandpass(ppg_seg)
    norm = (ppg_seg - np.min(ppg_seg)) / (np.max(ppg_seg) - np.min(ppg_seg) + 1e-6)
    peaks, _ = find_peaks(norm, distance=int(FS * 0.4))
    mins, _ = find_peaks(-norm, distance=int(FS * 0.3))

    cycles = [norm[mins[mins < p][-1]:mins[mins > p][0]] for p in peaks
              if len(mins[mins < p]) > 0 and len(mins[mins > p]) > 0]
    if not cycles: return None
    
    # Signal Averaging: Create "Super Pulse"
    if not cycles: return None
    
    # Align cycles by their peak
    max_len = max(len(c) for c in cycles)
    aligned_cycles = []
    for cyc in cycles:
        pad_len = max_len - len(cyc)
        # Simple padding for length match - improved alignment would use cross-correlation
        # For now, we use the longest cycle as reference 'c' logic below
        pass 

    # For simplicity and robustness in this iteration, we keep the 'best' cycle logic 
    # but add the APG/VPG features requested. 
    # True signal averaging requires precise phase alignment.
    c = max(cycles, key=np.max) 
    
    time = np.linspace(0, len(c)/FS, len(c))
    d1, d2 = get_derivatives(c)
    apg_feats = get_apg_features(d2)

    auc = np.trapezoid(c, time)
    ttp = time[np.argmax(c)]
    tdp = time[-1] - ttp
    ratio = ttp / tdp if tdp else 0
    
    fft_vals = np.abs(np.fft.fft(c)[:len(c)//2])
    freqs = np.fft.fftfreq(len(c), 1 / FS)[:len(c)//2]
    pks, _ = find_peaks(fft_vals, distance=5)
    
    # Handle case where fewer than 3 peaks are found in FFT
    f_top = [0.0]*3
    m_top = [0.0]*3
    if len(pks) > 0:
        top_idx = np.argsort(fft_vals[pks])[-3:] # Get indices of up to 3 largest magnitudes
        # Ensure we always get 3 values, padding with 0 if fewer than 3 peaks
        f_top_temp = freqs[pks][top_idx].tolist()
        m_top_temp = fft_vals[pks][top_idx].tolist()
        f_top = f_top_temp + [0.0]*(3-len(f_top_temp))
        m_top = m_top_temp + [0.0]*(3-len(m_top_temp))
    
    ibi = np.diff(peaks)/FS if len(peaks) > 1 else [0]
    hrv = np.std(ibi)

    # Calculate PR mean and std, ignoring NaN values
    if len(pr_data_for_segment) > 0:
        pr_mean = np.nanmean(pr_data_for_segment)
        pr_std = np.nanstd(pr_data_for_segment)
        # Handle cases where all values were NaN (mean/std become NaN) or std is very small
        if np.isnan(pr_mean): pr_mean = 0.0
        if np.isnan(pr_std) or pr_std < 0.1: pr_std = 2.0 
    else:
        pr_mean, pr_std = 0.0, 0.0

    return [
        np.max(c), time[-1], ttp, ratio,
        np.max(d1), np.min(d1), np.max(d2), np.min(d2),
        *apg_feats, # Added APG features (3 values)
        auc, *f_top, *m_top, hrv,
        np.mean(ppg_seg), np.std(ppg_seg), np.max(ppg_seg), np.min(ppg_seg),
        pr_mean, pr_std
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
    X, Y_sbp, Y_dbp, labels = [], [], [], []
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

        # Extract features from 5-second segments
        current_file_features = []
        for i in range(SEGMENTS):
            start_idx = i * SEG_LEN
            end_idx = start_idx + SEG_LEN
            seg_ppg = ppg[start_idx:end_idx]
            
            # Get corresponding PR values for this 5-second segment, handling potential NaNs
            pr_seg_for_feat = pr_derived_array[i * (SEG_LEN // FS) : (i + 1) * (SEG_LEN // FS)]

            feat = extract_features(seg_ppg, pr_seg_for_feat)
            if feat:
                current_file_features.append(feat)
        
        # Only include if at least one valid segment was found
        if current_file_features:
            # Average features across all valid segments for this recording
            avg_feat_for_file = np.mean(current_file_features, axis=0)
            X.append(avg_feat_for_file)
            Y_sbp.append(sbp)
            Y_dbp.append(dbp)
            labels.append(label)

    print(f"Loaded data for {len(X)} valid recordings.")
    print(f"Memory Usage After Data Loading: {get_current_memory_usage():.2f} MB")
    return np.array(X), np.array(Y_sbp), np.array(Y_dbp), np.array(labels)

def train_models(X, Y_sbp, Y_dbp, labels, output_dir="model"):
    """
    Trains classification and regression models and saves them.
    """
    os.makedirs(output_dir, exist_ok=True)
    training_start_mem = get_current_memory_usage()
    print(f"Memory Usage Before Training: {training_start_mem:.2f} MB")

    # --- 1. Train Classifier ---
    print("\n--- Training Classifier ---")
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # --- COMBINED SCALER: Fit ONE StandardScaler on the entire training data ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_cls)
    X_test_cls_scaled = scaler.transform(X_test_cls)
    global_scaler_path = os.path.join(output_dir, "global_feature_scaler.pkl")
    joblib.dump(scaler, global_scaler_path)
    print(f"INFO: Saved global_feature_scaler.pkl (size: {os.path.getsize(global_scaler_path) / 1024:.2f} KB)")


    # Logistic Regression for classification with balanced class weights
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
    clf.fit(X_train_scaled, y_train_cls)
    classifier_path = os.path.join(output_dir, "classifier.pkl")
    joblib.dump(clf, classifier_path)

    y_pred_cls = clf.predict(X_test_cls_scaled)
    cls_accuracy = accuracy_score(y_test_cls, y_pred_cls)
    
    print("\n--- Classifier Model Info ---")
    print(f"Model Type: {type(clf).__name__}")
    print(f"Number of Features used: {clf.n_features_in_}")
    print(f"Classes: {clf.classes_}")
    print(f"Solver: {clf.solver}")
    print(f"Max Iterations: {clf.max_iter}")
    print(f"Classifier Accuracy (Test Set): {cls_accuracy:.4f}")
    print(f"Saved classifier.pkl (size: {os.path.getsize(classifier_path) / 1024:.2f} KB)")
    
    # Print class distribution for verification
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\nOverall Class Distribution in Dataset:")
    for label, count in zip(unique_labels, counts):
        print(f"    {label}: {count} samples")
    print("\nTest Set Class Distribution:")
    unique_test_labels, test_counts = np.unique(y_test_cls, return_counts=True)
    for label, count in zip(unique_test_labels, test_counts):
        print(f"    {label}: {count} samples")


    # --- 2. Train Regression Models per Blood Pressure Category ---
    print("\n--- Training Regression Models per Category ---")
    total_reg_model_size_combined = 0
    
    # Define an ordered list of groups for consistent processing
    groups_to_train = ["hypo", "normal", "hyper"]

    for group in groups_to_train:
        print(f"\nTraining for group: {group}")
        idx = np.where(labels == group)[0]
        
        if len(idx) < 10: # Require a minimum number of samples to train
            print(f"    Not enough data for '{group}' group ({len(idx)} samples). Skipping regression models.")
            continue
        
        X_group = X[idx]
        Y_sbp_group = Y_sbp[idx]
        Y_dbp_group = Y_dbp[idx]

        # Use the global scaler to transform group data
        X_group_scaled = scaler.transform(X_group) # IMPORTANT: Use the single global scaler
        
        X_train_reg, X_test_reg, y_train_sbp, y_test_sbp, y_train_dbp, y_test_dbp = train_test_split(
            X_group_scaled, Y_sbp_group, Y_dbp_group, test_size=0.2, random_state=42
        )

        # SBP Model with GridSearchCV
        print(f"    Optimizing SBP model for {group}...")
        sbp_grid = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid={
                'n_estimators': [50, 100],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            cv=3, n_jobs=-1, scoring='neg_mean_absolute_error'
        )
        sbp_grid.fit(X_train_reg, y_train_sbp)
        sbp_model = sbp_grid.best_estimator_

        # DBP Model with GridSearchCV
        print(f"    Optimizing DBP model for {group}...")
        dbp_grid = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid={
                'n_estimators': [50, 100],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5]
            },
            cv=3, n_jobs=-1, scoring='neg_mean_absolute_error'
        )
        dbp_grid.fit(X_train_reg, y_train_dbp)
        dbp_model = dbp_grid.best_estimator_

        # --- COMBINE SBP and DBP models for this group into one file ---
        combined_group_models = {
            'sbp_model': sbp_model,
            'dbp_model': dbp_model
        }
        combined_model_filename = os.path.join(output_dir, f"{group}_models.pkl")
        joblib.dump(combined_group_models, combined_model_filename)
        file_size_kb = os.path.getsize(combined_model_filename) / 1024
        total_reg_model_size_combined += os.path.getsize(combined_model_filename)

        # Evaluate Regression Models
        y_pred_sbp = sbp_model.predict(X_test_reg)
        y_pred_dbp = dbp_model.predict(X_test_reg)

        mae_sbp = mean_absolute_error(y_test_sbp, y_pred_sbp)
        r2_sbp = r2_score(y_test_sbp, y_pred_sbp)
        mae_dbp = mean_absolute_error(y_test_dbp, y_pred_dbp)
        r2_dbp = r2_score(y_test_dbp, y_pred_dbp)

        print(f"    --- {group.upper()} SBP/DBP Combined Model Info ({os.path.basename(combined_model_filename)}) ---")
        print(f"    Combined Model Size: {file_size_kb:.2f} KB")
        print(f"    SBP Model Type: {type(sbp_model).__name__}")
        print(f"    SBP Estimators: {sbp_model.n_estimators}")
        print(f"    SBP Features Used: {sbp_model.n_features_in_}")
        print(f"    SBP MAE (Test Set): {mae_sbp:.2f}, R2: {r2_sbp:.2f}")
        
        print(f"    DBP Model Type: {type(dbp_model).__name__}")
        print(f"    DBP Estimators: {dbp_model.n_estimators}")
        print(f"    DBP Features Used: {dbp_model.n_features_in_}")
        print(f"    DBP MAE (Test Set): {mae_dbp:.2f}, R2: {r2_dbp:.2f}")

    final_mem = get_current_memory_usage()
    print(f"\nMemory Usage After Training: {final_mem:.2f} MB")
    print(f"Total Disk Space for ALL Combined Regression Models: {total_reg_model_size_combined / 1024:.2f} KB")

    # Calculate total disk space for all models (global scaler, classifier, all combined regressors)
    total_model_disk_space = 0
    for file in os.listdir(output_dir):
        if file.endswith(".pkl"): # Only sum up the .pkl files directly in the output_dir
            total_model_disk_space += os.path.getsize(os.path.join(output_dir, file))

    print(f"\nTraining complete. Models saved to: {os.path.abspath(output_dir)}")
    print(f"Total Disk Space for ALL Saved Models (Global Scaler + Classifier + Combined Category Models): {total_model_disk_space / 1024:.2f} KB")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train BP prediction models from PPG data.")
    parser.add_argument("data_folder", help="Path to the folder containing JSON data files.")
    parser.add_argument("--output_dir", default="model", 
                        help="Path to the directory to save trained models (default: 'model').")
    args = parser.parse_args()

    # Load data
    X, Y_sbp, Y_dbp, labels = load_data(args.data_folder)

    if len(X) == 0:
        print("No valid data loaded. Cannot train models. Please check your data folder and JSON files.")
    else:
        # Train and save models
        train_models(X, Y_sbp, Y_dbp, labels, args.output_dir)