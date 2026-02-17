# train_glucose_hb.py

import os
import argparse
import glob
import json
import logging
import numpy as np
import joblib

from scipy.signal import butter, filtfilt, find_peaks, medfilt
from scipy.fft import fft
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import RANSACRegressor # Added RANSAC for robust regression

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
FS = 100
SEGMENT_SECONDS = 5
SEGMENT_SAMPLES = FS * SEGMENT_SECONDS
SEGMENTS_PER_FILE = 6

# ------------ Feature Extraction ------------
def bandpass_filter(signal, fs=FS):
    nyq = 0.5 * fs
    b, a = butter(3, [0.4 / nyq, 11 / nyq], btype='band')
    return filtfilt(b, a, signal)

def extract_features(signal, fs=FS):
    """
    Extracts features from a single PPG signal segment.
    """
    if len(signal) < fs:
        return None

    signal = medfilt(signal, kernel_size=3)
    signal = bandpass_filter(signal, fs=fs)

    norm = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-6)

    peaks, _ = find_peaks(norm, distance=int(fs * 0.4))
    mins, _ = find_peaks(-norm, distance=int(fs * 0.3))

    if len(peaks) < 2 or len(mins) < 2:
        return None

    cycles = []
    for p in peaks:
        left = mins[mins < p]
        right = mins[mins > p]
        if len(left) > 0 and len(right) > 0:
            cycles.append(norm[left[-1]:right[0]])

    if not cycles:
        return None

    cycle = max(cycles, key=lambda x: np.max(x))
    time = np.linspace(0, len(cycle) / fs, len(cycle))

    d1 = np.gradient(cycle)
    d2 = np.gradient(d1)
    auc = np.trapz(cycle, time)
    pw = time[-1]
    ttp_idx = np.argmax(cycle)
    ttp = time[ttp_idx]
    tdp = pw - ttp
    ratio = ttp / tdp if tdp else 0

    freqs = np.fft.fftfreq(len(cycle), d=1 / fs)
    fft_vals = np.abs(fft(cycle)[:len(cycle)//2])
    freqs = freqs[:len(cycle)//2]
    
    peak_idxs, _ = find_peaks(fft_vals, distance=5)
    if len(peak_idxs) < 3:
        top_freqs = [0, 0, 0]
        top_mags = [0, 0, 0]
    else:
        sorted_idxs = np.argsort(fft_vals[peak_idxs])[-3:]
        top_freqs = freqs[peak_idxs][sorted_idxs]
        top_mags = fft_vals[peak_idxs][sorted_idxs]

    ibi_list = np.diff(peaks) / fs if len(peaks) > 1 else [0]
    hrv_std = np.std(ibi_list) if len(ibi_list) > 1 else 0

    features = [
        np.max(cycle), pw, ttp, ratio,
        np.max(d1), np.min(d1), np.max(d2), np.min(d2),
        auc,
        top_freqs[0], top_mags[0],
        top_freqs[1], top_mags[1],
        top_freqs[2], top_mags[2],
        hrv_std,
        np.mean(signal), np.std(signal), np.max(signal), np.min(signal)
    ]

    # Feature Engineering updates
    # 1. Log-transform signal related features (Beer-Lambert Law mimic)
    # Features 0, 8, 16, 17, 18, 19 are amplitude/area based
    for idx in [0, 8, 16, 17, 18, 19]:
        features[idx] = np.log1p(np.abs(features[idx]))

    return features

# ------------ Data Loading ------------
def load_data_from_json_files(input_dir):
    """
    Loads data from JSON files, extracts features, and returns separate lists for Hb and Glucose.
    """
    X_hb, y_hb = [], []
    X_glucose, y_glucose = [], []
    files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    logging.info(f"Found {len(files)} JSON files.")

    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [l for l in f if not l.strip().startswith("//")]
                data = json.loads("\n".join(lines))
        except Exception as e:
            logging.warning(f"Failed to read {file_path}: {e}")
            continue

        pleth = np.array(data.get("Pleth", []))
        hb = data.get("Hb", None)
        glucose = data.get("Glucose", data.get("Blood Glucose", None))
        age = data.get("Age", -1)
        gender = 1 if data.get("Gender", "").lower() == "male" else 0

        if not pleth.any():
            logging.warning(f"Missing Pleth data in {file_path}. Skipping.")
            continue

        segment_features = []
        for i in range(SEGMENTS_PER_FILE):
            seg = pleth[i * SEGMENT_SAMPLES: (i + 1) * SEGMENT_SAMPLES]
            if len(seg) < SEGMENT_SAMPLES:
                continue

            feats = extract_features(seg)
            if feats is None:
                continue
            
            # Interaction Features: Age * Gender
            interaction = age * gender
            full_feat = feats + [age, gender, interaction]
            segment_features.append(full_feat)

        if len(segment_features) < 3:
            logging.warning(f"Not enough valid segments in {file_path}. Skipping.")
            continue

        avg_feat = np.mean(segment_features, axis=0)

        if hb is not None:
            X_hb.append(avg_feat)
            y_hb.append(hb)
        
        if glucose is not None:
            X_glucose.append(avg_feat)
            y_glucose.append(glucose)

    return np.array(X_hb), np.array(y_hb), np.array(X_glucose), np.array(y_glucose)


# ------------ Main Function ------------
def main():
    parser = argparse.ArgumentParser(description="Train Hb and Glucose regressors from PPG JSON data")
    parser.add_argument("input_dir", help="Folder with training JSON files")
    parser.add_argument("--model-dir", default="./model_output", help="Folder to save model files")
    parser.add_argument("--min-samples", type=int, default=10)
    args = parser.parse_args()

    print("--- Data Loading and Feature Extraction ---")
    X_hb, y_hb, X_glucose, y_glucose = load_data_from_json_files(args.input_dir)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [4, 6, 8, None],
        'min_samples_split': [2, 5, 10]
    }

    # --- Train and evaluate Hb model ---
    if len(X_hb) < args.min_samples:
        print(f"⚠️ Skipping Hb model training: Not enough samples. Found {len(X_hb)}, required {args.min_samples}.")
    else:
        print(f"✅ Loaded {len(X_hb)} valid samples for Hb model.")
        print("--- Preprocessing for Hb Model ---")
        scaler_hb = StandardScaler()
        X_scaled_hb = scaler_hb.fit_transform(X_hb)
        X_train_hb, X_val_hb, y_train_hb, y_val_hb = train_test_split(X_scaled_hb, y_hb, test_size=0.2, random_state=42)

        print("--- Hyperparameter Tuning and Training Hb Model (Robust) ---")
        # Base estimator for RANSAC
        base_rf = RandomForestRegressor(random_state=42)
        
        # Grid Search on the underlying regressor is tricky with RANSAC in a simple pipeline
        # For simplicity in this upgrade, we GridSearch the RF first, then wrap best RF in RANSAC
        # or just use RANSAC with a strong default RF.
        # Let's simple-grid-search the RF first as before.
        hb_grid_search = GridSearchCV(base_rf, param_grid, cv=5, scoring='r2', n_jobs=-1)
        hb_grid_search.fit(X_train_hb, y_train_hb)
        best_rf_hb = hb_grid_search.best_estimator_
        
        # Wrap in RANSAC
        print("    Wrapping best Hb estimator in RANSAC...")
        hb_model = RANSACRegressor(estimator=best_rf_hb, min_samples=0.8, random_state=42)
        hb_model.fit(X_train_hb, y_train_hb)
        
        print(f"Optimal RF Parameters (inside RANSAC): {hb_grid_search.best_params_}")

        print("--- Evaluating Hb Model ---")
        y_pred_hb = hb_model.predict(X_val_hb)
        print(f"Hb MAE: {mean_absolute_error(y_val_hb, y_pred_hb):.2f}")
        # CORRECTED: Calculate RMSE by taking the square root of MSE
        rmse_hb = np.sqrt(mean_squared_error(y_val_hb, y_pred_hb))
        print(f"Hb RMSE: {rmse_hb:.2f}")
        print(f"Hb R2 Score: {r2_score(y_val_hb, y_pred_hb):.2f}")

        print("--- Saving Hb Model and Scaler ---")
        os.makedirs(args.model_dir, exist_ok=True)
        joblib.dump(scaler_hb, os.path.join(args.model_dir, "scaler_hb.pkl"))
        joblib.dump(hb_model, os.path.join(args.model_dir, "hb_regressor.pkl"))
        print(f"✅ Hb model and scaler saved to '{args.model_dir}'")

    # --- Train and evaluate Glucose model ---
    if len(X_glucose) < args.min_samples:
        print(f"⚠️ Skipping Glucose model training: Not enough samples. Found {len(X_glucose)}, required {args.min_samples}.")
    else:
        print(f"✅ Loaded {len(X_glucose)} valid samples for Glucose model.")
        print("--- Preprocessing for Glucose Model ---")
        scaler_glucose = StandardScaler()
        X_scaled_glucose = scaler_glucose.fit_transform(X_glucose)
        X_train_glucose, X_val_glucose, y_train_glucose, y_val_glucose = train_test_split(X_scaled_glucose, y_glucose, test_size=0.2, random_state=42)

        print("--- Hyperparameter Tuning and Training Glucose Model (Robust) ---")
        base_rf_glu = RandomForestRegressor(random_state=42)
        
        glucose_grid_search = GridSearchCV(base_rf_glu, param_grid, cv=5, scoring='r2', n_jobs=-1)
        glucose_grid_search.fit(X_train_glucose, y_train_glucose)
        best_rf_glu = glucose_grid_search.best_estimator_
        
        print("    Wrapping best Glucose estimator in RANSAC...")
        glucose_model = RANSACRegressor(estimator=best_rf_glu, min_samples=0.8, random_state=42)
        glucose_model.fit(X_train_glucose, y_train_glucose)
        
        print(f"Optimal Glucose Model Parameters: {glucose_grid_search.best_params_}")
        
        print("--- Evaluating Glucose Model ---")
        y_pred_glucose = glucose_model.predict(X_val_glucose)
        print(f"Glucose MAE: {mean_absolute_error(y_val_glucose, y_pred_glucose):.2f}")
        # CORRECTED: Calculate RMSE by taking the square root of MSE
        rmse_glucose = np.sqrt(mean_squared_error(y_val_glucose, y_pred_glucose))
        print(f"Glucose RMSE: {rmse_glucose:.2f}")
        print(f"Glucose R2 Score: {r2_score(y_val_glucose, y_pred_glucose):.2f}")

        print("--- Saving Glucose Model and Scaler ---")
        os.makedirs(args.model_dir, exist_ok=True)
        joblib.dump(scaler_glucose, os.path.join(args.model_dir, "scaler_glucose.pkl"))
        joblib.dump(glucose_model, os.path.join(args.model_dir, "glucose_regressor.pkl"))
        print(f"✅ Glucose model and scaler saved to '{args.model_dir}'")


if __name__ == "__main__":
    main()