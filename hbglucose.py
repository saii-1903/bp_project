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
from scipy.stats import skew as _skew, kurtosis as _kurtosis
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
FS               = 200
SEGMENT_SECONDS  = 5
SEGMENT_SAMPLES  = FS * SEGMENT_SECONDS
SEGMENTS_PER_FILE = 6

# Physiological validity ranges (Fix 5)
HB_RANGE      = (5.0, 25.0)    # g/dL
GLUCOSE_RANGE = (30.0, 600.0)  # mg/dL

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
    auc = np.trapezoid(cycle, time)
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
        # indices 16-19: normalised signal stats (scale-invariant across sensors)
        np.mean(norm), np.std(norm), np.max(norm), np.min(norm)
    ]

    # Fix 7: SpO2-inspired perfusion ratio features (Beer-Lambert Law basis)
    # AC component = peak-to-peak amplitude of the PPG (pulsatile)
    # DC component = mean of the signal (non-pulsatile baseline)
    ac_component = np.max(signal) - np.min(signal)
    dc_component = np.mean(np.abs(signal)) + 1e-9
    ac_dc_ratio  = ac_component / dc_component          # correlates with SpO2/Hb

    # Signal entropy (irregular Hb distribution → noisier signal)
    signal_norm = signal - np.min(signal)
    signal_norm = signal_norm / (np.sum(signal_norm) + 1e-9)
    entropy = float(-np.sum(signal_norm * np.log(signal_norm + 1e-9)))

    # Skewness (inverted/asymmetric waveform correlates with vascular changes)
    from scipy.stats import skew as _skew
    sig_skew = float(_skew(signal))

    features += [ac_dc_ratio, entropy, sig_skew]   # 20-22

    # Additional optical / shape features (same as glucosehb.py)
    perfusion_index = (np.max(signal) - np.min(signal)) / (np.mean(signal) + 1e-9)  # 23
    sqi             = float(np.max(cycle) / (np.std(signal) + 1e-6))                # 24
    cycle_skew      = float(_skew(cycle))                                            # 25
    cycle_kurt      = float(_kurtosis(cycle))                                        # 26
    features       += [perfusion_index, sqi, cycle_skew, cycle_kurt]

    # Log-transforms: amplitude/area + AC/DC + perfusion index
    for idx in [0, 8, 16, 17, 18, 19, 20, 23]:
        features[idx] = np.log1p(np.abs(features[idx]))

    return features

def load_data_from_json_files(input_dir):
    """
    Loads JSON files, validates labels, extracts features, and builds X/y arrays
    for Hb and Glucose regressors.
    Fix 2: Invalid age → skip.  Fix 3: Missing gender → warn.
    Fix 4: Richer demographics.  Fix 5: Physiological range validation.
    """
    X_hb, y_hb         = [], []
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
        if not pleth.any():
            logging.warning(f"Missing Pleth in {file_path}. Skipping.")
            continue

        hb      = data.get("Hb", None)
        glucose = data.get("Glucose", data.get("Blood Glucose", None))

        # Fix 5: physiological range validation
        if hb is not None:
            try:
                hb = float(hb)
                if not (HB_RANGE[0] <= hb <= HB_RANGE[1]):
                    logging.warning(f"Suspicious Hb={hb} in {file_path}, skipping Hb label.")
                    hb = None
            except (TypeError, ValueError):
                hb = None

        if glucose is not None:
            try:
                glucose = float(glucose)
                if not (GLUCOSE_RANGE[0] <= glucose <= GLUCOSE_RANGE[1]):
                    logging.warning(f"Suspicious glucose={glucose} in {file_path}, skipping glucose label.")
                    glucose = None
            except (TypeError, ValueError):
                glucose = None

        # Fix 2: age validation
        age_raw = data.get("Age", None)
        try:
            age = float(age_raw) if age_raw is not None else None
        except (TypeError, ValueError):
            age = None
        if age is None or not (0 < age < 120):
            logging.warning(f"Missing/invalid age ({age_raw}) in {file_path} — skipping.")
            continue

        # Fix 3: gender warning
        raw_gender = data.get("Gender", None)
        if raw_gender is None:
            logging.warning(f"Missing gender in {file_path} — defaulting to 0.")
            gender = 0
        else:
            gender = 1 if raw_gender.lower() == "male" else 0

        # Fix 4: richer demographic features
        age_sq    = age ** 2
        is_senior = 1 if age > 60 else 0
        bmi_raw   = data.get("BMI", None)
        try:
            bmi = float(bmi_raw) if bmi_raw is not None and 10 < float(bmi_raw) < 80 else -1.0
        except (TypeError, ValueError):
            bmi = -1.0
        interaction = age * gender
        demo_feats  = [age, age_sq, is_senior, gender, interaction, bmi]

        segment_features = []
        for i in range(SEGMENTS_PER_FILE):
            seg = pleth[i * SEGMENT_SAMPLES: (i + 1) * SEGMENT_SAMPLES]
            if len(seg) < SEGMENT_SAMPLES:
                continue
            feats = extract_features(seg)
            if feats is None:
                continue
            segment_features.append(feats + demo_feats)

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

    logging.info(f"Hb samples: {len(X_hb)}  |  Glucose samples: {len(X_glucose)}")
    return np.array(X_hb), np.array(y_hb), np.array(X_glucose), np.array(y_glucose)



# ------------ Training ------------
def _train_ransac_pipeline(X_train, y_train, param_grid, label="model"):
    """
    Fix 8: GridSearchCV is run INSIDE the RANSAC pipeline so hyperparameters
    are optimised taking RANSAC's inlier sub-sampling into account.
    """
    # Pipeline: RANSAC wraps RF — we tune RF params via the pipeline's param names
    ransac_pipe = Pipeline([
        ('ransac', RANSACRegressor(
            estimator=RandomForestRegressor(random_state=42),
            min_samples=0.8,
            random_state=42
        ))
    ])
    # GridSearch param names: ransac__estimator__<param>
    pipe_grid = {
        f"ransac__estimator__{k}": v for k, v in param_grid.items()
    }
    search = GridSearchCV(ransac_pipe, pipe_grid, cv=5,
                          scoring='r2', n_jobs=-1)
    search.fit(X_train, y_train)
    print(f"  Best params for {label}: "
          f"{ {k.replace('ransac__estimator__', ''): v for k, v in search.best_params_.items()} }")
    return search.best_estimator_


# ------------ Main Function ------------
def main():
    parser = argparse.ArgumentParser(
        description="Train Hb and Glucose regressors from PPG JSON data")
    parser.add_argument("input_dir", help="Folder with training JSON files")
    parser.add_argument("--model-dir", default="water/models",
                        help="Folder to save model files")
    parser.add_argument("--min-samples", type=int, default=10)
    args = parser.parse_args()

    print("--- Data Loading and Feature Extraction ---")
    X_hb, y_hb, X_glucose, y_glucose = load_data_from_json_files(args.input_dir)

    param_grid = {
        'n_estimators':     [50, 100, 200],
        'max_depth':        [4, 6, 8, None],
        'min_samples_split':[2, 5, 10]
    }

    os.makedirs(args.model_dir, exist_ok=True)

    # ── Hb model ─────────────────────────────────────────────────────────
    if len(X_hb) < args.min_samples:
        print(f"⚠️  Hb: insufficient samples ({len(X_hb)} < {args.min_samples}). Skipping.")
    else:
        print(f"\n✅ Hb training: {len(X_hb)} samples")
        scaler_hb   = StandardScaler()
        X_sc_hb     = scaler_hb.fit_transform(X_hb)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_sc_hb, y_hb, test_size=0.2, random_state=42)

        hb_model = _train_ransac_pipeline(X_tr, y_tr, param_grid, label="Hb")

        y_pred = hb_model.predict(X_val)
        print(f"  Hb MAE:  {mean_absolute_error(y_val, y_pred):.2f}")
        print(f"  Hb RMSE: {np.sqrt(mean_squared_error(y_val, y_pred)):.2f}")
        print(f"  Hb R²:   {r2_score(y_val, y_pred):.2f}")

        joblib.dump(scaler_hb, os.path.join(args.model_dir, "scaler_hb.pkl"))
        joblib.dump(hb_model,  os.path.join(args.model_dir, "hb_regressor.pkl"))
        print(f"  Saved scaler_hb.pkl + hb_regressor.pkl")

    # ── Glucose model ─────────────────────────────────────────────────────
    if len(X_glucose) < args.min_samples:
        print(f"⚠️  Glucose: insufficient samples ({len(X_glucose)} < {args.min_samples}). Skipping.")
    else:
        print(f"\n✅ Glucose training: {len(X_glucose)} samples")
        scaler_glu = StandardScaler()
        X_sc_glu   = scaler_glu.fit_transform(X_glucose)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_sc_glu, y_glucose, test_size=0.2, random_state=42)

        glu_model = _train_ransac_pipeline(X_tr, y_tr, param_grid, label="Glucose")

        y_pred = glu_model.predict(X_val)
        print(f"  Glucose MAE:  {mean_absolute_error(y_val, y_pred):.2f}")
        print(f"  Glucose RMSE: {np.sqrt(mean_squared_error(y_val, y_pred)):.2f}")
        print(f"  Glucose R²:   {r2_score(y_val, y_pred):.2f}")

        joblib.dump(scaler_glu, os.path.join(args.model_dir, "scaler_glucose.pkl"))
        joblib.dump(glu_model,  os.path.join(args.model_dir, "glucose_regressor.pkl"))
        print(f"  Saved scaler_glucose.pkl + glucose_regressor.pkl")


if __name__ == "__main__":
    main()
