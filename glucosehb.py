import os
import argparse
import glob
import json
import logging
import numpy as np
import joblib
import csv

from scipy.signal import butter, filtfilt, find_peaks, medfilt
from scipy.fft import fft

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
        np.mean(signal), np.std(signal), np.max(signal), np.min(signal)
    ]

    # Feature Engineering updates (Must match training!)
    # 1. Log-transform signal related features (Beer-Lambert Law mimic)
    # Features 0, 8, 16, 17, 18, 19 are amplitude/area based
    for idx in [0, 8, 16, 17, 18, 19]:
        features[idx] = np.log1p(np.abs(features[idx]))

    return features

# ------------ Prediction Function ------------
def predict_from_file(json_file_path, model_dir):
    """Predicts Hb and Glucose from a single JSON file and returns the results."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            lines = [l for l in f if not l.strip().startswith("//")]
            data = json.loads("\n".join(lines))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error processing {json_file_path}: {e}")
        return None

    pleth_data_raw = data.get("Pleth") or data.get("IRGainAllData")
    if not pleth_data_raw: 
        logging.error(f"Missing 'Pleth' or 'IRGainAllData' in {json_file_path}. Skipping.")
        return None

    pleth_flattened = []
    if isinstance(pleth_data_raw, list):
        for item in pleth_data_raw:
            if isinstance(item, (int, float)):
                pleth_flattened.append(item)
            elif isinstance(item, list):
                pleth_flattened.extend(item)
    else:
        logging.error(f"Skipping {json_file_path}: Signal data is not a list.")
        return None

    pleth = np.array(pleth_flattened)
    if not pleth.any() or len(pleth) < SEGMENT_SAMPLES:
        logging.error(f"Invalid signal data in {json_file_path}. Skipping.")
        return None

    age = data.get("Age", -1) 
    gender = 1 if data.get("Gender", "Unspecified").lower() == "male" else 0

    segment_features = []
    for i in range(SEGMENTS_PER_FILE):
        seg = pleth[i * SEGMENT_SAMPLES: (i + 1) * SEGMENT_SAMPLES]
        if len(seg) < SEGMENT_SAMPLES: continue
        feats = extract_features(seg)
        if feats is None: continue
        
        # Interaction Features: Age * Gender (Must match training!)
        interaction = age * gender
        segment_features.append(feats + [age, gender, interaction])
    
    if not segment_features: 
        logging.error(f"No valid segments found for {json_file_path}. Skipping.")
        return None
    
    X_pred = np.mean(segment_features, axis=0).reshape(1, -1)

    predicted_hb = "N/A"
    try:
        scaler_hb = joblib.load(os.path.join(model_dir, "scaler_hb.pkl"))
        hb_model = joblib.load(os.path.join(model_dir, "hb_regressor.pkl"))
        X_pred_scaled_hb = scaler_hb.transform(X_pred)
        predicted_hb = f"{hb_model.predict(X_pred_scaled_hb)[0]:.2f}"
    except (FileNotFoundError, Exception) as e:
        logging.warning(f"Could not predict Hb for {os.path.basename(json_file_path)}: {e}")

    predicted_glucose = "N/A"
    try:
        scaler_glucose = joblib.load(os.path.join(model_dir, "scaler_glucose.pkl"))
        glucose_model = joblib.load(os.path.join(model_dir, "glucose_regressor.pkl"))
        X_pred_scaled_glucose = scaler_glucose.transform(X_pred)
        predicted_glucose = f"{glucose_model.predict(X_pred_scaled_glucose)[0]:.2f}"
    except (FileNotFoundError, Exception) as e:
        logging.warning(f"Could not predict Glucose for {os.path.basename(json_file_path)}: {e}")

    return {
        'file_name': os.path.basename(json_file_path),
        'predicted_hb': predicted_hb,
        'predicted_hb': predicted_hb,
        'predicted_glucose': predicted_glucose
    }

# Global dictionary to store previous values for smoothing
# Key: file_name (or patient ID if available), Value: {'glucose': val, 'hb': val}
# For this script which processes files sequentially, we'll simulate it or use a simple list if we assume time-series order.
# However, without explicit time-stamps or patient IDs in the CLI usage, true smoothing is limited.
# We will implement the Logic as requested for a single file context or sequential "stream" simulation.

LAST_GLUCOSE = None

def smooth_glucose(new_val):
    global LAST_GLUCOSE
    if new_val is None: return None
    
    # Range Clamping [40, 400]
    if new_val < 40: new_val = 40.0
    if new_val > 400: new_val = 400.0
    
    if LAST_GLUCOSE is None:
        LAST_GLUCOSE = new_val
        return new_val
    else:
        # Weighted Moving Average: 0.7 * New + 0.3 * Last
        smoothed = 0.7 * new_val + 0.3 * LAST_GLUCOSE
        LAST_GLUCOSE = smoothed
        return smoothed

# ------------ Main function for command-line use ------------
def main():
    parser = argparse.ArgumentParser(description="Predict Hb and Glucose from PPG JSON data")
    parser.add_argument("input_path", help="Path to a single JSON file or a folder of JSON files")
    parser.add_argument("--model-dir", default="./model_output", help="Folder where model files are saved")
    parser.add_argument("--output-csv", help="Optional path to save the output as a CSV file")
    args = parser.parse_args()
    
    results = []
    
    if os.path.isfile(args.input_path):
        result = predict_from_file(args.input_path, args.model_dir)
        if result: results.append(result)
    elif os.path.isdir(args.input_path):
        json_files = glob.glob(os.path.join(args.input_path, "*.json"))
        if not json_files:
            logging.warning(f"No JSON files found in directory: {args.input_path}")
        for file_path in json_files:
            result = predict_from_file(file_path, args.model_dir)
            if result: 
                # Apply Smoothing to Glucose Step-by-Step in the list
                try:
                    raw_glu = float(result['predicted_glucose'])
                    smoothed_glu = smooth_glucose(raw_glu)
                    result['predicted_glucose'] = f"{smoothed_glu:.2f}"
                except ValueError:
                    pass # Keep as "N/A"
                
                results.append(result)
    else:
        logging.error(f"Invalid input path: {args.input_path}. Please provide a valid file or directory.")
        return

    if args.output_csv:
        if results:
            with open(args.output_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"✅ Prediction results saved to {args.output_csv}")
        else:
            print("⚠️ No valid predictions were made to save.")
    else:
        for result in results:
            print(f"--- Prediction for {result['file_name']} ---")
            print(f"Predicted Hemoglobin (Hb): {result['predicted_hb']} g/dL")
            print(f"Predicted Glucose: {result['predicted_glucose']} mg/dL")


if __name__ == "__main__":
    main()