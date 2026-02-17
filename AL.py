import os
import json
import numpy as np
import joblib
import csv
import argparse
from glob import glob
from scipy.signal import butter, filtfilt, find_peaks, medfilt
from scipy.fft import fft
from scipy.stats import entropy, kurtosis

FS = 100
SEGMENT_SECONDS = 5
SEGMENT_SAMPLES = FS * SEGMENT_SECONDS
SEGMENTS_PER_FILE = 6
MAX_ALLOWED_VARIATION = 10
INVALID_BP_VALUES = {-1, 1, 200, 202, 400, 404}

def get_bp_class(sbp, dbp):
    if sbp is None or dbp is None:
        return "normal"  # fallback default
    if sbp < 90 or dbp < 60:
        return "hypo"
    elif sbp > 140 or dbp > 90:
        return "hyper"
    else:
        return "normal"

def bandpass_filter(signal, lowcut=0.4, highcut=11, fs=FS, order=3):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, signal)

def extract_features(signal):
    if len(signal) < FS:
        return [0] * 20
    signal = bandpass_filter(signal)
    norm = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-6)
    peaks, _ = find_peaks(norm, distance=int(FS * 0.4))
    mins, _ = find_peaks(-norm, distance=int(FS * 0.3))
    cycles = []
    for p in peaks:
        left = mins[mins < p]
        right = mins[mins > p]
        if len(left) > 0 and len(right) > 0:
            cycles.append(norm[left[-1]:right[0]])
    if not cycles:
        return [0] * 20
    cycle = max(cycles, key=lambda x: np.max(x))
    time = np.linspace(0, len(cycle) / FS, len(cycle))
    d1 = np.gradient(cycle)
    d2 = np.gradient(d1)
    auc = np.trapz(cycle, time)
    pw = time[-1]
    ttp = time[np.argmax(cycle)]
    tdp = pw - ttp
    ratio = ttp / tdp if tdp else 0
    freqs = np.fft.fftfreq(len(cycle), d=1 / FS)
    fft_vals = np.abs(fft(cycle)[:len(cycle)//2])
    freqs = freqs[:len(cycle)//2]
    peak_idxs, _ = find_peaks(fft_vals, distance=5)
    sorted_idxs = np.argsort(fft_vals[peak_idxs])[-3:]
    top_freqs = freqs[peak_idxs][sorted_idxs] if len(sorted_idxs) >= 3 else [0, 0, 0]
    top_mags = fft_vals[peak_idxs][sorted_idxs] if len(sorted_idxs) >= 3 else [0, 0, 0]
    ibi_list = np.diff(peaks) / FS if len(peaks) > 1 else [0]
    hrv_std = np.std(ibi_list) if len(ibi_list) > 1 else 0
    return [
        np.max(cycle), pw, ttp, ratio,
        np.max(d1), np.min(d1), np.max(d2), np.min(d2),
        auc,
        top_freqs[0], top_mags[0],
        top_freqs[1], top_mags[1],
        top_freqs[2], top_mags[2],
        hrv_std,
        np.mean(signal), np.std(signal), np.max(signal), np.min(signal)
    ]

def load_model(model_dir):
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    sbp_model = joblib.load(os.path.join(model_dir, "sbp_model.pkl"))
    dbp_model = joblib.load(os.path.join(model_dir, "dbp_model.pkl"))
    return scaler, sbp_model, dbp_model

def predict_file(file_path, model_dict):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [l for l in f if not l.strip().startswith("//")]
        data = json.loads("\n".join(lines))

    pleth = np.array(data.get("Pleth", []))
    sbp_true = data.get("BPSystolic", data.get("SBP", None))
    dbp_true = data.get("BPDiastolic", data.get("DBP", None))

    bp_class = get_bp_class(sbp_true, dbp_true)
    if bp_class not in model_dict:
        return None

    scaler, sbp_model, dbp_model = model_dict[bp_class]
    pleth = medfilt(pleth, kernel_size=3)

    if len(pleth) < SEGMENTS_PER_FILE * SEGMENT_SAMPLES:
        return None

    predictions = []
    for i in range(SEGMENTS_PER_FILE):
        segment = pleth[i * SEGMENT_SAMPLES: (i + 1) * SEGMENT_SAMPLES]
        feats = extract_features(segment)
        if all(f == 0 for f in feats):
            continue
        X = scaler.transform([feats])
        sbp = float(sbp_model.predict(X)[0])
        dbp = float(dbp_model.predict(X)[0])
        predictions.append((sbp, dbp))

    if len(predictions) < 3:
        return None

    # Filter by trending: keep only stable segments
    sbps = [p[0] for p in predictions]
    dbps = [p[1] for p in predictions]
    median_sbp = np.median(sbps)
    median_dbp = np.median(dbps)

    good_preds = [(s, d) for s, d in predictions if abs(s - median_sbp) <= MAX_ALLOWED_VARIATION and abs(d - median_dbp) <= MAX_ALLOWED_VARIATION]

    if len(good_preds) < 3:
        return None

    good_sbp = [p[0] for p in good_preds]
    good_dbp = [p[1] for p in good_preds]

    return {
        "file": os.path.basename(file_path),
        "class": bp_class,
        "avg_sbp": round(np.mean(good_sbp), 2),
        "avg_dbp": round(np.mean(good_dbp), 2),
        "true_sbp": sbp_true,
        "true_dbp": dbp_true
    }

def main(input_dir, model_base_dir, output_csv):
    model_dict = {}
    for cls in ["hypo", "normal", "hyper"]:
        model_path = os.path.join(model_base_dir, f"model_{cls}")
        if os.path.exists(model_path):
            model_dict[cls] = load_model(model_path)

    files = glob(os.path.join(input_dir, "**", "*.json"), recursive=True)
    print(f"🔍 Found {len(files)} JSON files.")

    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "class", "avg_sbp", "avg_dbp", "true_sbp", "true_dbp", "sbp_error", "dbp_error"])

        for file_path in files:
            result = predict_file(file_path, model_dict)
            if result:
                sbp_err = abs(result["avg_sbp"] - result["true_sbp"]) if result["true_sbp"] not in INVALID_BP_VALUES else ""
                dbp_err = abs(result["avg_dbp"] - result["true_dbp"]) if result["true_dbp"] not in INVALID_BP_VALUES else ""
                writer.writerow([
                    result["file"], result["class"],
                    result["avg_sbp"], result["avg_dbp"],
                    result["true_sbp"], result["true_dbp"],
                    sbp_err, dbp_err
                ])
                print(f"✅ {result['file']}: {result['class']} SBP={result['avg_sbp']}, DBP={result['avg_dbp']}")
            else:
                print(f"❌ Skipped {os.path.basename(file_path)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict SBP/DBP from JSON PPG files using class-specific models.")
    parser.add_argument("input_dir", help="Input folder with PPG JSON files")
    parser.add_argument("model_dir", help="Directory with model_hypo, model_normal, model_hyper")
    parser.add_argument("--output_csv", default="bp_predictions.csv", help="Path to save predictions")
    args = parser.parse_args()
    main(args.input_dir, args.model_dir, args.output_csv)
