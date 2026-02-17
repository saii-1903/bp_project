import os, json, csv, argparse, logging
import numpy as np
import joblib
from glob import glob
from scipy.signal import butter, filtfilt, find_peaks, medfilt
from scipy.fft import fft

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

FS = 120
SEGMENT_SECONDS = 5
SEGMENT_SAMPLES = FS * SEGMENT_SECONDS
SEGMENTS_PER_FILE = 6

def bandpass_filter(signal):
    nyq = 0.5 * FS
    b, a = butter(3, [0.4 / nyq, 11 / nyq], btype='band')
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

def predict(input_dir, model_dir, output_csv):
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    hb_model = joblib.load(os.path.join(model_dir, "hb_regressor.pkl"))
    glucose_model = joblib.load(os.path.join(model_dir, "glucose_regressor.pkl"))

    files = sorted(glob(os.path.join(input_dir, "*.json")))

    with open(output_csv, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow(["file", "Name", "Age", "Gender", "Hb", "Glucose"])

        for file_path in files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = [l for l in f if not l.strip().startswith("//")]
                    data = json.loads("\n".join(lines))
            except Exception as e:
                logging.warning(f"Failed to parse {file_path}: {e}")
                continue

            pleth = np.array(data.get("Pleth", []))
            if len(pleth) < SEGMENTS_PER_FILE * SEGMENT_SAMPLES:
                logging.warning(f"Skipped {file_path} (not enough data)")
                continue

            pleth = medfilt(pleth, kernel_size=3)
            age = data.get("Age", -1)
            gender = 1 if data.get("Gender", "").lower() == "male" else 0
            name = data.get("Name", "")

            preds = []
            for i in range(SEGMENTS_PER_FILE):
                seg = pleth[i * SEGMENT_SAMPLES : (i + 1) * SEGMENT_SAMPLES]
                feats = extract_features(seg)
                if all(f == 0 for f in feats): continue
                full_feat = np.array(feats + [age, gender]).reshape(1, -1)
                X = scaler.transform(full_feat)
                hb = hb_model.predict(X)[0]
                glu = glucose_model.predict(X)[0]
                preds.append((hb, glu))

            if len(preds) < 3:
                logging.warning(f"{file_path}: Not enough valid predictions")
                continue

            hb_vals = [p[0] for p in preds]
            glu_vals = [p[1] for p in preds]
            hb_median = np.median(hb_vals)
            glu_median = np.median(glu_vals)

            stable_preds = [
                (hb, glu) for hb, glu in preds
                if abs(hb - hb_median) <= 2 and abs(glu - glu_median) <= 10
            ]

            if len(stable_preds) >= 3:
                avg_hb = round(np.mean([x[0] for x in stable_preds]), 2)
                avg_glu = round(np.mean([x[1] for x in stable_preds]), 2)
                writer.writerow([os.path.basename(file_path), name, age, gender, avg_hb, avg_glu])
                logging.info(f"{file_path}: Hb={avg_hb} g/dL, Glucose={avg_glu} mg/dL")
            else:
                logging.info(f"{file_path}: Skipped due to unstable predictions")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict Hb and Glucose from PPG JSON")
    parser.add_argument("input_dir", help="Input folder with test JSONs")
    parser.add_argument("model_dir", help="Folder with saved model files")
    parser.add_argument("--output_csv", default="predictions.csv", help="CSV output file")
    args = parser.parse_args()
    predict(args.input_dir, args.model_dir, args.output_csv)
