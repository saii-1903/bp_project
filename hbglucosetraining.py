import numpy as np
import joblib
import os, json, argparse
from glob import glob
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.signal import medfilt, butter, filtfilt, find_peaks
from scipy.fft import fft

FS = 120
SEGMENT_SAMPLES = 200
SEGMENTS_PER_FILE = 6
MIN_SAMPLES = 1200

def bandpass_filter(signal):
    b, a = butter(3, [0.4 / (FS/2), 11 / (FS/2)], btype='band')
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
    c = max(cycles, key=np.max)
    time = np.linspace(0, len(c) / FS, len(c))
    d1 = np.gradient(c)
    d2 = np.gradient(d1)
    auc = np.trapz(c, time)
    pw = time[-1]
    ttp = time[np.argmax(c)]
    tdp = pw - ttp
    ratio = ttp / tdp if tdp else 0
    fft_vals = np.abs(fft(c)[:len(c)//2])
    freqs = np.fft.fftfreq(len(c), d=1 / FS)[:len(c)//2]
    peak_idxs, _ = find_peaks(fft_vals, distance=5)
    sorted_idxs = np.argsort(fft_vals[peak_idxs])[-3:]
    top_freqs = freqs[peak_idxs][sorted_idxs] if len(sorted_idxs) >= 3 else [0]*3
    top_mags = fft_vals[peak_idxs][sorted_idxs] if len(sorted_idxs) >= 3 else [0]*3
    ibi = np.diff(peaks) / FS if len(peaks) > 1 else [0]
    hrv_std = np.std(ibi) if len(ibi) > 1 else 0
    return [
        np.max(c), pw, ttp, ratio,
        np.max(d1), np.min(d1), np.max(d2), np.min(d2),
        auc,
        top_freqs[0], top_mags[0],
        top_freqs[1], top_mags[1],
        top_freqs[2], top_mags[2],
        hrv_std,
        np.mean(signal), np.std(signal), np.max(signal), np.min(signal)
    ]

def load_pleth(val):
    # If val is a string path to .npy, load it; if it's a list, convert to np.array
    if isinstance(val, str) and val.endswith('.npy'):
        return np.load(val)
    elif isinstance(val, list):
        return np.array(val)
    else:
        return np.array(eval(val))  # if CSV stores as stringified list


def main(input_path, model_dir):
    X, y_hb, y_glu = [], [], []
    if os.path.isdir(input_path):
        files = sorted(glob(os.path.join(input_path, "*.json")))
    else:
        files = [input_path]

    for file in files:
        with open(file) as f:
            d = json.load(f)
        pleth = d.get('Pleth')
        if pleth is None:
            print(f"Skipping {file}: No Pleth data.")
            continue
        pleth = medfilt(np.array(pleth), kernel_size=3)
        if len(pleth) < MIN_SAMPLES:
            print(f"Skipping {file}: Not enough Pleth samples (found {len(pleth)}, need at least {MIN_SAMPLES}).")
            continue
        age = d.get('Age', 35)
        gender = 1 if str(d.get('Gender', '')).lower() == "male" else 0
        hb = d.get('Hb')
        glu = d.get('Glucose')
        if glu is None:
            glu = d.get('Blood Glucose')
        if hb is None or glu is None:
            print(f"Skipping {file}: Missing Hb or Glucose value.")
            continue
        num_segments = SEGMENTS_PER_FILE
        seg_count = 0
        feat_count = 0
        seg_stds = []
        for i in range(num_segments):
            seg = pleth[i * SEGMENT_SAMPLES: (i + 1) * SEGMENT_SAMPLES]
            if len(seg) < SEGMENT_SAMPLES:
                continue
            seg_count += 1
            seg_std = np.std(seg)
            seg_stds.append(seg_std)
            feats = extract_features(seg)
            # Accept segment if any feature is non-zero
            if not any(f != 0 for f in feats):
                print(f"  Segment {i}: std={seg_std:.6f}, features={feats}")
                continue
            feat_count += 1
            full_feat = feats + [age, gender]
            X.append(full_feat)
            y_hb.append(hb)
            y_glu.append(glu)
        print(f"{file}: {seg_count} segments, {feat_count} valid features extracted. Segment stds: {seg_stds}")
        if feat_count == 0:
            print(f"Warning: All segments skipped for {file} due to noisy or invalid data.")

    X = np.array(X)
    y_hb = np.array(y_hb)
    y_glu = np.array(y_glu)

    if X.size == 0 or y_hb.size == 0 or y_glu.size == 0:
        print("❌ No valid training samples found. Check your input JSONs for sufficient Pleth data and valid Hb/Glucose values.")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    hb_model = RandomForestRegressor(n_estimators=100, random_state=42)
    glu_model = RandomForestRegressor(n_estimators=100, random_state=42)
    hb_model.fit(X_scaled, y_hb)
    glu_model.fit(X_scaled, y_glu)

    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(hb_model, os.path.join(model_dir, "hb_regressor.pkl"))
    joblib.dump(glu_model, os.path.join(model_dir, "glucose_regressor.pkl"))
    print(f"Training complete. Models saved to {model_dir}/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to JSON file or folder of JSONs")
    parser.add_argument("model_dir", help="Directory to save trained models")
    args = parser.parse_args()
    main(args.input_path, args.model_dir)
