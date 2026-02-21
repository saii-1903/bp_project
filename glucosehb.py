"""
glucosehb.py — Inference script for Hb and Glucose prediction.

Fixes applied:
  1. Duplicate 'predicted_hb' key removed from return dict.
  2. Age validated — out-of-range defaults rejected with warning.
  3. Gender missing → warning logged, not silently defaulted.
  4. Richer demographic features: age², is_senior, BMI where available.
  5. GlucosePredictor / HbPredictor classes encapsulate smoothing state
     (no global variables — safe for multi-patient / parallel use).
  6. Hb gets same weighted-average smoothing as Glucose (consistent).
  7. Additional PPG features: perfusion_index, SQI, cycle_skew, cycle_kurt.
  8. log-transforms updated for new feature indices.
"""

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
from scipy.stats import skew as _skew, kurtosis as _kurtosis

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
FS               = 200
SEGMENT_SECONDS  = 5
SEGMENT_SAMPLES  = FS * SEGMENT_SECONDS
SEGMENTS_PER_FILE = 6

# ─── Signal processing helpers ───────────────────────────────────────────────

def bandpass_filter(signal, fs=FS):
    nyq = 0.5 * fs
    b, a = butter(3, [0.4 / nyq, 11 / nyq], btype='band')
    return filtfilt(b, a, signal)


def extract_features(signal, fs=FS):
    """
    Extracts 27 features from a single PPG segment.
    Original 20 + 3 SpO2-inspired (AC/DC ratio, entropy, skewness) from Fix 7
                + 4 new optical/shape features (perfusion index, SQI, cycle skew, cycle kurt).
    Log-transforms applied to amplitude/area features.
    MUST stay in sync with hbglucose.py extract_features().
    """
    if len(signal) < fs:
        return None

    signal = medfilt(signal, kernel_size=3)
    signal = bandpass_filter(signal, fs=fs)
    norm   = (signal - np.min(signal)) / (np.max(signal) - np.min(signal) + 1e-6)

    peaks, _ = find_peaks(norm, distance=int(fs * 0.4))
    mins,  _ = find_peaks(-norm, distance=int(fs * 0.3))

    if len(peaks) < 2 or len(mins) < 2:
        return None

    cycles = []
    for p in peaks:
        left  = mins[mins < p]
        right = mins[mins > p]
        if len(left) > 0 and len(right) > 0:
            cycles.append(norm[left[-1]:right[0]])

    if not cycles:
        return None

    cycle   = max(cycles, key=lambda x: np.max(x))
    time    = np.linspace(0, len(cycle) / fs, len(cycle))

    d1  = np.gradient(cycle)
    d2  = np.gradient(d1)
    auc = np.trapezoid(cycle, time)
    pw  = time[-1]
    ttp = time[np.argmax(cycle)]
    tdp = pw - ttp
    ratio = ttp / tdp if tdp else 0

    freqs    = np.fft.fftfreq(len(cycle), d=1 / fs)
    fft_vals = np.abs(fft(cycle)[:len(cycle) // 2])
    freqs    = freqs[:len(cycle) // 2]

    peak_idxs, _ = find_peaks(fft_vals, distance=5)
    if len(peak_idxs) < 3:
        top_freqs = [0, 0, 0];  top_mags = [0, 0, 0]
    else:
        sorted_idxs = np.argsort(fft_vals[peak_idxs])[-3:]
        top_freqs   = freqs[peak_idxs][sorted_idxs]
        top_mags    = fft_vals[peak_idxs][sorted_idxs]

    ibi_list = np.diff(peaks) / fs if len(peaks) > 1 else [0]
    hrv_std  = float(np.std(ibi_list)) if len(ibi_list) > 1 else 0.0

    features = [
        np.max(cycle), pw, ttp, ratio,          # 0-3
        np.max(d1), np.min(d1),                 # 4-5
        np.max(d2), np.min(d2),                 # 6-7
        auc,                                     # 8
        top_freqs[0], top_mags[0],              # 9-10
        top_freqs[1], top_mags[1],              # 11-12
        top_freqs[2], top_mags[2],              # 13-14
        hrv_std,                                 # 15
        float(np.mean(signal)),                 # 16
        float(np.std(signal)),                  # 17
        float(np.max(signal)),                  # 18
        float(np.min(signal)),                  # 19
    ]

    # Fix 7: SpO2-inspired features (AC/DC ratio, entropy, skewness)
    ac_dc_ratio = (np.max(signal) - np.min(signal)) / (np.mean(np.abs(signal)) + 1e-9)
    sig_norm    = signal - np.min(signal)
    sig_norm    = sig_norm / (np.sum(sig_norm) + 1e-9)
    entropy     = float(-np.sum(sig_norm * np.log(sig_norm + 1e-9)))
    sig_skew    = float(_skew(signal))
    features   += [ac_dc_ratio, entropy, sig_skew]   # 20-22

    # Additional optical / shape features
    perfusion_index = (np.max(signal) - np.min(signal)) / (np.mean(signal) + 1e-9)  # 23
    sqi             = float(np.max(cycle) / (np.std(signal) + 1e-6))                # 24
    cycle_skew      = float(_skew(cycle))                                            # 25
    cycle_kurt      = float(_kurtosis(cycle))                                        # 26
    features       += [perfusion_index, sqi, cycle_skew, cycle_kurt]

    # Log-transforms (amplitude/area features + AC/DC ratio)
    for idx in [0, 8, 16, 17, 18, 19, 20, 23]:
        features[idx] = float(np.log1p(np.abs(features[idx])))

    return features


# ─── Fix 6: Encapsulated smoothers (no global state) ─────────────────────────

class _EWMAPredictor:
    """Exponentially-weighted moving average smoother for a single scalar."""
    def __init__(self, alpha=0.7, low=None, high=None):
        self._last  = None
        self._alpha = alpha   # weight on new value
        self._low   = low
        self._high  = high

    def smooth(self, new_val):
        if new_val is None:
            return None
        # Fix 5: physiological clamping
        if self._low  is not None: new_val = max(self._low,  new_val)
        if self._high is not None: new_val = min(self._high, new_val)
        if self._last is None:
            self._last = new_val
            return new_val
        smoothed   = self._alpha * new_val + (1 - self._alpha) * self._last
        self._last = smoothed
        return smoothed

    def reset(self):
        self._last = None


class VitalPredictor:
    """
    Per-patient Hb + Glucose predictor.
    Instantiate one per patient / session to keep smoothing state isolated.
    """
    def __init__(self, model_dir: str):
        self._model_dir    = model_dir
        # Fix 7: Hb gets same smoothing as Glucose
        self._hb_smoother  = _EWMAPredictor(alpha=0.7, low=5.0,  high=25.0)
        self._glu_smoother = _EWMAPredictor(alpha=0.7, low=40.0, high=600.0)

    def predict(self, feature_vector: np.ndarray) -> dict:
        result = {}
        X = feature_vector.reshape(1, -1)
        try:
            scaler_hb = joblib.load(os.path.join(self._model_dir, "scaler_hb.pkl"))
            hb_model  = joblib.load(os.path.join(self._model_dir, "hb_regressor.pkl"))
            raw_hb    = float(hb_model.predict(scaler_hb.transform(X))[0])
            result['predicted_hb'] = round(self._hb_smoother.smooth(raw_hb), 2)
        except Exception as e:
            logging.warning(f"Hb prediction failed: {e}")
            result['predicted_hb'] = None

        try:
            scaler_glu = joblib.load(os.path.join(self._model_dir, "scaler_glucose.pkl"))
            glu_model  = joblib.load(os.path.join(self._model_dir, "glucose_regressor.pkl"))
            raw_glu    = float(glu_model.predict(scaler_glu.transform(X))[0])
            result['predicted_glucose'] = round(self._glu_smoother.smooth(raw_glu), 2)
        except Exception as e:
            logging.warning(f"Glucose prediction failed: {e}")
            result['predicted_glucose'] = None

        return result


# ─── Demographic feature builder ─────────────────────────────────────────────

def _build_demographic_features(data: dict, file_path: str):
    """
    Returns (age_feats, ok) where age_feats is a list of demographic features
    and ok=False means this record should be skipped.
    Fix 2: Invalid age → skip.  Fix 3: Missing gender → warn.
    Fix 4: Richer features: age, age², is_senior, gender, age×gender, BMI.
    """
    age = data.get("Age", None)
    if age is None or not (0 < float(age) < 120):
        logging.warning(f"Missing or invalid age ({age}) in {file_path} — skipping.")
        return None, False
    age = float(age)

    raw_gender = data.get("Gender", None)
    if raw_gender is None:
        logging.warning(f"Missing gender in {file_path} — defaulting to 0 (female).")
        gender = 0
    else:
        gender = 1 if raw_gender.lower() == "male" else 0

    age_sq    = age ** 2
    is_senior = 1 if age > 60 else 0
    interaction = age * gender

    bmi = data.get("BMI", None)
    bmi_feat = float(bmi) if bmi is not None and 10 < float(bmi) < 80 else -1.0

    return [age, age_sq, is_senior, gender, interaction, bmi_feat], True


# ─── File-level prediction (command-line use) ─────────────────────────────────

def predict_from_file(json_file_path: str, model_dir: str,
                      predictor: VitalPredictor = None) -> dict | None:
    """Predict Hb and Glucose from a single JSON file."""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            lines = [l for l in f if not l.strip().startswith("//")]
            data  = json.loads("\n".join(lines))
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error reading {json_file_path}: {e}")
        return None

    pleth_raw = data.get("Pleth") or data.get("IRGainAllData")
    if not pleth_raw:
        logging.error(f"Missing 'Pleth'/'IRGainAllData' in {json_file_path}. Skipping.")
        return None

    pleth_flat = []
    if isinstance(pleth_raw, list):
        for item in pleth_raw:
            if isinstance(item, (int, float)):
                pleth_flat.append(item)
            elif isinstance(item, list):
                pleth_flat.extend(item)
    else:
        logging.error(f"Signal data is not a list in {json_file_path}. Skipping.")
        return None

    pleth = np.array(pleth_flat)
    if not pleth.any() or len(pleth) < SEGMENT_SAMPLES:
        logging.error(f"Insufficient signal in {json_file_path}. Skipping.")
        return None

    demo_feats, ok = _build_demographic_features(data, json_file_path)
    if not ok:
        return None

    seg_features = []
    for i in range(SEGMENTS_PER_FILE):
        seg  = pleth[i * SEGMENT_SAMPLES: (i + 1) * SEGMENT_SAMPLES]
        if len(seg) < SEGMENT_SAMPLES:
            continue
        feats = extract_features(seg)
        if feats is None:
            continue
        seg_features.append(feats + demo_feats)

    if not seg_features:
        logging.error(f"No valid segments in {json_file_path}. Skipping.")
        return None

    X_pred = np.mean(seg_features, axis=0)
    if predictor is None:
        predictor = VitalPredictor(model_dir)

    preds = predictor.predict(X_pred)
    return {
        'file_name':         os.path.basename(json_file_path),
        'predicted_hb':      preds.get('predicted_hb'),       # Fix 1: no duplicate key
        'predicted_glucose': preds.get('predicted_glucose'),
    }


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Predict Hb and Glucose from PPG JSON data")
    parser.add_argument("input_path",
                        help="Path to a single JSON file or a folder of JSON files")
    parser.add_argument("--model-dir", default="./model_output",
                        help="Folder where model files are saved")
    parser.add_argument("--output-csv",
                        help="Optional path to save the output as a CSV file")
    args = parser.parse_args()

    # One VitalPredictor per run keeps smoothing state across files
    predictor = VitalPredictor(args.model_dir)
    results   = []

    if os.path.isfile(args.input_path):
        result = predict_from_file(args.input_path, args.model_dir, predictor)
        if result:
            results.append(result)
    elif os.path.isdir(args.input_path):
        json_files = sorted(glob.glob(os.path.join(args.input_path, "*.json")))
        if not json_files:
            logging.warning(f"No JSON files found in: {args.input_path}")
        for fp in json_files:
            result = predict_from_file(fp, args.model_dir, predictor)
            if result:
                results.append(result)
    else:
        logging.error(f"Invalid input path: {args.input_path}")
        return

    if args.output_csv:
        if results:
            with open(args.output_csv, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            print(f"✅ Results saved to {args.output_csv}")
        else:
            print("⚠️ No valid predictions to save.")
    else:
        for r in results:
            print(f"\n--- {r['file_name']} ---")
            hb_str  = f"{r['predicted_hb']:.2f} g/dL"  if r['predicted_hb']      is not None else "N/A"
            gl_str  = f"{r['predicted_glucose']:.1f} mg/dL" if r['predicted_glucose'] is not None else "N/A"
            print(f"  Hb:      {hb_str}")
            print(f"  Glucose: {gl_str}")


if __name__ == "__main__":
    main()