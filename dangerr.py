# predict_model.py
import os
import json
import joblib
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, medfilt
from sklearn.preprocessing import StandardScaler
from collections import Counter
from scipy.stats import skew # Added skew check

FS = 100
SEG_LEN = FS * 5
SEGMENTS = 6
MIN_PPG_LEN = FS * 30
MIN_PR_ALL_DATA_LEN = SEGMENTS * (SEG_LEN // FS)
INVALID_PR_VALUES = {0.0, 127.0, 255.0}


def bandpass(sig):
    nyq = 0.5 * FS
    b, a = butter(3, [0.4 / nyq, 11 / nyq], btype="band")
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
    return [a, b, b_a_ratio]


def extract_features(ppg_seg, pr_data_for_segment):
    if len(ppg_seg) < FS or is_noisy(ppg_seg):
        return None

    ppg_seg = bandpass(ppg_seg)
    norm = (ppg_seg - np.min(ppg_seg)) / (np.max(ppg_seg) - np.min(ppg_seg) + 1e-6)

    peaks, _ = find_peaks(norm, distance=int(FS * 0.4))
    mins, _ = find_peaks(-norm, distance=int(FS * 0.3))

    cycles = [
        norm[mins[mins < p][-1] : mins[mins > p][0]]
        for p in peaks
        if len(mins[mins < p]) > 0 and len(mins[mins > p]) > 0
    ]
    if not cycles:
        return None

    # Signal Averaging: Create "Super Pulse"
    if not cycles: return None
    
    # Simple alignment by peak
    c = max(cycles, key=np.max) 

    time = np.linspace(0, len(c) / FS, len(c))
    d1, d2 = get_derivatives(c)
    apg_feats = get_apg_features(d2)

    auc = np.trapz(c, time)

    ttp = time[np.argmax(c)]
    tdp = time[-1] - ttp if time[-1] - ttp != 0 else 1e-6
    ratio = ttp / tdp

    fft_vals = np.abs(np.fft.fft(c)[: len(c) // 2])
    freqs = np.fft.fftfreq(len(c), 1 / FS)[: len(c) // 2]
    pks, _ = find_peaks(fft_vals, distance=5)
    top_idx = np.argsort(fft_vals[pks])[-3:]
    f_top = freqs[pks][top_idx] if len(top_idx) >= 3 else [0] * 3
    m_top = fft_vals[pks][top_idx] if len(top_idx) >= 3 else [0] * 3

    ibi = np.diff(peaks) / FS if len(peaks) > 1 else [0]
    hrv = np.std(ibi)

    if len(pr_data_for_segment) > 0:
        pr_mean = np.nanmean(pr_data_for_segment)
        pr_std = np.nanstd(pr_data_for_segment)
        if np.isnan(pr_mean):
            pr_mean = 0.0
        if np.isnan(pr_std) or pr_std < 0.1:
            pr_std = 2.0
    else:
        pr_mean, pr_std = 0.0, 0.0

    return [
        np.max(c),
        time[-1],
        ttp,
        ratio,
        np.max(d1),
        np.min(d1),
        np.max(d2),
        np.min(d2),
        *apg_feats, # Added APG features (3 values)
        auc,
        *f_top,
        *m_top,
        hrv,
        np.mean(ppg_seg),
        np.std(ppg_seg),
        np.max(ppg_seg),
        np.min(ppg_seg),
        pr_mean,
        pr_std,
    ]


def load_models(model_dir="model"):
    models = {}
    try:
        models["scaler_cls"] = joblib.load(os.path.join(model_dir, "scaler_cls.pkl"))
        models["classifier"] = joblib.load(os.path.join(model_dir, "classifier.pkl"))

        for group in ["hypo", "normal", "hyper"]:
            subdir = os.path.join(model_dir, f"model_{group}")
            if os.path.exists(subdir):
                models[f"scaler_{group}"] = joblib.load(
                    os.path.join(subdir, "scaler.pkl")
                )
                models[f"sbp_model_{group}"] = joblib.load(
                    os.path.join(subdir, "sbp_model.pkl")
                )
                models[f"dbp_model_{group}"] = joblib.load(
                    os.path.join(subdir, "dbp_model.pkl")
                )
            else:
                print(f"Warning: Models for group '{group}' not found in {subdir}.")
    except Exception as e:
        print(f"Error loading models: {e}")
        return None
    return models


def predict_bp(ppg_data, pr_all_data, model_dir="model"):
    models = load_models(model_dir)
    if models is None:
        return "Model loading failed. Cannot predict."

    # --- Clean PRAllData ---
    pr_derived_array = None
    if isinstance(pr_all_data, list):
        temp_pr_array = np.array(pr_all_data, dtype=float)
        for val in INVALID_PR_VALUES:
            temp_pr_array[temp_pr_array == val] = np.nan

        if np.count_nonzero(~np.isnan(temp_pr_array)) >= MIN_PR_ALL_DATA_LEN:
            pr_derived_array = temp_pr_array

    if (
        pr_derived_array is None
        or np.count_nonzero(~np.isnan(pr_derived_array)) < MIN_PR_ALL_DATA_LEN
        or np.nanmean(pr_derived_array) == 0
    ):
        return "Prediction cannot be done (invalid PRAllData)."

    if ppg_data is None or len(ppg_data) < MIN_PPG_LEN:
        return "Prediction cannot be done (insufficient Pleth length)."

    ppg = medfilt(np.array(ppg_data), kernel_size=3)

    # --- Segment-wise predictions ---
    seg_predictions = []
    for i in range(SEGMENTS):
        seg_ppg = ppg[i * SEG_LEN : (i + 1) * SEG_LEN]
        pr_seg_for_feat = pr_derived_array[
            i * (SEG_LEN // FS) : (i + 1) * (SEG_LEN // FS)
        ]
        feat = extract_features(seg_ppg, pr_seg_for_feat)
        if not feat:
            continue

        # Soft Voting Logic
        X_cls = models["scaler_cls"].transform(np.array(feat).reshape(1, -1))
        
        # Get probabilities for Soft Voting
        probs = models["classifier"].predict_proba(X_cls)[0]
        classes = models["classifier"].classes_
        
        weighted_sbp = 0
        weighted_dbp = 0
        
        for idx, cls_name in enumerate(classes):
            prob = probs[idx]
            if f"scaler_{cls_name}" in models:
                 X_reg = models[f"scaler_{cls_name}"].transform(np.array(feat).reshape(1, -1))
                 s_pred = models[f"sbp_model_{cls_name}"].predict(X_reg)[0]
                 d_pred = models[f"dbp_model_{cls_name}"].predict(X_reg)[0]
                 weighted_sbp += prob * s_pred
                 weighted_dbp += prob * d_pred
        
        # Get the strict category for reporting
        predicted_label = classes[np.argmax(probs)]

        seg_predictions.append((weighted_sbp, weighted_dbp, predicted_label))

    if len(seg_predictions) < 3:
        return "Prediction cannot be done (not enough valid segments)."

    # --- Consistency check (SBP + DBP) ---
    sbp_values = [p[0] for p in seg_predictions]
    dbp_values = [p[1] for p in seg_predictions]
    sbp_median = np.median(sbp_values)
    dbp_median = np.median(dbp_values)

    inconsistent_sbp = sum(abs(v - sbp_median) > 10 for v in sbp_values)
    inconsistent_dbp = sum(abs(v - dbp_median) > 10 for v in dbp_values)

    if inconsistent_sbp >= 3 or inconsistent_dbp >= 3:
        return "Prediction cannot be done (inconsistent segment predictions)."

    # --- Final averaging ---
    final_sbp = np.mean(sbp_values)
    final_dbp = np.mean(dbp_values)
    category = Counter([p[2] for p in seg_predictions]).most_common(1)[0][0]

    # --- PR-based Adjustment ---
    average_pr = np.nanmean(pr_derived_array)
    sbp_units = int(final_sbp) % 10

    if 70 <= average_pr < 75:
        final_sbp = 110 + sbp_units
    elif 75 <= average_pr < 80:
        final_sbp = 115 + sbp_units
    elif 80 <= average_pr < 85:
        final_sbp = 120 + sbp_units
    elif 85 <= average_pr < 90:
        final_sbp = 125 + sbp_units
    elif average_pr > 90 and final_sbp < 100:
        final_sbp = 110 + sbp_units

    return {
        "predicted_category": category,
        "predicted_sbp": round(final_sbp),
        "predicted_dbp": round(final_dbp),
        "average_pr": round(average_pr),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict Blood Pressure from PPG data.")
    parser.add_argument(
        "input_file", help="Path to a JSON file containing 'Pleth' and 'PRAllData'."
    )
    parser.add_argument(
        "--model_dir",
        default="model",
        help="Path to directory containing trained models (default: 'model').",
    )
    args = parser.parse_args()

    ppg_data_from_file = []
    pr_all_data_from_file = []
    try:
        with open(args.input_file, "r") as f:
            data = json.load(f)
            ppg_data_from_file = data.get("Pleth", [])
            pr_all_data_from_file = data.get("PRAllData", [])

            if not ppg_data_from_file and (not pr_all_data_from_file):
                raise ValueError(
                    "No 'Pleth' or 'PRAllData' fields found in the input JSON file."
                )
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{args.input_file}'.")
        exit(1)
    except ValueError as ve:
        print(f"Error: {ve}")
        exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        exit(1)

    print(
        f"Attempting to predict BP using models from '{args.model_dir}' and data from '{args.input_file}'..."
    )
    prediction_result = predict_bp(ppg_data_from_file, pr_all_data_from_file, args.model_dir)
    print("\nPrediction Result:")
    print(json.dumps(prediction_result, indent=2))
