# [TRAINING CODE — keep this in train_pipeline.py]
import os, json, joblib, argparse
import numpy as np
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.signal import butter, filtfilt, find_peaks, medfilt

FS = 120
SEG_LEN = FS * 5
SEGMENTS = 6
INVALID_BP = {-1, 1, 200, 202, 400, 404}

def bandpass(sig):
    b, a = butter(3, [0.4/(0.5*FS), 11/(0.5*FS)], btype='band')
    return filtfilt(b, a, sig)

def extract_features(sig):
    if len(sig) < FS: return [0]*20
    sig = bandpass(sig)
    norm = (sig - np.min(sig)) / (np.max(sig) - np.min(sig) + 1e-6)
    peaks, _ = find_peaks(norm, distance=int(FS*0.4))
    mins, _ = find_peaks(-norm, distance=int(FS*0.3))
    cycles = []
    for p in peaks:
        l, r = mins[mins < p], mins[mins > p]
        if l.size > 0 and r.size > 0:
            cycles.append(norm[l[-1]:r[0]])
    if not cycles: return [0]*20
    c = max(cycles, key=lambda x: np.max(x))
    time = np.linspace(0, len(c)/FS, len(c))
    d1, d2 = np.gradient(c), np.gradient(np.gradient(c))
    auc = np.trapz(c, time)
    pw, ttp = time[-1], time[np.argmax(c)]
    tdp = pw - ttp
    ratio = ttp / tdp if tdp else 0
    freqs = np.fft.fftfreq(len(c), 1/FS)[:len(c)//2]
    fft_vals = np.abs(np.fft.fft(c)[:len(c)//2])
    pks, _ = find_peaks(fft_vals, distance=5)
    top_idx = np.argsort(fft_vals[pks])[-3:]
    f_top = freqs[pks][top_idx] if len(top_idx) >= 3 else [0]*3
    m_top = fft_vals[pks][top_idx] if len(top_idx) >= 3 else [0]*3
    ibi = np.diff(peaks)/FS if len(peaks) > 1 else [0]
    hrv = np.std(ibi) if len(ibi) > 1 else 0
    return [
        np.max(c), pw, ttp, ratio,
        np.max(d1), np.min(d1), np.max(d2), np.min(d2),
        auc, *f_top, *m_top, hrv,
        np.mean(sig), np.std(sig), np.max(sig), np.min(sig)
    ]

def bp_class(sbp, dbp):
    if sbp < 90 or dbp < 60: return "hypo"
    elif sbp > 140 or dbp > 90: return "hyper"
    else: return "normal"

def train_pipeline(input_dir, output_dir):
    X_cls, y_cls = [], []
    by_class = {"hypo": [], "normal": [], "hyper": []}
    files = sorted(glob(os.path.join(input_dir, "**", "*.json"), recursive=True))

    for f in files:
        try:
            with open(f) as fp:
                d = json.loads("".join([l for l in fp if not l.strip().startswith("//")]))
            ppg = medfilt(np.array(d["Pleth"]), kernel_size=3)
            sbp, dbp = d.get("SBP") or d.get("BPSystolic"), d.get("DBP") or d.get("BPDiastolic")
            if sbp in INVALID_BP or dbp in INVALID_BP or sbp is None or dbp is None: continue
            if len(ppg) < SEGMENTS * SEG_LEN: continue
            label = bp_class(sbp, dbp)
            for i in range(SEGMENTS):
                seg = ppg[i*SEG_LEN:(i+1)*SEG_LEN]
                feat = extract_features(seg)
                if all(f == 0 for f in feat): continue
                X_cls.append(feat)
                y_cls.append(label)
                by_class[label].append((feat, sbp, dbp))
        except: continue

    print("📊 Class distribution:", {k: len(v) for k, v in by_class.items()})
    X_cls = np.array(X_cls)
    y_cls = np.array(y_cls)
    scaler_cls = StandardScaler().fit(X_cls)
    X_scaled = scaler_cls.transform(X_cls)
    clf = RandomForestClassifier(n_estimators=200, class_weight="balanced").fit(X_scaled, y_cls)
    joblib.dump(scaler_cls, os.path.join(output_dir, "scaler_cls.pkl"))
    joblib.dump(clf, os.path.join(output_dir, "classifier.pkl"))

    print("🎯 Classifier performance:\n", classification_report(y_cls, clf.predict(X_scaled)))

    for cls, items in by_class.items():
        X, y_s, y_d = zip(*items)
        X = np.array(X)
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        X_train, X_test, y_s_train, y_s_test = train_test_split(X_scaled, y_s, test_size=0.2)
        _, _, y_d_train, y_d_test = train_test_split(X_scaled, y_d, test_size=0.2)
        sbp_model = LGBMRegressor().fit(X_train, y_s_train)  # Use LightGBM for SBP
        dbp_model = XGBRegressor().fit(X_train, y_d_train)   # Keep XGBoost for DBP
        subdir = os.path.join(output_dir, f"model_{cls}")
        os.makedirs(subdir, exist_ok=True)
        joblib.dump(scaler, os.path.join(subdir, "scaler.pkl"))
        joblib.dump(sbp_model, os.path.join(subdir, "sbp_model.pkl"))
        joblib.dump(dbp_model, os.path.join(subdir, "dbp_model.pkl"))
        print(f"📈 {cls.upper()} MAE — SBP: {mean_absolute_error(y_s_test, sbp_model.predict(X_test)):.2f}, DBP: {mean_absolute_error(y_d_test, dbp_model.predict(X_test)):.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    train_pipeline(args.input_dir, args.output_dir)
