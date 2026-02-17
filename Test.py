import os, json, joblib, argparse, csv
import numpy as np
from glob import glob
from scipy.signal import butter, filtfilt, find_peaks, medfilt

FS = 120
SEG_LEN = 5 * FS
SEGMENTS = 6
MAX_VAR = 10
INVALID_BP = {-1, 1, 200, 202, 400, 404}
def bandpass(sig):
    b, a = butter(3, [0.4/(0.5*FS), 11/(0.5*FS)], btype='band')
    return filtfilt(b, a, sig)

def extract_features(sig):
    if len(sig) < FS: return [0]*20
    sig = bandpass(sig)
    norm = (sig - np.min(sig)) / (np.max(sig) - np.min(sig) + 1e-6)
    peaks, _ = find_peaks(norm, distance=int(FS * 0.4))
    mins, _ = find_peaks(-norm, distance=int(FS * 0.3))
    cycles = []
    for p in peaks:
        l = mins[mins < p]
        r = mins[mins > p]
        if l.size > 0 and r.size > 0:
            cycles.append(norm[l[-1]:r[0]])
    if not cycles: return [0]*20
    c = max(cycles, key=lambda x: np.max(x))
    time = np.linspace(0, len(c)/FS, len(c))
    d1, d2 = np.gradient(c), np.gradient(np.gradient(c))
    auc = np.trapz(c, time)
    pw = time[-1]
    ttp = time[np.argmax(c)]
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

def load_models(model_dir):
    cls = joblib.load(os.path.join(model_dir, "classifier.pkl"))
    sc_cls = joblib.load(os.path.join(model_dir, "scaler_cls.pkl"))
    models = {}
    for c in ["hypo", "normal", "hyper"]:
        path = os.path.join(model_dir, f"model_{c}")
        if os.path.exists(path):
            models[c] = (
                joblib.load(os.path.join(path, "scaler.pkl")),
                joblib.load(os.path.join(path, "sbp_model.pkl")),
                joblib.load(os.path.join(path, "dbp_model.pkl")),
            )
    return sc_cls, cls, models

def predict_file(fpath, sc, clf, model_dict):
    with open(fpath) as f:
        d = json.loads("".join([l for l in f if not l.strip().startswith("//")]))
    ppg = medfilt(np.array(d["Pleth"]), kernel_size=3)
    actual_sbp = d.get("SBP") or d.get("BPSystolic")
    actual_dbp = d.get("DBP") or d.get("BPDiastolic")
    # Only compute for valid data
    if (
        actual_sbp in INVALID_BP or actual_dbp in INVALID_BP or
        actual_sbp is None or actual_dbp is None
    ):
        return None, actual_sbp, actual_dbp, "Invalid BP value"
    if len(ppg) < SEG_LEN * SEGMENTS:
        return None, actual_sbp, actual_dbp, "Too short signal"

    preds = []
    for i in range(SEGMENTS):
        seg = ppg[i*SEG_LEN:(i+1)*SEG_LEN]
        feat = extract_features(seg)
        if all(f == 0 for f in feat): continue
        label = clf.predict(sc.transform([feat]))[0]
        sc2, sbp_m, dbp_m = model_dict[label]
        feat_scaled = sc2.transform([feat])
        sbp = float(sbp_m.predict(feat_scaled)[0])
        dbp = float(dbp_m.predict(feat_scaled)[0])
        preds.append((sbp, dbp))

    if len(preds) < 3:
        return None, actual_sbp, actual_dbp, "Too few valid segments"
    
    sbps = np.array([p[0] for p in preds])
    dbps = np.array([p[1] for p in preds])
    sbp_med, dbp_med = np.median(sbps), np.median(dbps)
    filtered = [(s, d) for s, d in preds if abs(s - sbp_med) <= MAX_VAR and abs(d - dbp_med) <= MAX_VAR]
    
    if len(filtered) < 3:
        return None, actual_sbp, actual_dbp, "MAD deviation too high"

    sbp_final = np.mean([f[0] for f in filtered])
    dbp_final = np.mean([f[1] for f in filtered])
    sbp_err = abs(sbp_final - float(actual_sbp)) if actual_sbp not in ("", None) else ""
    dbp_err = abs(dbp_final - float(actual_dbp)) if actual_dbp not in ("", None) else ""

    reason = "Passed all checks"
    if sbp_err != "" and sbp_err > 10:
        reason = "High error (SBP > 10)"
    if dbp_err != "" and dbp_err > 10:
        reason = "High error (DBP > 10)"

    return (sbp_final, dbp_final), actual_sbp, actual_dbp, reason

def main(input_dir, model_dir, out_csv):
    sc, clf, model_dict = load_models(model_dir)
    files = sorted(glob(os.path.join(input_dir, "**", "*.json"), recursive=True))

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename",
            "Pred_SBP", "Pred_DBP",
            "Actual_SBP", "Actual_DBP",
            "SBP_Error", "DBP_Error",
            "Reason"
        ])

        for fpath in files:
            out, actual_sbp, actual_dbp, reason = predict_file(fpath, sc, clf, model_dict)
            fname = os.path.basename(fpath)
            # Skip display and writing for invalid BP values
            if (
                actual_sbp in INVALID_BP or actual_dbp in INVALID_BP or
                actual_sbp is None or actual_dbp is None
            ):
                continue
            if out:
                pred_sbp, pred_dbp = out
                sbp_val = float(actual_sbp) if actual_sbp not in ("", None) else ""
                dbp_val = float(actual_dbp) if actual_dbp not in ("", None) else ""
                sbp_err = abs(pred_sbp - sbp_val) if sbp_val != "" else ""
                dbp_err = abs(pred_dbp - dbp_val) if dbp_val != "" else ""
                print(f"✅ {fname} | Pred: {pred_sbp:.1f}/{pred_dbp:.1f}, Actual: {sbp_val}/{dbp_val}, Err: {sbp_err}/{dbp_err} | {reason}")
                writer.writerow([
                    fname,
                    round(pred_sbp,1), round(pred_dbp,1),
                    actual_sbp, actual_dbp,
                    sbp_err, dbp_err,
                    reason
                ])
            else:
                print(f"❌ Skipped {fname} | Reason: {reason}")
                writer.writerow([
                    fname, "", "",
                    actual_sbp, actual_dbp,
                    "", "",
                    reason
                ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict BP and log errors with reasons")
    parser.add_argument("input_dir", help="Input folder with JSON files")
    parser.add_argument("model_dir", help="Folder with trained models")
    parser.add_argument("--out_csv", default="bp_predictions_with_reasons.csv", help="Output CSV path")
    args = parser.parse_args()
    main(args.input_dir, args.model_dir, args.out_csv)
