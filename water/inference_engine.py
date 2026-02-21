"""
inference_engine.py — Unified model inference for BP, Hb, and Glucose.

Feature vectors are matched EXACTLY to the training scripts:
  BP  (25 features) → dangerr.py  extract_features()
  Hb/Glu (20 features) → hbglucose.py  extract_cycle_features()
"""

import os
import joblib
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt, resample
from scipy.fft import fft
import config as cfg


class VitalInferenceEngine:
    def __init__(self):
        self.fs = cfg.SAMPLING_RATE_HZ
        self.models = self._load_all_models()

    # ─── Model Loading ───────────────────────────────────────────────────

    def _load_all_models(self):
        models = {}
        try:
            # BP classifier + group-specific regression models
            if os.path.exists(cfg.BP_MODEL_CONFIG["classifier"]):
                models["bp_classifier"]    = joblib.load(cfg.BP_MODEL_CONFIG["classifier"])
                models["bp_global_scaler"] = joblib.load(cfg.BP_MODEL_CONFIG["global_scaler"])
                for group in ["hypo", "normal", "hyper"]:
                    path = cfg.BP_MODEL_CONFIG[group]
                    if os.path.exists(path):
                        g = joblib.load(path)
                        models[f"bp_{group}_sbp"] = g["sbp_model"]
                        models[f"bp_{group}_dbp"] = g["dbp_model"]

            # Hb / Glucose
            if os.path.exists(cfg.HB_GLU_MODEL_CONFIG["hb_model"]):
                models["hb_scaler"] = joblib.load(cfg.HB_GLU_MODEL_CONFIG["hb_scaler"])
                models["hb_model"]  = joblib.load(cfg.HB_GLU_MODEL_CONFIG["hb_model"])
            if os.path.exists(cfg.HB_GLU_MODEL_CONFIG["glucose_model"]):
                models["glucose_scaler"] = joblib.load(cfg.HB_GLU_MODEL_CONFIG["glucose_scaler"])
                models["glucose_model"]  = joblib.load(cfg.HB_GLU_MODEL_CONFIG["glucose_model"])

            print(f"✅ Loaded models from {cfg.MODEL_DIR}")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
        return models

    # ─── Signal Helpers ──────────────────────────────────────────────────

    def _bandpass(self, sig):
        nyq = 0.5 * self.fs
        b, a = butter(3, [0.4 / nyq, 11.0 / nyq], btype="band")
        return filtfilt(b, a, sig)

    def _is_noisy(self, sig):
        """Simple noise gate — rejects flat or clipped segments."""
        return np.std(sig) < 1e-4 or np.max(sig) - np.min(sig) < 0.01

    # ─── BP Feature Extraction (matches dangerr.py exactly) ─────────────

    def _bp_features(self, ppg_seg):
        """
        25-feature vector identical to dangerr.py extract_features().
        Returns None if signal is unusable.
        """
        if len(ppg_seg) < self.fs or self._is_noisy(ppg_seg):
            return None

        ppg_filt = self._bandpass(ppg_seg)
        norm = (ppg_filt - ppg_filt.min()) / (ppg_filt.max() - ppg_filt.min() + 1e-6)

        peaks, _ = find_peaks(norm, distance=int(self.fs * 0.4))
        mins,  _ = find_peaks(-norm, distance=int(self.fs * 0.3))

        if len(peaks) < 2 or len(mins) < 2:
            return None

        cycles = [norm[mins[mins < p][-1]: mins[mins > p][0]]
                  for p in peaks
                  if len(mins[mins < p]) > 0 and len(mins[mins > p]) > 0]
        if not cycles:
            return None

        c = max(cycles, key=np.max)
        t = np.linspace(0, len(c) / self.fs, len(c))

        d1 = np.gradient(c)
        d2 = np.gradient(d1)

        # APG waves
        a_w = np.max(d2);  b_w = np.min(d2)
        apg_feats = [a_w, b_w, b_w / (a_w + 1e-9)]

        auc   = np.trapezoid(c, t)
        ttp   = t[np.argmax(c)]
        tdp   = t[-1] - ttp if t[-1] - ttp != 0 else 1e-6
        ratio = ttp / tdp

        # FFT top-3 peaks
        fft_vals = np.abs(fft(c)[: len(c) // 2])
        freqs    = np.fft.fftfreq(len(c), 1 / self.fs)[: len(c) // 2]
        pks, _   = find_peaks(fft_vals, distance=5)
        top_idx  = np.argsort(fft_vals[pks])[-3:]
        f_top = list(freqs[pks][top_idx])  if len(top_idx) >= 3 else [0.0, 0.0, 0.0]
        m_top = list(fft_vals[pks][top_idx]) if len(top_idx) >= 3 else [0.0, 0.0, 0.0]

        # HRV + PR
        ibi    = np.diff(peaks) / self.fs if len(peaks) > 1 else np.array([0.0])
        hrv    = float(np.std(ibi))
        pr_mean = float(60.0 / np.mean(ibi)) if np.mean(ibi) > 0 else 70.0
        pr_std  = float(np.std(60.0 / ibi))  if len(ibi) > 1 else 2.0

        # 25 features — same order as dangerr.py
        return [
            float(np.max(c)),   # 0
            float(t[-1]),       # 1
            float(ttp),         # 2
            float(ratio),       # 3
            float(np.max(d1)),  # 4
            float(np.min(d1)),  # 5
            float(np.max(d2)),  # 6
            float(np.min(d2)),  # 7
            *apg_feats,         # 8-10  (a, b, b/a)
            float(auc),         # 11
            *f_top,             # 12-14
            *m_top,             # 15-17
            float(hrv),         # 18
            float(np.mean(ppg_seg)),  # 19
            float(np.std(ppg_seg)),   # 20
            float(np.max(ppg_seg)),   # 21
            float(np.min(ppg_seg)),   # 22
            pr_mean,            # 23
            pr_std,             # 24
        ]

    # ─── Hb/Glu Feature Extraction (matches hbglucose.py exactly) ───────

    def _hb_glu_features(self, ppg_seg):
        """
        20-feature vector identical to hbglucose.py extract_cycle_features(),
        including the same log1p transforms applied at training time.
        Returns None if signal is unusable.
        """
        if len(ppg_seg) < self.fs or self._is_noisy(ppg_seg):
            return None

        ppg_filt = self._bandpass(ppg_seg)
        norm = (ppg_filt - ppg_filt.min()) / (ppg_filt.max() - ppg_filt.min() + 1e-6)

        peaks, _ = find_peaks(norm, distance=int(self.fs * 0.4))
        mins,  _ = find_peaks(-norm, distance=int(self.fs * 0.3))

        cycles = [norm[mins[mins < p][-1]: mins[mins > p][0]]
                  for p in peaks
                  if len(mins[mins < p]) > 0 and len(mins[mins > p]) > 0]
        if not cycles:
            return None

        cycle = max(cycles, key=lambda x: np.max(x))
        t = np.linspace(0, len(cycle) / self.fs, len(cycle))

        d1 = np.gradient(cycle)
        d2 = np.gradient(d1)

        auc  = np.trapezoid(cycle, t)
        pw   = t[-1]
        ttp  = t[np.argmax(cycle)]
        tdp  = pw - ttp if pw - ttp != 0 else 1e-6
        ratio = ttp / tdp

        # FFT (interleaved freq, mag — same order as hbglucose.py)
        fft_vals = np.abs(fft(cycle)[: len(cycle) // 2])
        freqs    = np.fft.fftfreq(len(cycle), 1 / self.fs)[: len(cycle) // 2]
        pks, _   = find_peaks(fft_vals, distance=5)
        if len(pks) < 3:
            top_freqs = [0.0, 0.0, 0.0];  top_mags = [0.0, 0.0, 0.0]
        else:
            si = np.argsort(fft_vals[pks])[-3:]
            top_freqs = list(freqs[pks][si]);  top_mags = list(fft_vals[pks][si])

        ibi_list = np.diff(peaks) / self.fs if len(peaks) > 1 else [0.0]
        hrv_std  = float(np.std(ibi_list)) if len(ibi_list) > 1 else 0.0

        features = [
            float(np.max(cycle)), float(pw), float(ttp), float(ratio),
            float(np.max(d1)), float(np.min(d1)), float(np.max(d2)), float(np.min(d2)),
            float(auc),
            float(top_freqs[0]), float(top_mags[0]),
            float(top_freqs[1]), float(top_mags[1]),
            float(top_freqs[2]), float(top_mags[2]),
            hrv_std,
            float(np.mean(norm)), float(np.std(norm)),
            float(np.max(norm)),  float(np.min(norm)),
        ]

        # Apply same log1p transforms used at training time
        for idx in [0, 8, 16, 17, 18, 19]:
            features[idx] = float(np.log1p(abs(features[idx])))

        return features

    # ─── Public API ──────────────────────────────────────────────────────

    def predict_vitals(self, ppg_segment, actual_rate_hz: float = None):
        """
        Run all models on the given PPG segment.
        BP logic mirrors dangerr.predict_bp() exactly:
          - medfilt pre-processing
          - 6 × 5-second segments
          - soft-voting per segment
          - consistency check
          - PR-based SBP post-correction
        """
        from scipy.signal import medfilt as _medfilt
        from collections import Counter

        ppg_segment = np.array(ppg_segment, dtype=float)

        # ── Resample to canonical rate (200 Hz) ──────────────────────────
        src_rate = actual_rate_hz if actual_rate_hz else cfg.SAMPLING_RATE_HZ
        n_src    = len(ppg_segment)
        duration = n_src / src_rate
        n_target = int(round(duration * cfg.SAMPLING_RATE_HZ))
        if n_target < int(cfg.SAMPLING_RATE_HZ * 2):
            print(f"DEBUG: segment too short ({duration:.1f}s) — skipping")
            return None
        if n_target != n_src:
            ppg_segment = resample(ppg_segment, n_target)

        results = {}

        # 1. Blood Pressure — segment-wise (dangerr.py approach) ─────────
        try:
            FS       = cfg.SAMPLING_RATE_HZ                  # 200
            SEG_LEN  = FS * 5                                 # 1000 samples / 5 s
            SEGMENTS = 6                                      # need 6 segments → 30 s

            if len(ppg_segment) < FS * 30:
                print(f"DEBUG BP: need 30s ({FS*30} samples), have {len(ppg_segment)} — skipping")
            elif "bp_classifier" not in self.models:
                print("DEBUG BP: models not loaded")
            else:
                # Pre-process exactly like dangerr.py line 183
                ppg_med = _medfilt(ppg_segment, kernel_size=3)

                seg_predictions = []
                for i in range(SEGMENTS):
                    seg = ppg_med[i * SEG_LEN : (i + 1) * SEG_LEN]
                    feat = self._bp_features(seg)
                    if feat is None:
                        continue

                    X = self.models["bp_global_scaler"].transform(
                        np.array(feat, dtype=float).reshape(1, -1)
                    )
                    probs   = self.models["bp_classifier"].predict_proba(X)[0]
                    classes = self.models["bp_classifier"].classes_

                    # Soft vote: weighted sum across all groups
                    w_sbp = w_dbp = 0.0
                    for idx, cls_name in enumerate(classes):
                        p = probs[idx]
                        key_sbp = f"bp_{cls_name}_sbp"
                        key_dbp = f"bp_{cls_name}_dbp"
                        if key_sbp in self.models:
                            w_sbp += p * self.models[key_sbp].predict(X)[0]
                            w_dbp += p * self.models[key_dbp].predict(X)[0]

                    label = classes[np.argmax(probs)]
                    seg_predictions.append((w_sbp, w_dbp, label))

                if len(seg_predictions) < 3:
                    print(f"DEBUG BP: only {len(seg_predictions)} valid segments — need >=3")
                else:
                    sbp_vals = [p[0] for p in seg_predictions]
                    dbp_vals = [p[1] for p in seg_predictions]
                    sbp_med  = np.median(sbp_vals)
                    dbp_med  = np.median(dbp_vals)

                    bad_sbp = sum(abs(v - sbp_med) > 10 for v in sbp_vals)
                    bad_dbp = sum(abs(v - dbp_med) > 10 for v in dbp_vals)
                    if bad_sbp >= 3 or bad_dbp >= 3:
                        print(f"DEBUG BP: inconsistent segments (bad SBP={bad_sbp}, DBP={bad_dbp})")
                    else:
                        final_sbp = float(np.mean(sbp_vals))
                        final_dbp = float(np.mean(dbp_vals))
                        category  = Counter([p[2] for p in seg_predictions]).most_common(1)[0][0]

                        results["sbp"]         = round(final_sbp, 1)
                        results["dbp"]         = round(final_dbp, 1)
                        results["bp_category"] = category
                        print(f"DEBUG BP: SBP={results['sbp']} DBP={results['dbp']} ({category})")

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"DEBUG BP error: {e}")

        # 2. Hemoglobin — independent try/except ─────────────────────────
        hg_raw = None
        try:
            hg_raw = self._hb_glu_features(ppg_segment)
            if hg_raw is not None and "hb_model" in self.models:
                X_hb = self.models["hb_scaler"].transform(
                    np.array(hg_raw, dtype=float).reshape(1, -1)
                )
                results["hb"] = round(float(self.models["hb_model"].predict(X_hb)[0]), 2)
                print(f"DEBUG Hb: {results['hb']} g/dL")
        except Exception as e:
            print(f"DEBUG Hb error: {e}")

        # 3. Glucose — independent try/except ────────────────────────────
        try:
            if hg_raw is not None and "glucose_model" in self.models:
                X_glu = self.models["glucose_scaler"].transform(
                    np.array(hg_raw, dtype=float).reshape(1, -1)
                )
                glu = float(self.models["glucose_model"].predict(X_glu)[0])
                results["glucose"] = round(max(40.0, min(400.0, glu)), 1)
                print(f"DEBUG Glu: {results['glucose']} mg/dL")
        except Exception as e:
            print(f"DEBUG Glucose error: {e}")

        return results if results else None


