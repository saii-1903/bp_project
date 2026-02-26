"""
inference_engine.py — Unified model inference for BP, Hb, and Glucose.

Feature vectors are matched EXACTLY to the training scripts:
  BP  (31 features) → dangerr_v2.py  extract_features()
  Hb/Glu (27 features) → hbglucose.py  extract_cycle_features()

BP classifier is loaded as a bundle dict (dangerr_v2 format):
  {"model": clf, "label_to_int": {...}, "int_to_label": {...}}
"""

import os
import joblib
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt, resample
from scipy.fft import fft
from scipy.stats import iqr
from collections import Counter
import config as cfg


class GenericTrendTracker:
    """
    Accumulates successive readings and returns a session trend.
    """
    def __init__(self, window=5, threshold=1.0):
        self._history = []
        self._window = window
        self._threshold = threshold

    def update(self, value):
        if value is None or np.isnan(value):
            return
        self._history.append(value)

    def get_trend(self):
        n = len(self._history)
        if n < 2:
            return {"trend": "Stable →", "slope": 0.0, "readings": n}

        arr = np.array(self._history[-self._window:])
        x = np.arange(len(arr))
        slope = float(np.polyfit(x, arr, 1)[0])

        if slope > self._threshold:
            trend = "Rising ↑"
        elif slope < -self._threshold:
            trend = "Falling ↓"
        else:
            trend = "Stable →"

        return {
            "trend": trend,
            "slope": round(slope, 2),
            "latest": self._history[-1],
            "readings": n,
        }

class BPTrendTracker:
    """
    Accumulates successive BP predictions and returns a session trend.
    """
    def __init__(self, window=5):
        self._sbp_history = []
        self._dbp_history = []
        self._cat_history = []
        self._window = window

    def update(self, data_or_sbp, dbp=None, category=None):
        if isinstance(data_or_sbp, dict):
            sbp = data_or_sbp.get("sbp")
            dbp = data_or_sbp.get("dbp")
            category = data_or_sbp.get("bp_category")
        else:
            sbp = data_or_sbp

        if sbp is None or dbp is None or np.isnan(sbp) or np.isnan(dbp):
            return
        self._sbp_history.append(sbp)
        self._dbp_history.append(dbp)
        self._cat_history.append(category or "unknown")

    def get_trend(self):
        n = len(self._sbp_history)
        if n < 2:
            return {"trend": "Stable", "slope": 0.0, "readings": n}

        sbp_arr = np.array(self._sbp_history[-self._window:])
        x = np.arange(len(sbp_arr))
        sbp_slope = float(np.polyfit(x, sbp_arr, 1)[0])

        if sbp_slope > 1.0:
            trend = "Rising ↑"
        elif sbp_slope < -1.0:
            trend = "Falling ↓"
        else:
            trend = "Stable →"

        cats = [c for c in self._cat_history[-self._window:] if c]
        latest_cat = Counter(cats).most_common(1)[0][0] if cats else "Normal"

        return {
            "trend": trend,
            "slope": round(sbp_slope, 2),
            "latest_sbp": self._sbp_history[-1],
            "current_category": latest_cat,
            "readings": n,
        }


class VitalInferenceEngine:
    def __init__(self):
        self.fs = cfg.MODEL_SAMPLING_RATE_HZ  # Models expect 120Hz
        self.models = self._load_all_models()

    # ─── Model Loading ───────────────────────────────────────────────────

    def _load_all_models(self):
        models = {}
        try:
            # BP classifier + group-specific regression models
            # dangerr_v2 saves the classifier as a bundle dict:
            #   {"model": clf, "label_to_int": {...}, "int_to_label": {...}}
            if os.path.exists(cfg.BP_MODEL_CONFIG["classifier"]):
                _clf_bundle = joblib.load(cfg.BP_MODEL_CONFIG["classifier"])
                if isinstance(_clf_bundle, dict):
                    models["bp_classifier"]   = _clf_bundle["model"]
                    models["bp_int_to_label"] = _clf_bundle.get("int_to_label", {0:"hypo",1:"normal",2:"hyper"})
                    models["bp_label_to_int"] = _clf_bundle.get("label_to_int", {"hypo":0,"normal":1,"hyper":2})
                else:
                    # Legacy: raw classifier object (old dangerr.py models)
                    models["bp_classifier"]   = _clf_bundle
                    models["bp_int_to_label"] = {0:"hypo", 1:"normal", 2:"hyper"}
                    models["bp_label_to_int"] = {"hypo":0, "normal":1, "hyper":2}
                models["bp_global_scaler"] = joblib.load(cfg.BP_MODEL_CONFIG["global_scaler"])
                for group in ["hypo", "normal", "hyper"]:
                    path = cfg.BP_MODEL_CONFIG[group]
                    scl_path = cfg.BP_MODEL_CONFIG.get(f"scaler_{group}")
                    if os.path.exists(path) and scl_path and os.path.exists(scl_path):
                        g = joblib.load(path)
                        models[f"bp_{group}_sbp"] = g["sbp_model"]
                        models[f"bp_{group}_dbp"] = g["dbp_model"]
                        models[f"bp_{group}_sbp_meta"] = g.get("sbp_meta")
                        models[f"bp_{group}_dbp_meta"] = g.get("dbp_meta")
                        models[f"bp_scaler_{group}"] = joblib.load(scl_path)

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

    def _iqr_filter(self, values, factor=1.5):
        """Return a boolean mask of inliers using IQR method."""
        if len(values) < 4:
            return np.ones(len(values), dtype=bool)
        q1, q3 = np.percentile(values, 25), np.percentile(values, 75)
        iq = q3 - q1
        return (np.array(values) >= q1 - factor * iq) & (np.array(values) <= q3 + factor * iq)

    # ─── BP Feature Extraction (matches dangerr.py exactly) ─────────────

    def _bp_features(self, ppg_seg, pr_seg=None):
        """
        31-feature vector identical to dangerr_v2.py extract_features().
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
        ibi = np.diff(peaks) / self.fs if len(peaks) > 1 else np.array([0.0])
        hrv = float(np.std(ibi))

        # PR stats (Fix: use pr_seg if available)
        if pr_seg is not None and len(pr_seg) > 0:
            pr_mean = float(np.nanmean(pr_seg)) if not np.all(np.isnan(pr_seg)) else 0.0
            pr_std  = float(np.nanstd(pr_seg))  if not np.all(np.isnan(pr_seg)) else 2.0
            if pr_std < 0.1: pr_std = 2.0
        else:
            pr_mean = float(60.0 / np.mean(ibi)) if np.mean(ibi) > 0 else 70.0
            pr_std  = float(np.std(60.0 / ibi))  if len(ibi) > 1 else 2.0

        # 25 features + 6 new vascular/HRV features = 31 total (matches danger.py)
        # ── Fix 4: new vascular / HRV features ──────────────────────────
        peak_idx  = np.argmax(c)
        post_peak = c[peak_idx:]
        notch_mins, _ = find_peaks(-post_peak)
        ri  = float(post_peak[notch_mins[0]] / (np.max(c) + 1e-9)) if len(notch_mins) > 0 else 0.0
        aix = float((post_peak[notch_mins[0]] - np.max(c)) / (np.max(c) + 1e-9)) if len(notch_mins) > 0 else 0.0
        large_si = float(0.1 / (ttp + 1e-9))
        if len(ibi) > 1:
            rmssd = float(np.sqrt(np.mean(np.diff(ibi) ** 2)))
            pnn50 = float(np.sum(np.abs(np.diff(ibi)) > 0.05) / max(len(ibi) - 1, 1))
        else:
            rmssd, pnn50 = 0.0, 0.0
        above_half = np.where(c >= np.max(c) * 0.5)[0]
        pw50 = float(len(above_half) / self.fs) if len(above_half) > 0 else 0.0
        # ───────────────────────────────────────────────────────────────────

        return [
            float(np.max(c)),   # 0
            float(t[-1]),       # 1
            float(ttp),         # 2
            float(ratio),       # 3
            float(np.max(d1)),  # 4
            float(np.min(d1)),  # 5
            float(np.max(d2)),  # 6
            float(np.min(d2)),  # 7
            *apg_feats,         # 8-10
            float(auc),         # 11
            *f_top,             # 12-14
            *m_top,             # 15-17
            float(hrv),         # 18
            float(np.mean(norm)),  # 19
            float(np.std(norm)),   # 20
            float(np.max(norm)),   # 21
            float(np.min(norm)),   # 22
            pr_mean,            # 23
            pr_std,             # 24
            ri,                 # 25 Reflection Index
            aix,                # 26 Augmentation Index
            large_si,           # 27 Large Artery Stiffness Index
            rmssd,              # 28 RMSSD
            pnn50,              # 29 pNN50
            pw50,               # 30 Pulse Width at 50%
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

        # Normalise ppg_filt to 0-1 range for features 16-19.
        # Raw ppg_filt absolute values are sensor-dependent (ADCSample ~620k vs
        # PlethWave 1-100). Normalising makes these features scale-invariant and
        # consistent with the normalized signal used for all other features.
        _pf_norm = (ppg_filt - ppg_filt.min()) / (ppg_filt.max() - ppg_filt.min() + 1e-6)

        features = [
            float(np.max(cycle)), float(pw), float(ttp), float(ratio),
            float(np.max(d1)), float(np.min(d1)), float(np.max(d2)), float(np.min(d2)),
            float(auc),
            float(top_freqs[0]), float(top_mags[0]),
            float(top_freqs[1]), float(top_mags[1]),
            float(top_freqs[2]), float(top_mags[2]),
            hrv_std,
            # indices 16-19: normalised signal stats (scale-invariant across sensors)
            float(np.mean(_pf_norm)), float(np.std(_pf_norm)),
            float(np.max(_pf_norm)),  float(np.min(_pf_norm)),
        ]

        # Additional optical/shape features (indices 20-26) — same as hbglucose.py
        from scipy.stats import skew as _skew, kurtosis as _kurtosis
        ac_dc_ratio     = (np.max(ppg_filt) - np.min(ppg_filt)) / (np.mean(np.abs(ppg_filt)) + 1e-9)
        sig_norm_e      = ppg_filt - np.min(ppg_filt)
        sig_norm_e      = sig_norm_e / (np.sum(sig_norm_e) + 1e-9)
        entropy         = float(-np.sum(sig_norm_e * np.log(sig_norm_e + 1e-9)))
        sig_skew        = float(_skew(ppg_filt))
        perfusion_index = (np.max(ppg_filt) - np.min(ppg_filt)) / (np.mean(ppg_filt) + 1e-9)
        sqi             = float(np.max(cycle) / (np.std(ppg_filt) + 1e-6))
        cycle_skew      = float(_skew(cycle))
        cycle_kurt      = float(_kurtosis(cycle))
        features       += [ac_dc_ratio, entropy, sig_skew,       # 20-22
                           perfusion_index, sqi, cycle_skew, cycle_kurt]  # 23-26

        # Apply same log1p transforms as hbglucose.py: indices 0,8,16,17,18,19,20,23
        for idx in [0, 8, 16, 17, 18, 19, 20, 23]:
            features[idx] = float(np.log1p(abs(features[idx])))

        return features


    # ─── Fix 3: Personal Calibration ────────────────────────────────────

    def apply_personal_calibration(
        self,
        predicted_sbp: float,
        predicted_dbp: float,
        ref_sbp: float,
        ref_dbp: float,
        baseline_pred_sbp: float,
        baseline_pred_dbp: float,
    ) -> tuple:
        """
        Applies a one-time personal calibration offset.

        If the user has a single reference cuff reading and the model's
        baseline prediction for that same session, this shifts all future
        predictions by the systematic error measured at calibration time.
        This alone typically halves MAE (from ~12 to ~6 mmHg).

        Args:
            predicted_sbp/dbp     : current model output
            ref_sbp/dbp           : user's actual cuff reading
            baseline_pred_sbp/dbp : model output from the calibration session

        Returns:
            (calibrated_sbp, calibrated_dbp)
        """
        sbp_offset = ref_sbp - baseline_pred_sbp
        dbp_offset = ref_dbp - baseline_pred_dbp
        cal_sbp = round(predicted_sbp + sbp_offset, 1)
        cal_dbp = round(predicted_dbp + dbp_offset, 1)
        print(f"DEBUG Calibration: offsets SBP={sbp_offset:+.1f} DBP={dbp_offset:+.1f} "
              f"→ SBP={cal_sbp} DBP={cal_dbp}")
        return cal_sbp, cal_dbp

    # ─── Public API ──────────────────────────────────────────────────────

    def predict_vitals(
        self,
        ppg_segment,
        actual_rate_hz: float = None,
        age: float = None,
        gender: str = None,   # 'Male' / 'Female' / None
        bmi: float = None,
        pr_all_data: list = None,
        offsets: dict = None,  # Expect dict like {'sbp': 10, 'dbp': -5, 'hb': 0.5, 'glucose': 20}
    ):
        """
        Run all models on a PPG segment and return vitals.
        Pass age/gender/bmi/pr_all_data so feature vectors match training.
        """

        from scipy.signal import medfilt as _medfilt
        from collections import Counter

        ppg_segment = np.array(ppg_segment, dtype=float)

        # ── Resample to model rate (120 Hz) ──────────────────────────
        src_rate = actual_rate_hz if actual_rate_hz else cfg.SAMPLING_RATE_HZ
        n_src    = len(ppg_segment)
        duration = n_src / src_rate
        
        # We ALWAYS want to end up at MODEL_SAMPLING_RATE_HZ (120Hz)
        n_target = int(round(duration * cfg.MODEL_SAMPLING_RATE_HZ))
        
        if n_target < int(cfg.MODEL_SAMPLING_RATE_HZ * 2):
            print(f"DEBUG: segment too short ({duration:.1f}s) — skipping")
            return None
            
        if n_target != n_src:
            ppg_segment = resample(ppg_segment, n_target)
            # Update internal fs for this prediction if it differs from the global one
            # though self.fs is usually already 120.
            current_fs = cfg.MODEL_SAMPLING_RATE_HZ
        else:
            current_fs = src_rate

        results = {}

        # 1. Blood Pressure — segment-wise (dangerr.py approach) ─────────
        try:
            FS       = cfg.MODEL_SAMPLING_RATE_HZ            # 120
            SEG_LEN  = FS * 5                                 # 600 samples / 5 s
            SEGMENTS = 6                                      # need 6 segments → 30 s

            if len(ppg_segment) < FS * 30:
                print(f"DEBUG BP: need 30s ({FS*30} samples), have {len(ppg_segment)} — skipping")
            elif "bp_classifier" not in self.models:
                print("DEBUG BP: models not loaded")
            else:
                # Resolve PR data
                pr_arr = None
                if pr_all_data and len(pr_all_data) > 0:
                    pr_arr = np.array(pr_all_data, dtype=float)
                    pr_arr[(pr_arr == 0) | (pr_arr == 127) | (pr_arr == 255)] = np.nan
                
                if pr_arr is None or np.count_nonzero(~np.isnan(pr_arr)) < 6:
                    # Generic PR mean if device data is missing
                    pr_arr = np.full(6, 75.0)

                ppg_med = _medfilt(ppg_segment, kernel_size=3)
                seg_predictions = []

                for i in range(SEGMENTS):
                    seg = ppg_med[i * SEG_LEN : (i + 1) * SEG_LEN]
                    # Map 30s PR array (1Hz) to 5s segment → 5 values per segment
                    # pr_arr[i*5 : (i+1)*5] gives 5 per-second HR readings per 5s window,
                    # enabling a real pr_std instead of always clamping to 2.0.
                    pr_seg = pr_arr[i*5 : (i+1)*5] if len(pr_arr) >= 30 else pr_arr if len(pr_arr) >= 6 else np.array([75.0])
                    
                    feat = self._bp_features(seg, pr_seg)
                    if feat is None:
                        continue

                    X_raw = np.array(feat, dtype=float).reshape(1, -1)
                    
                    # 1. Classification (Global Scaler)
                    X_cls = self.models["bp_global_scaler"].transform(X_raw)
                    probs = self.models["bp_classifier"].predict_proba(X_cls)[0]
                    classes = self.models["bp_classifier"].classes_

                    # 2. Regression (Soft Vote with Per-Group Scalers + Meta Ridge)
                    # classes_ may be integers [0,1,2] (dangerr_v2) or strings (legacy).
                    # Always resolve to string group names for model dict key lookup.
                    int_to_label = self.models.get("bp_int_to_label", {0:"hypo",1:"normal",2:"hyper"})
                    w_sbp = w_dbp = 0.0
                    for idx, cls_raw in enumerate(classes):
                        p = probs[idx]
                        cls_name = int_to_label.get(cls_raw, str(cls_raw))  # 0→"hypo" etc.
                        scl_key = f"bp_scaler_{cls_name}"
                        sbp_key = f"bp_{cls_name}_sbp"
                        dbp_key = f"bp_{cls_name}_dbp"
                        sbp_meta_key = f"bp_{cls_name}_sbp_meta"
                        dbp_meta_key = f"bp_{cls_name}_dbp_meta"

                        if scl_key in self.models and sbp_key in self.models:
                            X_reg = self.models[scl_key].transform(X_raw)
                            s_pred = float(self.models[sbp_key].predict(X_reg)[0])
                            d_pred = float(self.models[dbp_key].predict(X_reg)[0])

                            # Meta Ridge Correction (Fix 3)
                            if sbp_meta_key in self.models and self.models[sbp_meta_key]:
                                s_pred = float(self.models[sbp_meta_key].predict([[s_pred]])[0])
                                d_pred = float(self.models[dbp_meta_key].predict([[d_pred]])[0])

                            w_sbp += p * s_pred
                            w_dbp += p * d_pred

                    _raw_label = classes[np.argmax(probs)]
                    label_top = int_to_label.get(_raw_label, str(_raw_label))
                    seg_predictions.append({
                        "sbp": w_sbp, "dbp": w_dbp, "label": label_top, 
                        "conf": float(np.max(probs))
                    })

                if len(seg_predictions) < 3:
                     print(f"DEBUG BP: only {len(seg_predictions)} valid segments — need >=3")
                else:
                    # IQR Outlier Filtering (Fix 4)
                    sbp_list = [r["sbp"] for r in seg_predictions]
                    dbp_list = [r["dbp"] for r in seg_predictions]
                    sbp_mask = self._iqr_filter(sbp_list)
                    dbp_mask = self._iqr_filter(dbp_list)
                    mask = sbp_mask & dbp_mask

                    if mask.sum() < 2:
                        mask = np.ones(len(seg_predictions), dtype=bool)

                    clean_sbp = [sbp_list[i] for i, m in enumerate(mask) if m]
                    clean_dbp = [dbp_list[i] for i, m in enumerate(mask) if m]
                    clean_labels = [seg_predictions[i]["label"] for i, m in enumerate(mask) if m]

                    # Final Clamp & Result (Fix 7)
                    f_sbp = float(np.clip(np.mean(clean_sbp), *cfg.BP_SBP_LIMITS))
                    f_dbp = float(np.clip(np.mean(clean_dbp), *cfg.BP_DBP_LIMITS))
                    category = Counter(clean_labels).most_common(1)[0][0]

                    if offsets:
                        f_sbp += offsets.get('sbp', 0.0)
                        f_dbp += offsets.get('dbp', 0.0)

                    results["sbp"] = round(f_sbp, 1)
                    results["dbp"] = round(f_dbp, 1)
                    results["bp_category"] = category
                    print(f"DEBUG BP: SBP={results['sbp']} DBP={results['dbp']} ({category}) (Offsets: {offsets})")


        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"DEBUG BP error: {e}")

        # 2. Hemoglobin ─────────────────────────────────────────────────────
        hg_raw = None
        try:
            hg_raw = self._hb_glu_features(ppg_segment)
            if hg_raw is not None and "hb_model" in self.models:
                # Append 6 demographic features to match hbglucose.py training:
                # [age, age², is_senior, gender_bin, age*gender, bmi]
                age_val    = float(age)    if (age    and 0 < float(age) < 120) else 35.0
                gender_bin = 1.0           if str(gender).lower() == "male"     else 0.0
                bmi_val    = float(bmi)    if (bmi    and 10 < float(bmi) < 80) else -1.0
                demo = [age_val, age_val**2, 1.0 if age_val > 60 else 0.0,
                        gender_bin, age_val * gender_bin, bmi_val]
                X_hb = self.models["hb_scaler"].transform(
                    np.array(hg_raw + demo, dtype=float).reshape(1, -1)
                )
                raw_hb = float(self.models["hb_model"].predict(X_hb)[0])
                if offsets:
                    raw_hb += offsets.get('hb', 0.0)
                results["hb"] = round(raw_hb, 2)
                print(f"DEBUG Hb: {results['hb']} g/dL (Offset: {offsets.get('hb',0) if offsets else 0})")
        except Exception as e:
            print(f"DEBUG Hb error: {e}")

        # 3. Glucose ──────────────────────────────────────────────────────────
        try:
            if hg_raw is not None and "glucose_model" in self.models:
                age_val    = float(age)    if (age    and 0 < float(age) < 120) else 35.0
                gender_bin = 1.0           if str(gender).lower() == "male"     else 0.0
                bmi_val    = float(bmi)    if (bmi    and 10 < float(bmi) < 80) else -1.0
                demo = [age_val, age_val**2, 1.0 if age_val > 60 else 0.0,
                        gender_bin, age_val * gender_bin, bmi_val]
                X_glu = self.models["glucose_scaler"].transform(
                    np.array(hg_raw + demo, dtype=float).reshape(1, -1)
                )
                glu = float(self.models["glucose_model"].predict(X_glu)[0])
                if offsets:
                    glu += offsets.get('glucose', 0.0)
                results["glucose"] = round(max(40.0, min(400.0, glu)), 1)
                print(f"DEBUG Glu: {results['glucose']} mg/dL (Offset: {offsets.get('glucose',0) if offsets else 0})")
        except Exception as e:
            print(f"DEBUG Glucose error: {e}")

        return results if results else None



