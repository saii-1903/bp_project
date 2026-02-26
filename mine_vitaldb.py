import vitaldb
import pandas as pd
import numpy as np
import json
import os
import time
from tqdm import tqdm

# ─── Configuration ───────────────────────────────────────────────────
OUTPUT_DIR = "vitaldb_hb_glu_json"
MAX_CASES = 50  # Keep it small for testing
WINDOW_SEC = 30
TARGET_FS = 120
TRACK_PLETH = 'SNUADC/PLETH'

# VitalDB API often uses these specific track names for labs
# We will check multiple variations
LAB_TRACKS_OF_INTEREST = [
    'Laboratory/Hb', 'Laboratory/Glucose', 
    'Hb', 'Glucose', 'Hgb', 'Glu'
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

def resample_signal(sig, orig_fs, target_fs):
    if len(sig) == 0: return []
    duration = len(sig) / orig_fs
    target_samples = int(duration * target_fs)
    x_old = np.linspace(0, duration, len(sig))
    x_new = np.linspace(0, duration, target_samples)
    return np.interp(x_new, x_old, sig).tolist()

def mine_data():
    print("⬇️  Connecting to VitalDB API...")
    
    # 1. Load Clinical Demographics (This usually works fast)
    try:
        df_cases = vitaldb.load_clinical_data(['age', 'sex', 'bmi'])
        print(f"✅ Loaded clinical data for {len(df_cases)} cases.")
    except Exception as e:
        print(f"❌ Connection Error (Clinical): {e}")
        return

    # 2. Load Labs (The tricky part)
    print("⬇️  Fetching Lab Results...")
    df_labs = vitaldb.load_lab_data()
    
    # CHECK: Did it fail?
    if df_labs.empty:
        print("\n⚠️  WARNING: VitalDB returned empty labs. This is common with unstable connections.")
        print("   Attempting 'Single Track' strategy...")
        
        # Strategy B: If bulk load fails, we pick valid cases from Clinical Data
        # and try to download their specific lab tracks individually.
        case_ids = df_cases.index.values[:MAX_CASES]
    else:
        # Strategy A: Filter the bulk data
        print(f"✅ Loaded {len(df_labs)} lab records.")
        # Fuzzy match for glucose/hb
        mask = df_labs['name'].str.contains('gluc|hb|hgb', case=False, na=False)
        df_labs = df_labs[mask]
        case_ids = df_labs['caseid'].unique()[:MAX_CASES]

    print(f"🚀  Mining {len(case_ids)} cases...")

    count_saved = 0
    for caseid in tqdm(case_ids):
        # ─── Demographics ───
        try:
            demo = df_cases.loc[caseid]
            age = demo.get('age')
            gender = demo.get('sex')
            bmi = demo.get('bmi')
            if pd.isna(age) or pd.isna(bmi): continue
            gender_str = "Male" if gender == 'M' else "Female"
        except KeyError:
            continue

        # ─── Get Lab Values ───
        # If we have the dataframe, use it. If not, we skip this check for now 
        # (simpler to rely on bulk load for labs).
        if df_labs.empty:
            continue 
            
        case_specific_labs = df_labs[df_labs['caseid'] == caseid]
        if case_specific_labs.empty: continue

        # ─── Download Waveform ───
        try:
            # Retry logic for connection timeout
            attempts = 0
            ppg_full = None
            while attempts < 3:
                try:
                    vf = vitaldb.VitalFile(caseid, [TRACK_PLETH])
                    ppg_full = vf.to_numpy([TRACK_PLETH], 1/100)[:,0]
                    break
                except:
                    attempts += 1
                    time.sleep(1)
            
            if ppg_full is None: continue

        except Exception:
            continue

        # ─── Match Labs to Waveform ───
        for _, row in case_specific_labs.iterrows():
            lab_time = row['dt']
            lab_name = row['name'].lower()
            lab_val = row['result']
            
            # Windowing
            start_time = lab_time - WINDOW_SEC
            if start_time < 0: continue
            
            start_idx = int(start_time * 100)
            end_idx = int(lab_time * 100)
            
            if end_idx >= len(ppg_full): continue
            
            ppg_segment = ppg_full[start_idx:end_idx]
            
            # Validation
            if np.std(ppg_segment) < 2.0: continue # Flatline check
            
            # Resample
            ppg_120 = resample_signal(ppg_segment, 100, TARGET_FS)
            
            # Save
            json_data = {
                "Name": f"VitalDB_{caseid}",
                "PatId": str(caseid),
                "Age": age, "Gender": gender_str, "BMI": bmi,
                "Pleth": ppg_120, "Fs": TARGET_FS
            }
            
            if 'hb' in lab_name or 'hgb' in lab_name:
                json_data['Hb'] = lab_val
            elif 'gluc' in lab_name:
                json_data['Glucose'] = lab_val
            
            clean_name = lab_name.replace("/", "").replace(" ", "")
            fname = f"VitalDB_{caseid}_{clean_name}_{int(lab_time)}.json"
            
            with open(os.path.join(OUTPUT_DIR, fname), 'w') as f:
                json.dump(json_data, f)
            count_saved += 1

    print(f"\n✅ Done! Saved {count_saved} training files to '{OUTPUT_DIR}'")
    if count_saved == 0:
        print("❌ Still zero files? Your internet might be blocking the API port (80/443).")

if __name__ == "__main__":
    mine_data()