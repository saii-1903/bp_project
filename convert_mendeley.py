import os
import glob
import scipy.io
import pandas as pd
import numpy as np
import json
import scipy.signal

# ─── CONFIGURATION ───────────────────────────────────────────────────
# Point this to where you extracted the Mendeley dataset
# It should have subfolders 'RawData' and 'Labels'
DATA_ROOT = "data/PPG_Dataset" 
OUTPUT_DIR = "data/hyper_only"
TARGET_FS = 120  # Your model expects 120Hz

# The Mendeley dataset is sampled at 2175 Hz (Very high!)
SOURCE_FS = 2175 

os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_mat_file(filepath):
    """Safe mat loader."""
    try:
        mat = scipy.io.loadmat(filepath)
        # The key often matches the filename, or is just 'data'
        # We look for the largest array
        max_len = 0
        best_key = None
        for k in mat.keys():
            if k.startswith('__'): continue
            val = mat[k]
            if hasattr(val, 'shape') and np.prod(val.shape) > max_len:
                max_len = np.prod(val.shape)
                best_key = k
        return mat[best_key].flatten() if best_key else None
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def process_mendeley_data():
    print(f"📂 Scanning for Mendeley data in '{DATA_ROOT}'...")
    
    # 1. Map Labels (Subject ID -> Glucose Value)
    # The label files are named Label_01_0001.mat
    # We need to read them to find the Glucose/Hb values.
    
    label_files = sorted(glob.glob(os.path.join(DATA_ROOT, "Labels", "*.mat")))
    raw_files = sorted(glob.glob(os.path.join(DATA_ROOT, "RawData", "*.mat")))
    
    if not label_files:
        print("❌ No Label files found! Did you unzip the dataset?")
        print("   Expected folder structure: ./Labels/*.mat and ./RawData/*.mat")
        return

    print(f"found {len(raw_files)} signals and {len(label_files)} labels.")

    count = 0
    for l_path in label_files:
        # Match filenames: Label_XX_XXXX.mat matches signal_XX_XXXX.mat
        basename = os.path.basename(l_path) # Label_01_0001.mat
        sig_name = basename.replace("Label", "signal") # signal_01_0001.mat
        s_path = os.path.join(DATA_ROOT, "RawData", sig_name)
        
        if not os.path.exists(s_path):
            continue
            
        # Load Label Data
        # Format: [ID, Gender(1=M,0=F), Age, Glucose(mg/dL), Height, Weight]
        label_data = load_mat_file(l_path)
        if label_data is None or len(label_data) < 4: continue
        
        age = float(label_data[2])
        glucose = float(label_data[3])
        gender_code = label_data[1]
        gender = "Male" if gender_code == 1 else "Female"
        bmi = 24.0 # Default if height/weight missing, or calc if available:
        if len(label_data) >= 6 and label_data[4] > 0:
            h_m = label_data[4] / 100.0
            w_kg = label_data[5]
            bmi = w_kg / (h_m * h_m)

        # Load PPG Signal
        ppg_raw = load_mat_file(s_path)
        if ppg_raw is None: continue
        
        # ─── RESAMPLE (2175Hz -> 120Hz) ──────────────────────────────
        # This dataset is huge (2kHz), we must downsample or your PC will crash.
        duration = len(ppg_raw) / SOURCE_FS
        target_samples = int(duration * TARGET_FS)
        ppg_120 = scipy.signal.resample(ppg_raw, target_samples)
        
        # ─── SEGMENTATION ────────────────────────────────────────────
        # Cut into 30-second chunks to match your pipeline
        chunk_len = 30 * TARGET_FS
        
        for i in range(0, len(ppg_120), chunk_len):
            segment = ppg_120[i : i + chunk_len]
            if len(segment) < chunk_len: break
            
            # Save JSON
            json_data = {
                "Name": f"Mendeley_Sub{label_data[0]}",
                "PatId": str(int(label_data[0])),
                "Age": age,
                "Gender": gender,
                "BMI": round(bmi, 1),
                "Glucose": glucose, # The Ground Truth
                "Pleth": segment.tolist(),
                "Fs": TARGET_FS
            }
            
            # Use 'Glucose' as 'Hb' dummy? No, let's keep them separate.
            # Your hbglucose.py handles missing keys correctly.
            
            fname = f"Mendeley_{basename.replace('.mat','')}_seg{i}.json"
            with open(os.path.join(OUTPUT_DIR, fname), 'w') as f:
                json.dump(json_data, f)
            
            count += 1

    print(f"✅ Success! Created {count} JSON training files in '{OUTPUT_DIR}'")
    print("   Now run: python hbglucose.py training_data_glucose --model-dir water/models")

if __name__ == "__main__":
    process_mendeley_data()