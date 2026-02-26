import os
import json
import numpy as np
import h5py
import scipy.signal
from scipy.signal import find_peaks
from tqdm import tqdm

# ─── Configuration ───────────────────────────────────────────────────
INPUT_FOLDER = r"c:\Users\saish\OneDrive\Attachments\Documents\porject\bp project\data" 
OUTPUT_FOLDER = r"c:\Users\saish\OneDrive\Attachments\Documents\porject\bp project\data\uci_training_data_json"

# Sampling Rates
UCI_FS = 125        # The UCI dataset is sampled at 125 Hz
TARGET_FS = 120     # Your model (danger_v2.py) expects 120 Hz

# Segmentation
WINDOW_SEC = 30     # 30-second windows (matches your pipeline)
MIN_BP_SYS = 60     # Filter out unrealistic BP (dead/noise)
MAX_BP_SYS = 200
MIN_BP_DIA = 40
MAX_BP_DIA = 130

# ─────────────────────────────────────────────────────────────────────

def get_bp_labels(abp_segment):
    """
    Extracts SBP and DBP from the Arterial Blood Pressure (ABP) waveform.
    Instead of just max/min (which is noisy), we find peaks/valleys.
    """
    # Find systolic peaks
    peaks, _ = find_peaks(abp_segment, distance=UCI_FS*0.4) # ~0.4s min distance
    # Find diastolic valleys (inverse peaks)
    valleys, _ = find_peaks(-abp_segment, distance=UCI_FS*0.4)
    
    if len(peaks) < 5 or len(valleys) < 5:
        return None, None # Signal too short or flat

    sbp = np.median(abp_segment[peaks])
    dbp = np.median(abp_segment[valleys])
    
    return sbp, dbp

def process_mat_file(mat_path, output_dir):
    filename = os.path.basename(mat_path)
    var_name = filename.replace('.mat', '')
    print(f"\nProcessing {filename} (Variable: {var_name})...")
    
    try:
        with h5py.File(mat_path, 'r') as f:
            if var_name not in f:
                # Fallback to any visible variable that isn't metadata
                possible = [k for k in f.keys() if not k.startswith('#')]
                if not possible:
                    print(f"Skipping {filename}: No data variable found.")
                    return
                print(f"Warning: {var_name} not found. using {possible[0]} instead.")
                var_name = possible[0]

            # Access the cell array of references
            dataset = f[var_name]
            # Handle both (1, N) and (N, 1) cell array shapes
            if dataset.ndim == 2:
                if dataset.shape[0] == 1:
                    refs = dataset[0]
                else:
                    refs = dataset[:, 0]
            else:
                refs = dataset[:]
            
            print(f"Found {len(refs)} patient records. Converting...")
            
            for i, ref in enumerate(tqdm(refs)):
                try:
                    # Dereference the object reference to get the actual matrix
                    record = f[ref]
                    data = np.array(record)
                    
                    # Transpose if necessary to get (Channels, Samples)
                    # UCI format is typically (Samples, 3) in HDF5/v7.3
                    if data.shape[1] == 3 and data.shape[0] > 100:
                        data = data.T
                        
                    if data.shape[0] != 3:
                        continue 

                    ppg_raw = data[0] # Photoplethysmogram
                    abp_raw = data[1] # Arterial Blood Pressure
                    
                    if np.isnan(ppg_raw).any() or np.isnan(abp_raw).any():
                        continue

                    # ─── Segment into 30s windows ───────────────────
                    samples_per_window = WINDOW_SEC * UCI_FS
                    total_samples = len(ppg_raw)
                    
                    for start_idx in range(0, total_samples, samples_per_window):
                        end_idx = start_idx + samples_per_window
                        if end_idx > total_samples:
                            break
                            
                        ppg_seg = ppg_raw[start_idx:end_idx]
                        abp_seg = abp_raw[start_idx:end_idx]
                        
                        if np.std(ppg_seg) < 0.05 or np.std(abp_seg) < 1.0:
                            continue

                        # ─── Calculate Labels ────────
                        sbp, dbp = get_bp_labels(abp_seg)
                        if sbp is None: continue
                        
                        if not (MIN_BP_SYS <= sbp <= MAX_BP_SYS) or \
                           not (MIN_BP_DIA <= dbp <= MAX_BP_DIA) or \
                           (sbp <= dbp + 5):
                            continue

                        # ─── Resample (125Hz -> 120Hz) ──────────────
                        target_len = WINDOW_SEC * TARGET_FS
                        ppg_120 = scipy.signal.resample(ppg_seg, target_len)
                        
                        # ─── Save JSON ──────────────────────────────
                        json_data = {
                            "Name": f"UCI_{var_name}_{i}_{start_idx}",
                            "PatId": f"UCI_{var_name}_{i}",
                            "Date": "2015-01-01", 
                            "Pleth": ppg_120.tolist(),
                            "PlethCnt": len(ppg_120),
                            "SBP": round(sbp, 1),
                            "DBP": round(dbp, 1),
                            "PRAllData": [], 
                            "Spo2": 98,    
                            "Fs": TARGET_FS
                        }
                        
                        out_name = f"UCI_{var_name}_Rec{i}_Seg{start_idx}.json"
                        with open(os.path.join(output_dir, out_name), 'w') as jf:
                            json.dump(json_data, jf)

                except Exception:
                    continue

    except Exception as e:
        print(f"Error opening {mat_path}: {e}")

if __name__ == "__main__":
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        print(f"Created output folder: {OUTPUT_FOLDER}")

    mat_files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".mat") and not f.startswith("._")]
    
    if not mat_files:
        print(f"No .mat files found in {INPUT_FOLDER}!")
    else:
        print(f"Found files: {mat_files}")
        for mat in mat_files:
            process_mat_file(os.path.join(INPUT_FOLDER, mat), OUTPUT_FOLDER)
            
    print("\nConversion Complete! You can now run danger_v2.py on the 'uci_training_data_json' folder.")
