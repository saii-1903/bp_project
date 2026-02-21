import os
import json
import glob
import numpy as np
from scipy.signal import resample

# CONFIGURATION
INPUT_FOLDER = "hyper_only"   # Folder where your old JSONs are
OUTPUT_FOLDER = "hyperonly_training_data_200hz"   # Folder where you will put ALL data (Old + New)
OLD_FS = 120                      # Your old sampling rate
NEW_FS = 200                      # Your new target sampling rate (200 Hz)

def resample_file(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Get the Pleth waveform
        pleth = data.get("Pleth", [])
        if not pleth:
            print(f"Skipping {file_path}: No 'Pleth' data found.")
            return

        # Calculate new length
        # ratio = 200 / 120 = 1.666...
        num_samples = len(pleth)
        duration_seconds = num_samples / OLD_FS
        new_num_samples = int(duration_seconds * NEW_FS)

        # RESAMPLE
        # scipy.signal.resample uses Fourier method, good for periodic signals like PPG
        new_pleth = resample(pleth, new_num_samples)
        
        # Update the JSON data
        data["Pleth"] = new_pleth.tolist()
        data["FS"] = NEW_FS # Explicitly save the new rate
        
        # Note: We do NOT change PRAllData or SpO2 data 
        # because those are typically 1 value per second, which doesn't change 
        # when we change the internal sampling rate of the waveform.

        # Save to output folder
        filename = os.path.basename(file_path)
        output_path = os.path.join(OUTPUT_FOLDER, filename)
        
        with open(output_path, 'w') as f:
            json.dump(data, f)
            
        print(f"Converted {filename}: {num_samples} -> {new_num_samples} samples")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def main():
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
        
    # Find all JSON files
    files = glob.glob(os.path.join(INPUT_FOLDER, "*.json"))
    print(f"Found {len(files)} files in {INPUT_FOLDER}...")
    
    for file in files:
        resample_file(file)
        
    print(f"\n✅ Done! Use the '{OUTPUT_FOLDER}' folder for training.")

if __name__ == "__main__":
    main()
