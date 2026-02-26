import h5py
import numpy as np
import os

MAT_PATH = r"c:\Users\saish\OneDrive\Attachments\Documents\porject\bp project\data\Part_2.mat"

def inspect_mat(path):
    print(f"Inspecting {path}...")
    try:
        with h5py.File(path, 'r') as f:
            print("Top level keys:", list(f.keys()))
            if 'Part_1' in f:
                p = f['Part_1']
                print(f"Shape of 'Part_1': {p.shape}")
                print(f"Type of 'Part_1': {p.dtype}")
                
                # Check the first reference
                # If it's a cell array, it might be (1, N) or (N, 1)
                if p.ndim == 2:
                    first_ref = p[0][0]
                else:
                    first_ref = p[0]
                
                print(f"First reference: {first_ref}")
                
                record = f[first_ref]
                data = np.array(record)
                print(f"Shape of data in first record: {data.shape}")
                
                # UCI usually has 3 rows: PPG, ABP, ECG
                if data.shape[0] == 3:
                    print("Data has 3 rows (PPG, ABP, ECG)")
                elif data.shape[1] == 3:
                    print("Data has 3 columns (PPG, ABP, ECG)")
                else:
                    print("Data shape does not immediately match 3 channels.")
            else:
                print("'p' not found in file.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if os.path.exists(MAT_PATH):
        inspect_mat(MAT_PATH)
    else:
        print(f"File not found: {MAT_PATH}")
