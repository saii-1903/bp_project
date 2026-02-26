import json
import sys

file_path = r'data/training_data/AAshif_2025-08-08_09-00-58.json'
try:
    with open(file_path, 'r') as f:
        d = json.load(f)
    print(f"Keys: {list(d.keys())}")
    print(f"SBP: {d.get('SBP')}")
    print(f"DBP: {d.get('DBP')}")
    print(f"Pleth length: {len(d.get('Pleth', []))}")
except Exception as e:
    print(f"Error: {e}")
