import os
import json
import glob
import argparse
import numpy as np

# ─── Configuration ───────────────────────────────────────────────────
# Invalid sentinel values to ignore (from BerryMed devices)
INVALID_BP = {0.0, -1.0, 1.0, 200.0, 202.0, 400.0, 404.0}

def get_bp_label(sbp, dbp):
    """
    Categorizes BP based on ACC/AHA 2017 Guidelines.
    Matches logic in danger_v2.py
    """
    if sbp < 90 or dbp < 60:
        return "hypo"
    elif sbp >= 130 or dbp >= 80:  # Stage 1 Hypertension starts at SBP 130 or DBP 80
        return "hyper"
    else:
        return "normal"

def count_labels(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.json"))
    print(f"Scanning {len(files)} files in: {folder_path}...\n")

    stats = {
        "hypo": 0,
        "normal": 0,
        "hyper": 0,
        "skipped": 0
    }
    
    valid_files = []

    for path in files:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                # Handle potential trailing commas or formatting issues
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    # Fallback for simple line-based reading if needed
                    f.seek(0)
                    content = "".join([line for line in f if not line.strip().startswith("//")])
                    data = json.loads(content)
                    
            # 1. Extract SBP/DBP
            sbp = data.get("SBP") or data.get("BPSystolic")
            dbp = data.get("DBP") or data.get("BPDiastolic")

            # 2. Validation Filters (Must match danger_v2.py exactly)
            if sbp is None or dbp is None:
                stats["skipped"] += 1
                continue
                
            # Ensure float/int
            try:
                sbp = float(sbp)
                dbp = float(dbp)
            except (ValueError, TypeError):
                stats["skipped"] += 1
                continue

            # Check for sentinel/invalid values
            if sbp in INVALID_BP or dbp in INVALID_BP:
                stats["skipped"] += 1
                continue
                
            # Check physiological range
            if not (60 <= sbp <= 220) or not (30 <= dbp <= 130):
                stats["skipped"] += 1
                continue
            
            # Sanity check
            if sbp <= dbp:
                stats["skipped"] += 1
                continue

            # 3. Categorize
            category = get_bp_label(sbp, dbp)
            stats[category] += 1
            valid_files.append(path)

        except Exception as e:
            # print(f"Error reading {os.path.basename(path)}: {e}")
            stats["skipped"] += 1

    # ─── Report ──────────────────────────────────────────────────────
    total_valid = stats["hypo"] + stats["normal"] + stats["hyper"]
    
    print("-" * 40)
    print(f"Total Files Scanned: {len(files)}")
    print(f"Valid Files Used:    {total_valid}")
    print(f"Skipped/Invalid:     {stats['skipped']}")
    print("-" * 40)
    print("DISTRIBUTION:")
    print(f"  🟢 Normal: {stats['normal']:>4}  ({(stats['normal']/total_valid*100) if total_valid else 0:.1f}%)")
    print(f"  🔴 Hyper:  {stats['hyper']:>4}  ({(stats['hyper']/total_valid*100) if total_valid else 0:.1f}%)")
    print(f"  🔵 Hypo:   {stats['hypo']:>4}  ({(stats['hypo']/total_valid*100) if total_valid else 0:.1f}%)")
    print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Count BP categories in a JSON folder.")
    parser.add_argument("folder", help="Path to the folder containing JSON files")
    args = parser.parse_args()
    
    if os.path.isdir(args.folder):
        count_labels(args.folder)
    else:
        print(f"Error: Directory not found -> {args.folder}")