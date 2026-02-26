import json
import os
import glob

def audit_data(directory):
    files = glob.glob(os.path.join(directory, "*.json"))
    stats = {
        "hyper": 0,
        "hypo": 0,
        "normal": 0,
        "other": 0
    }
    fs_counts = {}
    
    for f in files:
        try:
            with open(f, 'r') as jfile:
                data = json.load(jfile)
                
            # Try multiple possible keys for BP
            sbp = data.get("BPSystolic") or data.get("SBP", 0)
            dbp = data.get("BPDiastolic") or data.get("DBP", 0)
            pleth = data.get("Pleth", [])
            length = len(pleth)
            
            # Duration is assumed to be 30s based on project history
            # If duration is different, we can check bpc_duration or similar
            # But let's look for common lengths
            if length not in fs_counts:
                fs_counts[length] = 0
            fs_counts[length] += 1
            
            # Categorization
            if sbp < 90 or dbp < 60:
                stats["hypo"] += 1
            elif sbp >= 130 or dbp >= 80:
                stats["hyper"] += 1
            else:
                stats["normal"] += 1
                
        except json.JSONDecodeError:
            # Silent skip for corrupted JSON to clean up terminal
            stats["other"] += 1
        except Exception as e:
            # print(f"Error processing {f}: {e}")
            stats["other"] += 1
            
    return stats, fs_counts

if __name__ == "__main__":
    dir_path = r"c:\Users\saish\OneDrive\Attachments\Documents\porject\bp project\data\old_data_120hz"
    stats, fs_counts = audit_data(dir_path)
    
    with open("audit_summary.json", "w") as f:
        json.dump({"stats": stats, "fs_counts": fs_counts}, f, indent=4)
        
    print("Audit summary saved to audit_summary.json")
