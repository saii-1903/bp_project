import joblib, os, json

base = r"C:\Users\saish\OneDrive\Attachments\Documents\porject\bp project\models"
out = {}

for fname in ["hypo_models.pkl", "normal_models.pkl", "hyper_models.pkl"]:
    path = os.path.join(base, fname)
    obj = joblib.load(path)
    out[fname] = list(obj.keys())
    for k, v in obj.items():
        nf = getattr(v, "n_features_in_", "N/A")
        out[f"{fname}.{k}"] = f"{type(v).__name__}, n_features={nf}"

print(json.dumps(out, indent=2))
