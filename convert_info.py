import json

with open(r"C:\Users\saish\OneDrive\Attachments\Documents\porject\bp project\model_info.json") as f:
    data = json.load(f)

lines = []
for item in data:
    lines.append("---")
    for k, v in item.items():
        lines.append(f"  {k}: {v}")

with open(r"C:\Users\saish\OneDrive\Attachments\Documents\porject\bp project\model_info_readable.py", "w") as f:
    f.write("INFO = '''\n" + "\n".join(lines) + "\n'''")
print("Done")
