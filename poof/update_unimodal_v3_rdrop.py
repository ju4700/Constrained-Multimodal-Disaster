import json

file_path = "d:/Development/iiucdatathon/unimodal_v3.ipynb"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for cell in data.get("cells", []):
    if cell.get("cell_type") == "code":
        new_source = []
        for line in cell.get("source", []):
            if "use_rdrop   = True" in line:
                new_source.append(line.replace("use_rdrop   = True", "use_rdrop   = False"))
            elif "n_folds = 10" in line:
                new_source.append(line.replace("n_folds = 10", "n_folds = 5"))
            else:
                new_source.append(line)
        cell["source"] = new_source

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=1)

print("Successfully updated unimodal_v3.ipynb: disabled R-Drop and set 5 folds to prevent 12-hour Kaggle timeout.")
