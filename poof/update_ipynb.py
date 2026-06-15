import json

file_path = "d:/Development/iiucdatathon/unimodal_v3.ipynb"

with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

for cell in data.get("cells", []):
    if cell.get("cell_type") == "code":
        new_source = []
        for line in cell.get("source", []):
            if "max_len = 128" in line:
                new_source.append(line.replace("max_len = 128", "max_len = 256"))
            elif "batch_size   = 8" in line:
                new_source.append(line.replace("batch_size   = 8", "batch_size   = 4"))
            elif "grad_accum   = 2" in line:
                new_source.append(line.replace("grad_accum   = 2", "grad_accum   = 4"))
            else:
                new_source.append(line)
        cell["source"] = new_source

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=1)

print("Successfully updated unimodal_v3.ipynb with max_len=256, batch_size=4, grad_accum=4")
