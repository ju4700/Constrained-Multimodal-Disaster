import json

with open(r'd:\Development\iiucdatathon\targeted_fix_v3.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

extra_lines = [
    '\n# ── Advanced Hard Rules ──\n',
    'test.loc[(test["category"] == "Tropical Storm") & (test["label"] == "Catastrophic"), "label"] = "Severe"\n',
    'test.loc[(test["category"] == "Human Damage") & (test["label"] == "Catastrophic"), "label"] = "Severe"\n',
    'test.loc[(test["category"] == "Flood") & (test["label"] == "Minimal"), "label"] = "Mild"\n'
]

def inject_rules(source_list):
    new_source = []
    for line in source_list:
        new_source.append(line)
        if 'test.loc[test["category"] == "Non Disaster", "label"] = "Minimal"' in line:
            # Match indentation
            indent = line[:len(line) - len(line.lstrip())]
            for ext_line in extra_lines:
                new_source.append(indent + ext_line)
    return new_source

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['source'] = inject_rules(cell['source'])

with open(r'd:\Development\iiucdatathon\targeted_fix_v3.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Successfully injected advanced hard rules into targeted_fix_v3.ipynb')
