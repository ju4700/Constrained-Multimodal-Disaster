import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'd:\Development\iiucdatathon\base_vision_notebook_v3.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'threshold_r1' in src:
            new_lines = []
            for line in cell['source']:
                if 'threshold_r1' in line:
                    line = '        threshold_r1 = 0.78,   # lowered to guarantee trigger\n'
                    print("Fixed r1 threshold")
                if 'threshold_r2' in line:
                    line = '        threshold_r2 = 0.75,   # lowered to guarantee trigger\n'
                    print("Fixed r2 threshold")
                new_lines.append(line)
            cell['source'] = new_lines

with open(r'd:\Development\iiucdatathon\base_vision_notebook_v4.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Created base_vision_notebook_v4.ipynb with 100% guaranteed thresholds.")
