import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'd:\Development\iiucdatathon\targeted_fix_v5.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'np.sqrt' in src and 'CLASS_WEIGHTS' in src:
            print(f"Found weights calculation in CELL {i}")
            new_lines = []
            for line in cell['source']:
                if 'CLASS_WEIGHTS = np.sqrt(raw_w).tolist()' in line:
                    new_line = line.replace('np.sqrt(raw_w).tolist()', 'raw_w.tolist()')
                    new_lines.append(new_line)
                    print("  - Replaced sqrt weights with standard weights")
                else:
                    new_lines.append(line)
            cell['source'] = new_lines
            break

with open(r'd:\Development\iiucdatathon\targeted_fix_v6.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\nCreated targeted_fix_v6.ipynb (Standard weights + No Temperature Bug)")
