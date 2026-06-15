import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'd:\Development\iiucdatathon\base_vision_notebook_v2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        new_lines = []
        
        # Turn on AWP and fix epsilon
        if 'use_awp' in src and 'awp_eps' in src:
            print(f"Found MODEL_CFGS in CELL {i}")
            for line in cell['source']:
                if 'use_awp' in line and 'False' in line:
                    line = line.replace('False', 'True')
                    print("  → Turned AWP ON")
                if 'awp_eps' in line and '1e-3' in line:
                    line = line.replace('1e-3', '1e-2')
                    print("  → Updated awp_eps to 1e-2")
                new_lines.append(line)
            cell['source'] = new_lines
            
        # Lower pseudo-threshold to 0.80 to ensure it triggers despite AWP+sqrt
        elif 'PSEUDO_THRESH' in src:
            print(f"Found PSEUDO_THRESH in CELL {i}")
            for line in cell['source']:
                if 'PSEUDO_THRESH' in line and '0.85' in line:
                    line = line.replace('0.85', '0.80')
                    print("  → Lowered PSEUDO_THRESH to 0.80")
                new_lines.append(line)
            cell['source'] = new_lines

with open(r'd:\Development\iiucdatathon\base_vision_notebook_v3.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\nCreated base_vision_notebook_v3.ipynb (High CV F1 setup)")
