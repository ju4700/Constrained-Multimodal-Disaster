import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'd:\Development\iiucdatathon\targeted_fix_v3.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'find_temperature' in src or 'calibrat' in src.lower():
            print(f"=== CELL {i} (Temperature/Calibration) ===")
            print(src[:2000])
            print("---END---\n")
        if 'ensemble' in src.lower() or 'blend' in src.lower():
            print(f"=== CELL {i} (Ensemble/Blend) ===")
            print(src[:3000])
            print("---END---\n")
