import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'd:\Development\iiucdatathon\targeted_fix_v3.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=== All cells in targeted_fix_v3 ===")
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        print(f"\n--- CELL {i} (first 200 chars) ---")
        print(src[:200])
        print("...")
