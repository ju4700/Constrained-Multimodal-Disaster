import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'd:\Development\iiucdatathon\goodscores\unimodaltest.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Extract cells 6 and 7 (ensemble + pseudo-labeling) - the critical post-processing
for i in [5, 6, 7]:
    cell = nb['cells'][i]
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        print(f"========== CELL {i} ==========")
        print(src)
        print()
