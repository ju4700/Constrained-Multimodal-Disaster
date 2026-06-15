import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'd:\Development\iiucdatathon\goodscores\unimodaltest.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print("=== ORIGINAL 0.784 MODEL - FULL CELL LISTING ===\n")
for i, cell in enumerate(nb['cells']):
    ct = cell['cell_type']
    if ct == 'code':
        src = ''.join(cell['source'])
        lines = src.split('\n')
        print(f"CELL {i} [code] ({len(lines)} lines)")
        # Print first 5 lines
        for l in lines[:5]:
            print(f"  {l}")
        if len(lines) > 5:
            print(f"  ... ({len(lines)-5} more lines)")
    else:
        src = ''.join(cell['source'])
        print(f"CELL {i} [markdown]: {src[:80]}")
    print()
