import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open(r'd:\Development\iiucdatathon\goodscores\unimodaltest.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Print full source of cells 6, 7, 8, 9
for i in [6, 7, 8, 9]:
    cell = nb['cells'][i]
    src = ''.join(cell['source'])
    ct = cell['cell_type']
    print("="*80)
    print("Cell {} ({}):".format(i, ct))
    print("="*80)
    print(src)
    print()
