import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

with open(r'd:\Development\iiucdatathon\goodscores\unimodaltest.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    ct = cell['cell_type']
    print("Cell {} ({}):".format(i, ct))
    print("  " + src[:300].replace('\n', '\n  '))
    print()
