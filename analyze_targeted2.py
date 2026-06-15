import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'd:\Development\iiucdatathon\targeted_fix_v3.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'find_temperature' in src:
            print(f"=== CELL {i} (find_temperature) ===")
            print(src[:3000])
            print("---END---\n")
        if 'oof_logits' in src and 'train_fold' in src.lower():
            print(f"=== CELL {i} (Training with OOF) ===")
            print(src[:3000])
            print("---END---\n")
        if 'CLASS_WEIGHTS' in src or 'class_weight' in src:
            print(f"=== CELL {i} (Class Weights) ===")
            print(src[:2000])
            print("---END---\n")
