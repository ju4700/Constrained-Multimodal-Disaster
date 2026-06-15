import json, sys
sys.stdout.reconfigure(encoding='utf-8')

# Original 0.784 model
with open(r'd:\Development\iiucdatathon\goodscores\unimodaltest.ipynb', 'r', encoding='utf-8') as f:
    nb1 = json.load(f)

print("=== ORIGINAL 0.784 MODEL ===")
for cell in nb1['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'text' in src and 'category' in src and ('context' in src or 'fillna' in src):
            if 'train[' in src or 'test[' in src:
                # Find the text construction line
                for line in src.split('\n'):
                    if 'text' in line and ('category' in line or 'context' in line):
                        print(f"  {line.strip()}")
        # Check class weights
        if 'CLASS_WEIGHTS' in src or 'class_weight' in src.lower():
            for line in src.split('\n'):
                if 'class_weight' in line.lower() or 'CLASS_WEIGHTS' in line:
                    print(f"  {line.strip()}")
        # Check pseudo labeling threshold
        if 'pseudo_threshold' in src or 'pseudo' in src.lower():
            for line in src.split('\n'):
                if 'threshold' in line.lower() and 'pseudo' in line.lower():
                    print(f"  {line.strip()}")

# targeted_fix_v3
with open(r'd:\Development\iiucdatathon\targeted_fix_v3.ipynb', 'r', encoding='utf-8') as f:
    nb2 = json.load(f)

print("\n=== TARGETED_FIX_V3 ===")
for cell in nb2['cells']:
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'text' in src and 'category' in src and ('context' in src or 'fillna' in src):
            if 'train[' in src or 'test[' in src:
                for line in src.split('\n'):
                    if 'text' in line and ('category' in line or 'context' in line):
                        print(f"  {line.strip()}")
        if 'CLASS_WEIGHTS' in src:
            for line in src.split('\n'):
                if 'CLASS_WEIGHTS' in line and '=' in line:
                    print(f"  {line.strip()}")
        if 'pseudo_threshold' in src:
            for line in src.split('\n'):
                if 'threshold' in line.lower() and 'pseudo' in line.lower():
                    print(f"  {line.strip()}")
