import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'd:\Development\iiucdatathon\targeted_fix_v6.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        
        # Replace XLM-R with MuRIL
        if 'xlm-roberta-base' in src:
            print(f"Found XLM-R in CELL {i}")
            new_lines = []
            for line in cell['source']:
                if 'xlm-roberta-base' in line:
                    line = line.replace('xlm-roberta-base', 'google/muril-base-cased')
                    line = line.replace('xlmr_base', 'muril_base')
                    print("  → Swapped to google/muril-base-cased")
                new_lines.append(line)
            cell['source'] = new_lines
            
        # Update AWP Epsilon
        if 'awp_eps' in src and '1e-3' in src:
            print(f"Found awp_eps in CELL {i}")
            new_lines = []
            for line in cell['source']:
                if 'awp_eps' in line and '1e-3' in line:
                    line = line.replace('1e-3', '1e-2')
                    print("  → Updated awp_eps to 1e-2")
                new_lines.append(line)
            cell['source'] = new_lines

with open(r'd:\Development\iiucdatathon\targeted_fix_v7.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\nCreated targeted_fix_v7.ipynb (MuRIL + awp_eps 1e-2)")
