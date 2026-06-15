import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'd:\Development\iiucdatathon\targeted_fix_v3.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Find and modify the temperature calibration cell
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'find_temperature' in src and 'test_logits_cal' in src:
            print(f"Found temperature calibration cell: CELL {i}")
            # Replace: skip calibration, just copy raw logits
            new_src = '''# ── Cell 7: SKIP Temperature Calibration ──────────────────────────────────
# Temperature calibration was hitting T=4.0 (ceiling) and producing zero
# improvement. With sqrt class weights, the logits are already well-calibrated.
# Removing this prevents the logits from being flattened, which allows
# pseudo-labeling to find confident samples.

print("Skipping temperature calibration (was hurting pseudo-labeling).")
for key in results:
    # Use raw test logits directly (no division by T)
    results[key]["test_logits_cal"] = results[key]["test_logits"].copy()
    print(f"  {key}: using raw logits (no calibration)")
'''
            cell['source'] = [line + '\n' for line in new_src.split('\n')[:-1]] + [new_src.split('\n')[-1]]
            print("  → Replaced with skip-calibration code")
            break

# Also lower the pseudo threshold from 0.85 to 0.80 to help
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        if 'pseudo_threshold' in src and 'CFG' in src and '0.85' in src:
            new_lines = []
            for line in cell['source']:
                if 'pseudo_threshold' in line and '0.85' in line:
                    line = line.replace('0.85', '0.80')
                    print(f"  → Lowered pseudo_threshold from 0.85 to 0.80")
                new_lines.append(line)
            cell['source'] = new_lines
            break

with open(r'd:\Development\iiucdatathon\targeted_fix_v5.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("\nCreated targeted_fix_v5.ipynb (sqrt weights kept, temperature removed)")
