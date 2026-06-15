import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'd:\Development\iiucdatathon\notebook6ba67268c2.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

all_output = []
for cell in nb['cells']:
    if cell['cell_type'] == 'code' and cell.get('outputs'):
        for out in cell['outputs']:
            text = ''
            if 'text' in out:
                text = ''.join(out['text'])
            elif out.get('output_type') == 'stream':
                text = ''.join(out.get('text', []))
            if text.strip():
                all_output.append(text)

full = '\n'.join(all_output)

# Find key sections
keywords = ['Calibration', 'Temperature', 'Ensemble', 'distribution', 'Catastrophic', 
            'submission', 'CV Summary', 'Mean CV', 'sqrt', 'class_weight', 'weights']
for kw in keywords:
    idx = full.find(kw)
    if idx >= 0:
        start = max(0, idx - 200)
        end = min(len(full), idx + 500)
        print(f"\n{'='*60}")
        print(f"FOUND '{kw}' at position {idx}")
        print(f"{'='*60}")
        print(full[start:end])
