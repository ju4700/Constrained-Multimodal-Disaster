import json, re

with open(r'd:\Development\iiucdatathon\base_vision_notebook.py', 'r', encoding='utf-8') as f:
    source = f.read()

parts = re.split(r'# \u2550+\n# CELL \d+: (.+?)\n# \u2550+', source)
cells = []

header = parts[0].strip()
cells.append({
    'cell_type': 'markdown',
    'metadata': {},
    'source': [header]
})

for i in range(1, len(parts), 2):
    title = parts[i].strip()
    code  = parts[i + 1].strip() if i + 1 < len(parts) else ''
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': [f'# {title}']})
    if code:
        cells.append({
            'cell_type': 'code',
            'metadata': {'trusted': True},
            'source': [line + '\n' for line in code.split('\n')],
            'outputs': [],
            'execution_count': None
        })

nb = {
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.12.13'}
    },
    'nbformat': 4,
    'nbformat_minor': 4,
    'cells': cells
}

out_path = r'd:\Development\iiucdatathon\base_vision_notebook.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'Created {out_path}')
