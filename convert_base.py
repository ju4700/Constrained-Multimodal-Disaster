import json, re

with open(r'd:\Development\iiucdatathon\base_vision_notebook.py', 'r', encoding='utf-8') as f:
    src = f.read()

# Split by the big comment blocks
blocks = re.split(r'(# ═══════════════════════════════════════════════════════════════════════════════\n# CELL \d+.*?\n# ═══════════════════════════════════════════════════════════════════════════════\n)', src)

cells = []

# First block might be imports or docstring
if blocks[0].strip():
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + '\n' for line in blocks[0].strip().split('\n')]
    })

for i in range(1, len(blocks), 2):
    header = blocks[i]
    code = blocks[i+1]
    
    source_lines = [line + '\n' for line in header.split('\n')[:-1]]
    if code.strip():
        source_lines.extend([line + '\n' for line in code.split('\n')])
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines
    })

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

with open(r'd:\Development\iiucdatathon\base_vision_notebook_v2.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Created base_vision_notebook_v2.ipynb successfully.")
