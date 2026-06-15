"""
Convert ultimate_notebook.py to a Kaggle-ready .ipynb notebook.
Each CELL comment becomes a separate notebook cell.
"""
import json, re

with open(r"d:\Development\iiucdatathon\ultimate_notebook.py", "r", encoding="utf-8") as f:
    source = f.read()

# Split on CELL markers
# Pattern: # ═══...═══\n# CELL N: Title\n# ═══...═══
parts = re.split(r'# ═+\n# CELL \d+: (.+?)\n# ═+', source)

cells = []

# First part is the docstring header — make it a markdown cell
header = parts[0].strip()
if header.startswith('"""') and header.endswith('"""'):
    header = header[3:-3].strip()
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": header.split("\n")
    })

# Remaining parts alternate: title, code
for i in range(1, len(parts), 2):
    title = parts[i].strip()
    code = parts[i + 1].strip() if i + 1 < len(parts) else ""

    # Add markdown title
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"# {title}"]
    })

    # Add code cell
    if code:
        cells.append({
            "cell_type": "code",
            "metadata": {"trusted": True},
            "source": code.split("\n"),
            "outputs": [],
            "execution_count": None
        })

nb = {
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.13"}
    },
    "nbformat": 4,
    "nbformat_minor": 4,
    "cells": cells
}

out_path = r"d:\Development\iiucdatathon\ultimate_notebook_v5.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Created {out_path}")
print(f"Total cells: {len(cells)} ({sum(1 for c in cells if c['cell_type'] == 'code')} code, {sum(1 for c in cells if c['cell_type'] == 'markdown')} markdown)")
