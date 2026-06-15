import json, re

with open(r'd:\Development\iiucdatathon\targeted_fix_v4.py', 'r', encoding='utf-8') as f:
    source = f.read()

# Split on cell markers
cells_raw = re.split(r'# ── Cell \d+:', source)

nb = {
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12.13"}
    },
    "nbformat": 4,
    "nbformat_minor": 4,
    "cells": []
}

# Add markdown header
nb["cells"].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# targeted_fix_v4: Original 0.784 Model + Hard Rules Only\n",
        "\n",
        "**Strategy**: Use the EXACT original model that scored 0.784 on the public leaderboard.\n",
        "Add ONLY the hard rules from EDA analysis. No sqrt weights, no temperature calibration,\n",
        "no LLRD, no AWP — those over-regularized the model and killed pseudo-labeling in v3.\n",
        "\n",
        "**Expected score**: 0.79-0.81"
    ]
})

for i, chunk in enumerate(cells_raw):
    chunk = chunk.strip()
    if not chunk:
        continue
    
    # Re-add the cell marker comment for context
    if i > 0:
        chunk = f"# ── Cell {i}: " + chunk
    
    nb["cells"].append({
        "cell_type": "code",
        "metadata": {"trusted": True},
        "source": chunk.split('\n'),
        "outputs": [],
        "execution_count": None
    })

# Fix: source should be list of strings with newlines
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        lines = cell["source"]
        cell["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]

with open(r'd:\Development\iiucdatathon\targeted_fix_v4.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Created targeted_fix_v4.ipynb")
