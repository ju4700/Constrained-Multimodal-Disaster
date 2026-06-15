import pandas as pd
import numpy as np
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

train = pd.read_csv(r'd:\Development\iiucdatathon\datathon-iiuc-cse-fest-2026\DisasterSeverity\train.csv')
val = pd.read_csv(r'd:\Development\iiucdatathon\datathon-iiuc-cse-fest-2026\DisasterSeverity\validation.csv')
test = pd.read_csv(r'd:\Development\iiucdatathon\datathon-iiuc-cse-fest-2026\DisasterSeverity\test.csv')

combined = pd.concat([train, val]).reset_index(drop=True)

# ===========================================================================
# CRITICAL: Train vs Validation distribution SHIFT analysis
# ===========================================================================
print('='*80)
print('CRITICAL: TRAIN vs VALIDATION DISTRIBUTION SHIFT')
print('='*80)
print('\nThis is the KEY reason your model might underperform on test set!')

for cat in sorted(train['category'].unique()):
    print(f'\n--- {cat} ---')
    t = train[train['category'] == cat]['label'].value_counts(normalize=True).round(3) * 100
    v = val[val['category'] == cat]['label'].value_counts(normalize=True).round(3) * 100
    labels = ['Minimal', 'Mild', 'Moderate', 'Severe', 'Catastrophic']
    print(f'  {"Label":<15} {"Train%":>8} {"Val%":>8} {"Shift":>8}')
    for lbl in labels:
        tp = t.get(lbl, 0)
        vp = v.get(lbl, 0)
        shift = vp - tp
        flag = " ⚠️" if abs(shift) > 10 else ""
        print(f'  {lbl:<15} {tp:>7.1f}% {vp:>7.1f}% {shift:>+7.1f}%{flag}')

# ===========================================================================
# TEXT ANALYSIS: What makes severity levels different?
# ===========================================================================
print('\n' + '='*80)
print('TEXT CONTENT ANALYSIS: What makes texts different across labels?')
print('='*80)

# Check if "Non Disaster" texts are trivially different
nd = combined[combined['category'] == 'Non Disaster']
print(f'\nNon Disaster sample texts:')
for i in range(min(5, len(nd))):
    print(f'  {nd.iloc[i]["context"][:120]}')

# Check severity keywords
print('\nSample CATASTROPHIC texts (non-ND):')
cat_texts = combined[(combined['label'] == 'Catastrophic')]
for i in range(min(5, len(cat_texts))):
    print(f'  [{cat_texts.iloc[i]["category"]}] {cat_texts.iloc[i]["context"][:120]}')

print('\nSample MILD texts (non-ND):')
mild_texts = combined[(combined['label'] == 'Mild') & (combined['category'] != 'Non Disaster')]
for i in range(min(5, len(mild_texts))):
    print(f'  [{mild_texts.iloc[i]["category"]}] {mild_texts.iloc[i]["context"][:120]}')

# ===========================================================================
# CLASS IMBALANCE ANALYSIS
# ===========================================================================
print('\n' + '='*80)
print('OVERALL CLASS IMBALANCE (combined train+val)')
print('='*80)
label_counts = combined['label'].value_counts()
total = len(combined)
for lbl in ['Minimal', 'Mild', 'Moderate', 'Severe', 'Catastrophic']:
    cnt = label_counts.get(lbl, 0)
    pct = cnt / total * 100
    bar = '#' * int(pct)
    print(f'  {lbl:<15} {cnt:>4} ({pct:>5.1f}%) {bar}')

# ===========================================================================
# NOTEBOOK PERFORMANCE ANALYSIS: What's the ceiling?
# ===========================================================================
print('\n' + '='*80)
print('MODEL PERFORMANCE ANALYSIS (from notebook)')
print('='*80)
print("""
From the notebook training logs:
  - Model: csebuetnlp/banglabert (ELECTRA-base, 110M params)
  - CV F1 scores per fold: ~0.55 avg (THIS IS VERY LOW!)
  - Kaggle score: 0.784

KEY OBSERVATION: The CV score (~0.55) is DRAMATICALLY lower than the Kaggle score (0.784).
This means:
  1. The "Non Disaster" → "Minimal" rule is worth ~12.6% of the test set (100/790 samples)
  2. The model is likely getting ~0.55 on the disaster categories in CV
  3. But the 100% accuracy on Non Disaster boosts the overall score

ESTIMATED BREAKDOWN:
  - Non Disaster: 100/790 = 12.7% → likely 100% accuracy (rule-based)
  - Disaster: 690/790 = 87.3% → model needs ~0.78 F1 here to reach 0.784 overall
  
To reach 0.85:
  - If Non Disaster stays at 100%, disaster F1 needs: ~0.83
  - Current disaster F1 is approximately: ~0.76-0.78
  - Gap to close: ~5-7% on disaster categories alone
""")

# ===========================================================================
# VALIDATION SET LABEL SHIFT (CRITICAL for strategy)
# ===========================================================================
print('='*80)
print('VALIDATION vs TRAIN: Per-Category Label Distribution Divergence')
print('='*80)
print("""
MASSIVE DISTRIBUTION SHIFTS DETECTED:

Drought:
  Train: Severe=32.9%, Moderate=27.1%, Mild=19.7%
  Val:   Mild=46.0%, Minimal=30.0%, Moderate=16.0%
  → Val has WAY MORE Mild+Minimal, WAY LESS Severe!

Earthquake:  
  Train: Moderate=47.1%, Severe=23.4%, Mild=18.0%
  Val:   Severe=41.0%, Moderate=31.0%, Catastrophic=27.0%
  → Val has WAY MORE Severe+Catastrophic, WAY LESS Moderate!
  
Flood:
  Train: Moderate=47.1%, Severe=23.7%, Mild=20.6%
  Val:   Severe=48.0%, Catastrophic=27.0%, Moderate=23.0%
  → Val has WAY MORE Severe+Catastrophic!

Human Damage:
  Train: Moderate=55.1%, Mild=25.1%, Severe=18.0%
  Val:   Moderate=37.0%, Severe=34.0%, Mild=22.0%
  → Val has WAY MORE Severe!

Landslides:
  Train: Severe=35.7%, Mild=23.7%, Moderate=22.0%
  Val:   Moderate=46.0%, Severe=36.0%, Minimal=9.0%
  → Val has WAY MORE Moderate!

Wildfire:
  Train: Severe=40.0%, Moderate=29.1%, Mild=18.0%
  Val:   Severe=61.1%, Catastrophic=20.0%, Moderate=18.9%
  → Val has WAY MORE Severe+Catastrophic!

THIS IS THE BIGGEST PROBLEM: Train and Val have VERY DIFFERENT label distributions
within the same categories. This means the test set likely ALSO has a shifted distribution.
Your model trained on train distribution will be biased toward train's priors!
""")

# Check image resolution / size stats
import os
from PIL import Image

print('='*80)
print('IMAGE SIZE ANALYSIS (sampling)')
print('='*80)
for split_name, split_dir in [('Train', 'Train'), ('Validation', 'Validation'), ('Test', 'Test')]:
    img_dir = rf'd:\Development\iiucdatathon\datathon-iiuc-cse-fest-2026\DisasterSeverity\{split_dir}'
    files = os.listdir(img_dir)[:50]  # Sample 50 images
    widths, heights, sizes = [], [], []
    for f in files:
        path = os.path.join(img_dir, f)
        try:
            img = Image.open(path)
            widths.append(img.size[0])
            heights.append(img.size[1])
            sizes.append(os.path.getsize(path))
        except:
            pass
    print(f'\n{split_name} (sampled {len(widths)} images):')
    print(f'  Width:  mean={np.mean(widths):.0f}, min={np.min(widths)}, max={np.max(widths)}')
    print(f'  Height: mean={np.mean(heights):.0f}, min={np.min(heights)}, max={np.max(heights)}')
    print(f'  Size:   mean={np.mean(sizes)/1024:.1f}KB, min={np.min(sizes)/1024:.1f}KB, max={np.max(sizes)/1024:.1f}KB')
    # Check aspect ratios
    aspects = [w/h for w,h in zip(widths, heights)]
    print(f'  Aspect: mean={np.mean(aspects):.2f}, min={np.min(aspects):.2f}, max={np.max(aspects):.2f}')
