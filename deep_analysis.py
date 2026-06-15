import pandas as pd, numpy as np, os, sys
sys.stdout.reconfigure(encoding='utf-8')

BASE = r'd:\Development\iiucdatathon\datathon-iiuc-cse-fest-2026\DisasterSeverity'
train = pd.read_csv(os.path.join(BASE, 'train.csv'))
val   = pd.read_csv(os.path.join(BASE, 'validation.csv'))
test  = pd.read_csv(os.path.join(BASE, 'test.csv'))
combined = pd.concat([train, val]).reset_index(drop=True)

print('=== CATEGORY x LABEL (counts) ===')
labels_order = ['Minimal','Mild','Moderate','Severe','Catastrophic']
for cat in sorted(combined['category'].unique()):
    subset = combined[combined['category'] == cat]
    print(f'\n{cat} (n={len(subset)}):')
    for label in labels_order:
        c = len(subset[subset['label']==label])
        pct = c/len(subset)*100
        print(f'  {label:15s}: {c:4d} ({pct:5.1f}%)')

print('\n=== TEXT LENGTH STATS ===')
combined['text_len'] = combined['context'].fillna('').str.len()
tl = combined['text_len']
print(f'Mean: {tl.mean():.0f} chars | Median: {tl.median():.0f} | Max: {tl.max():.0f} | Min: {tl.min():.0f}')
print(f'Null contexts: {combined["context"].isna().sum()}')

print('\n=== TEST CATEGORY DISTRIBUTION ===')
for cat in sorted(test['category'].unique()):
    c = len(test[test['category']==cat])
    print(f'  {cat:15s}: {c:4d} ({c/len(test)*100:.1f}%)')

print('\n=== DUPLICATE TEXT CHECK ===')
dups = combined['context'].duplicated().sum()
print(f'Duplicate context strings: {dups}/{len(combined)}')

print('\n=== IMAGE FILE CHECK ===')
train_imgs = os.listdir(os.path.join(BASE, 'Train'))
test_imgs = os.listdir(os.path.join(BASE, 'Test'))
val_imgs = os.listdir(os.path.join(BASE, 'Validation'))
print(f'Train images: {len(train_imgs)} | Val images: {len(val_imgs)} | Test images: {len(test_imgs)}')

# Check if all CSVs reference existing images
train_missing = sum(1 for img in combined['image_name'] if not os.path.exists(os.path.join(BASE, 'Train', img)) and not os.path.exists(os.path.join(BASE, 'Validation', img)))
test_missing = sum(1 for img in test['image_name'] if not os.path.exists(os.path.join(BASE, 'Test', img)))
print(f'Missing train images: {train_missing} | Missing test images: {test_missing}')

print('\n=== EXPECTED TEST LABEL DISTRIBUTION (based on train priors) ===')
for label in labels_order:
    train_pct = len(combined[combined['label']==label]) / len(combined)
    expected = int(train_pct * len(test))
    print(f'  {label:15s}: ~{expected:4d} ({train_pct*100:.1f}%)')

print('\n=== VS ACTUAL SUBMISSION (0.784 run) ===')
actual = {'Mild': 197, 'Moderate': 165, 'Severe': 162, 'Catastrophic': 161, 'Minimal': 105}
for label in labels_order:
    train_pct = len(combined[combined['label']==label]) / len(combined)
    expected = int(train_pct * len(test))
    a = actual.get(label, 0)
    ratio = a / max(expected, 1)
    flag = ' <-- OVER' if ratio > 1.5 else (' <-- UNDER' if ratio < 0.7 else '')
    print(f'  {label:15s}: predicted={a:4d} expected=~{expected:4d} ratio={ratio:.2f}x{flag}')

print('\n=== HARD RULES FROM DATA ===')
for cat in sorted(combined['category'].unique()):
    subset = combined[combined['category'] == cat]
    dominant = subset['label'].value_counts()
    top_label = dominant.index[0]
    top_pct = dominant.iloc[0] / len(subset) * 100
    if top_pct > 80:
        print(f'  RULE: {cat} -> {top_label} ({top_pct:.1f}% in train)')
    
    # Check for labels that NEVER appear in this category
    for label in labels_order:
        c = len(subset[subset['label']==label])
        if c == 0:
            print(f'  ZERO: {cat} NEVER has {label}')

print('\n=== LANGUAGE ANALYSIS ===')
# Check if texts are Bengali or English
sample_texts = combined['context'].dropna().head(20)
has_bengali = 0
has_english = 0
for t in sample_texts:
    if any(ord(c) >= 0x0980 and ord(c) <= 0x09FF for c in t):
        has_bengali += 1
    if any(c.isascii() and c.isalpha() for c in t):
        has_english += 1
print(f'Sample of 20: {has_bengali} contain Bengali chars, {has_english} contain ASCII chars')

# Check overall
all_texts = combined['context'].dropna()
bengali_count = sum(1 for t in all_texts if any(ord(c) >= 0x0980 and ord(c) <= 0x09FF for c in str(t)))
print(f'Overall: {bengali_count}/{len(all_texts)} texts contain Bengali characters')
