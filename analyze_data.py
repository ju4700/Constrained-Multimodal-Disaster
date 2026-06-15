import pandas as pd
import numpy as np
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

train = pd.read_csv(r'd:\Development\iiucdatathon\datathon-iiuc-cse-fest-2026\DisasterSeverity\train.csv')
val = pd.read_csv(r'd:\Development\iiucdatathon\datathon-iiuc-cse-fest-2026\DisasterSeverity\validation.csv')
test = pd.read_csv(r'd:\Development\iiucdatathon\datathon-iiuc-cse-fest-2026\DisasterSeverity\test.csv')

print('='*80)
print('TRAIN.CSV')
print('='*80)
print(f'Shape: {train.shape}')
print(f'Columns: {list(train.columns)}')
print(f'Null counts:')
print(train.isnull().sum())
print(f'\nLabel distribution:')
print(train['label'].value_counts().sort_index())
print(f'\nCategory distribution:')
print(train['category'].value_counts())
print(f'\nContext text length stats (chars):')
print(train['context'].str.len().describe())
print(f'\nContext word count stats:')
print(train['context'].split().str.len().describe() if False else train['context'].apply(lambda x: len(str(x).split())).describe())

# Sample texts
print(f'\nSample texts (first 5):')
for i in range(5):
    row = train.iloc[i]
    ctx = str(row['context'])[:150]
    print(f"  [{i}] cat={row['category']}, label={row['label']}")
    print(f"       text: {ctx}...")

print('\n' + '='*80)
print('CROSS-TAB: Category x Label (Train)')
print('='*80)
ct = pd.crosstab(train['category'], train['label'])
print(ct.to_string())
print('\nNormalized (row %):')
ct_pct = pd.crosstab(train['category'], train['label'], normalize='index').round(3) * 100
print(ct_pct.to_string())

print('\n' + '='*80)
print('VALIDATION.CSV')
print('='*80)
print(f'Shape: {val.shape}')
print(f'Columns: {list(val.columns)}')
print(f'Null counts:')
print(val.isnull().sum())
print(f'\nLabel distribution:')
print(val['label'].value_counts().sort_index())
print(f'\nCategory distribution:')
print(val['category'].value_counts())

print('\n' + '='*80)
print('CROSS-TAB: Validation Category x Label')
print('='*80)
ct_v = pd.crosstab(val['category'], val['label'])
print(ct_v.to_string())

print('\n' + '='*80)
print('TEST.CSV')
print('='*80)
print(f'Shape: {test.shape}')
print(f'Columns: {list(test.columns)}')
print(f'Null counts:')
print(test.isnull().sum())
print(f'\nCategory distribution:')
print(test['category'].value_counts())

# Check if test has label column
if 'label' in test.columns:
    print(f'\nTest label distribution:')
    print(test['label'].value_counts())
else:
    print('\nTest has NO label column')

# Check image_name patterns
print('\n' + '='*80)
print('IMAGE_NAME ANALYSIS')
print('='*80)
print(f'Train sample image names: {train["image_name"].head(5).tolist()}')
print(f'Test sample image names: {test["image_name"].head(5).tolist()}')
print(f'Train unique images: {train["image_name"].nunique()} / {len(train)}')
print(f'Val unique images: {val["image_name"].nunique()} / {len(val)}')
print(f'Test unique images: {test["image_name"].nunique()} / {len(test)}')

# Check if any images overlap
overlap_tt = set(train['image_name'].values) & set(test['image_name'].values)
print(f'Train-Test image overlap: {len(overlap_tt)}')
overlap_tv = set(train['image_name'].values) & set(val['image_name'].values)
print(f'Train-Val image overlap: {len(overlap_tv)}')
overlap_vt = set(val['image_name'].values) & set(test['image_name'].values)
print(f'Val-Test image overlap: {len(overlap_vt)}')

# Check for duplicate texts
print('\n' + '='*80)
print('DUPLICATE ANALYSIS')
print('='*80)
print(f'Train duplicate contexts: {train["context"].duplicated().sum()}')
print(f'Val duplicate contexts: {val["context"].duplicated().sum()}')
# Cross-set duplicates
train_val_text_overlap = set(train['context'].values) & set(val['context'].values)
print(f'Train-Val text overlap: {len(train_val_text_overlap)}')

# Check image directories
import os
for split in ['Train', 'Validation', 'Test']:
    img_dir = rf'd:\Development\iiucdatathon\datathon-iiuc-cse-fest-2026\DisasterSeverity\{split}'
    if os.path.exists(img_dir):
        files = os.listdir(img_dir)
        print(f'\n{split} image dir: {len(files)} files')
        print(f'  Sample: {files[:5]}')
        # Check extensions
        exts = [os.path.splitext(f)[1].lower() for f in files]
        from collections import Counter
        print(f'  Extensions: {dict(Counter(exts))}')
    else:
        print(f'\n{split} image dir: DOES NOT EXIST')

# Deep category-label analysis
print('\n' + '='*80)
print('DETAILED CATEGORY-LABEL PATTERNS')
print('='*80)
combined = pd.concat([train, val]).reset_index(drop=True)
print(f'Combined (train+val) shape: {combined.shape}')
for cat in sorted(combined['category'].unique()):
    subset = combined[combined['category'] == cat]
    print(f'\n--- {cat} ({len(subset)} samples) ---')
    dist = subset['label'].value_counts()
    for lbl in ['Minimal', 'Mild', 'Moderate', 'Severe', 'Catastrophic']:
        cnt = dist.get(lbl, 0)
        pct = cnt / len(subset) * 100
        print(f'  {lbl}: {cnt} ({pct:.1f}%)')

# Text length by label
print('\n' + '='*80)
print('TEXT LENGTH BY LABEL')
print('='*80)
for lbl in ['Minimal', 'Mild', 'Moderate', 'Severe', 'Catastrophic']:
    subset = combined[combined['label'] == lbl]
    lens = subset['context'].apply(lambda x: len(str(x).split()))
    print(f'{lbl}: mean={lens.mean():.1f}, median={lens.median():.1f}, min={lens.min()}, max={lens.max()}')

# Non Disaster category specific analysis
print('\n' + '='*80)
print('NON DISASTER ANALYSIS')
print('='*80)
nd_train = combined[combined['category'] == 'Non Disaster']
if len(nd_train) > 0:
    print(f'Non Disaster samples: {len(nd_train)}')
    print(f'Labels: {nd_train["label"].value_counts().to_dict()}')
    nd_test = test[test['category'] == 'Non Disaster']
    print(f'Non Disaster in test: {len(nd_test)}')
