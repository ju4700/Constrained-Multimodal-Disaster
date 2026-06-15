import pandas as pd
import os
import matplotlib.pyplot as plt

base_path = r'd:\Development\iiucdatathon\datathon-iiuc-cse-fest-2026\DisasterSeverity'
train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
val_df = pd.read_csv(os.path.join(base_path, 'validation.csv'))
test_df = pd.read_csv(os.path.join(base_path, 'test.csv'))

print("=== Shapes ===")
print(f"Train: {train_df.shape}")
print(f"Validation: {val_df.shape}")
print(f"Test: {test_df.shape}")
print("\n=== Columns ===")
print(f"Train: {list(train_df.columns)}")
print(f"Validation: {list(val_df.columns)}")
print(f"Test: {list(test_df.columns)}")

print("\n=== Missing Values ===")
print("Train:")
print(train_df.isnull().sum())
print("\nValidation:")
print(val_df.isnull().sum())
print("\nTest:")
print(test_df.isnull().sum())

print("\n=== Category Distribution ===")
print("Train:")
print(train_df['category'].value_counts())
print("\nValidation:")
print(val_df['category'].value_counts())

if 'label' in train_df.columns:
    print("\n=== Label Distribution ===")
    print("Train:")
    print(train_df['label'].value_counts())
    print("\nValidation:")
    print(val_df['label'].value_counts())

    print("\n=== Label by Category (Train) ===")
    print(pd.crosstab(train_df['category'], train_df['label']))

print("\n=== Text Analysis (Train) ===")
train_df['text_length'] = train_df['context'].apply(lambda x: len(str(x).split()))
print(train_df['text_length'].describe())

print("\n=== Check Image IDs ===")
print(f"Train Unique image_ids: {train_df['image_id'].nunique()} out of {len(train_df)}")
print(f"Validation Unique image_ids: {val_df['image_id'].nunique()} out of {len(val_df)}")
print(f"Test Unique image_ids: {test_df['image_id'].nunique()} out of {len(test_df)}")

