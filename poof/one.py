# ==========================================
# STEP 1: SETUP & REPRODUCIBILITY (Rule Compliance)
# ==========================================
import os
import random
import warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments, 
    set_seed
)
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import zipfile

warnings.filterwarnings("ignore")

# 🔒 LOCK SEEDS FOR REPRODUCIBILITY (30% Penalty Avoidance)
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)

seed_everything(42)

# ==========================================
# STEP 2: CONFIGURATION
# ==========================================
class CFG:
    model_name = "csebuetnlp/banglabert"
    base_path = "/kaggle/input/competitions/datathon-iiuc-cse-fest-2026/DisasterSeverity/"
    max_len = 256            # Increased for better context capture
    epochs = 6               # Increased to allow more training, relying on early stopping
    batch_size = 16          # Fits perfectly on Kaggle T4
    lr = 2e-5
    n_folds = 5              # 5-Fold Cross Validation
    seed = 42
    
label_map = {"Minimal": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Catastrophic": 4}
reverse_label_map = {v: k for k, v in label_map.items()}

# ==========================================
# STEP 3: DATA PREPARATION
# ==========================================
print("Loading Data...")
train = pd.read_csv(f"{CFG.base_path}train.csv")
test = pd.read_csv(f"{CFG.base_path}test.csv")

# We can merge train and val together for Cross Validation to get more training data!
val = pd.read_csv(f"{CFG.base_path}validation.csv")
train = pd.concat([train, val]).reset_index(drop=True)

# Map labels and fill NAs
train['label_id'] = train['label'].map(label_map)
train['context'] = train['context'].fillna("")
test['context'] = test['context'].fillna("")

# Create Stratified Folds (Ensures Catastrophic class is spread evenly)
skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
train['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(train, train['label_id'])):
    train.loc[val_idx, 'fold'] = fold

tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

def tokenize_fn(examples):
    return tokenizer(examples["context"], padding="max_length", truncation=True, max_length=CFG.max_len)

test_dataset = Dataset.from_pandas(test[['context']])
tokenized_test = test_dataset.map(tokenize_fn, batched=True)

# ==========================================
# STEP 4: CUSTOM METRICS & LOSS FOR IMBALANCE
# ==========================================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # The judges' exact evaluation metric
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1}


# ==========================================
# STEP 5: THE 5-FOLD TRAINING LOOP
# ==========================================
test_predictions = [] # To store predictions from all 5 models

for fold in range(CFG.n_folds):
    print(f"\n{'='*20} FOLD {fold+1}/{CFG.n_folds} {'='*20}")
    
    # Split Data
    trn_df = train[train['fold'] != fold].reset_index(drop=True)
    val_df = train[train['fold'] == fold].reset_index(drop=True)
    
    # Create HuggingFace Datasets
    trn_ds = Dataset.from_pandas(trn_df[['context', 'label_id']].rename(columns={'label_id': 'label'}))
    val_ds = Dataset.from_pandas(val_df[['context', 'label_id']].rename(columns={'label_id': 'label'}))
    
    tok_trn = trn_ds.map(tokenize_fn, batched=True)
    tok_val = val_ds.map(tokenize_fn, batched=True)
    
    # Initialize fresh model for each fold
    model = AutoModelForSequenceClassification.from_pretrained(CFG.model_name, num_labels=5)
    
    training_args = TrainingArguments(
        output_dir=f"/kaggle/working/fold_{fold}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=CFG.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        per_device_train_batch_size=CFG.batch_size,
        per_device_eval_batch_size=CFG.batch_size,
        num_train_epochs=CFG.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",      # Optimize for Weighted F1
        greater_is_better=True,
        report_to="none",
        save_total_limit=1
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tok_trn,
        eval_dataset=tok_val,
        compute_metrics=compute_metrics
    )
    
    # Train and predict on test set
    trainer.train()
    print(f"Predicting Test Set for Fold {fold+1}...")
    preds = trainer.predict(tokenized_test).predictions
    test_predictions.append(preds)

# ==========================================
# STEP 6: ENSEMBLING & POST-PROCESSING
# ==========================================
print("\nEnsembling 5-Fold Predictions...")
# Average the raw logits across all 5 models
final_logits = np.mean(test_predictions, axis=0)
final_preds = np.argmax(final_logits, axis=-1)

# Map numbers back to strings
test['label'] = [reverse_label_map[p] for p in final_preds]

print("Applying Rule-Based Category Hack...")
# 🚨 THE DATASET HACK: Override model predictions based on EDA logic 🚨
test.loc[test['category'] == 'Non Disaster', 'label'] = 'Minimal'

# ==========================================
# STEP 7: SUBMISSION FORMATTING
# ==========================================
submission = test[['image_id', 'label']]
submission.to_csv("submission.csv", index=False)

with zipfile.ZipFile('submission.zip', 'w') as zipf:
    zipf.write('submission.csv', arcname='submission.csv')

print("\n✅ Training Complete! 'submission.zip' is ready for upload.")
print(submission.head(10))