import os
import random
import warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModel,
    AutoImageProcessor,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from PIL import Image
import zipfile
import shutil
import gc
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

class CFG:
    text_model = "csebuetnlp/banglabert_large"
    vision_model = "google/vit-base-patch16-224-in21k"
    base_path = "/kaggle/input/competitions/datathon-iiuc-cse-fest-2026/DisasterSeverity/"
    max_len = 256            # Claude's truncation fix for Landslides
    epochs = 4
    batch_size = 8
    lr = 2e-5
    n_folds = 5
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(CFG.seed)

label_map = {"Minimal": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Catastrophic": 4}
reverse_label_map = {v: k for k, v in label_map.items()}

print("Loading Text & Vision Data...")
train = pd.read_csv(f"{CFG.base_path}train.csv")
val = pd.read_csv(f"{CFG.base_path}validation.csv")
test = pd.read_csv(f"{CFG.base_path}test.csv")

# FIX 1 & 2: Use image_name (with .jpg) and set paths BEFORE concat
train['image_path'] = train['image_name'].apply(lambda x: f"{CFG.base_path}Train/{x}")
test['image_path'] = test['image_name'].apply(lambda x: f"{CFG.base_path}Test/{x}")

val_folder = 'Validation'
for f in ['Validation', 'validation', 'Valid', 'valid', 'Train', 'train']:
    if os.path.exists(f"{CFG.base_path}{f}"):
        val_folder = f
        break
val['image_path'] = val['image_name'].apply(lambda x: f"{CFG.base_path}{val_folder}/{x}")

# 1. Claude's Bayesian Prior Calibration (Weighting Validation Heavily)
train_val_concat = pd.concat([train, val], ignore_index=True)
cross_all = pd.crosstab(train_val_concat['category'], train_val_concat['label'].map(label_map), normalize='index')
cross_val = pd.crosstab(val['category'], val['label'].map(label_map), normalize='index')
cross_val = pd.crosstab(val['category'], val['label'].map(label_map), normalize='index')

historical_probs_dict = {}
for cat in cross_all.index:
    p_all = cross_all.loc[cat].reindex(range(5), fill_value=0).values
    p_val = cross_val.loc[cat].reindex(range(5), fill_value=0).values if cat in cross_val.index else p_all
    # 50% Train+Val Prior, 50% Val Prior (chronologically closer to test set)
    historical_probs_dict[cat] = 0.5 * p_all + 0.5 * p_val

historical_probs_dict['Non Disaster'] = np.array([1.00, 0.00, 0.00, 0.00, 0.00])

train = pd.concat([train, val]).reset_index(drop=True)
train['label_id'] = train['label'].map(label_map)
train['context'] = train['context'].fillna("")
test['context'] = test['context'].fillna("")

# 2. Claude's Prepend Hack (Injecting explicit category & length tier into the text)
def build_text(row):
    length = len(str(row['context']))
    tier = "short" if length < 60 else ("medium" if length < 130 else "long")
    return f"[{row['category']}] [{tier}] {row['context']}"

train['text'] = train.apply(build_text, axis=1)
test['text'] = test.apply(build_text, axis=1)

# (Image paths already set before concat)

skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
train['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(train, train['label_id'])):
    train.loc[val_idx, 'fold'] = fold

# 3. Precompute class weights so we don't recalculate per batch
_counts = np.bincount(train['label_id'].values)
LOSS_WEIGHTS = torch.tensor(len(train) / (len(_counts) * _counts), dtype=torch.float32).to(CFG.device)

# --- Dual-Brain Dataset ---
tokenizer = AutoTokenizer.from_pretrained(CFG.text_model)
image_processor = AutoImageProcessor.from_pretrained(CFG.vision_model)

class MultimodalDataset(Dataset):
    def __init__(self, df, is_test=False):
        self.df = df
        self.is_test = is_test
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Text Brain Input
        text_inputs = tokenizer(
            row['text'],
            padding="max_length",
            truncation=True,
            max_length=CFG.max_len,
            return_tensors="pt"
        )
        
        # Vision Brain Input
        try:
            image = Image.open(row['image_path']).convert("RGB")
        except:
            # Fallback to black image if corrupted/missing so training doesn't crash
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
        
        item = {
            "input_ids": text_inputs["input_ids"].squeeze(0),
            "attention_mask": text_inputs["attention_mask"].squeeze(0),
            "pixel_values": pixel_values.squeeze(0),
        }
        
        if not self.is_test:
            item["labels"] = torch.tensor(row['label_id'], dtype=torch.long)
            
        return item

# --- Dual-Brain Multimodal Architecture ---
class MultimodalModel(nn.Module):
    def __init__(self, num_labels=5):
        super().__init__()
        # Load independent pre-trained encoders
        self.text_encoder = AutoModel.from_pretrained(CFG.text_model)
        self.vision_encoder = AutoModel.from_pretrained(CFG.vision_model)
        
        # Freeze initial layers to prevent catastrophic forgetting (Optional, let's train all for max score)
        text_dim = self.text_encoder.config.hidden_size # 1024 for banglabert_large
        vision_dim = self.vision_encoder.config.hidden_size # 768 for vit-base
        
        # Classifier Head (with heavy dropout to prevent MLP overfit)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(text_dim + vision_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )
        
    def forward(self, input_ids, attention_mask, pixel_values):
        # Extract Intelligence from Text
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(text_outputs, 'pooler_output') and text_outputs.pooler_output is not None:
            text_features = text_outputs.pooler_output
        else:
            text_features = text_outputs.last_hidden_state[:, 0, :] # [CLS] token
            
        # Extract Intelligence from Image
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        vision_features = vision_outputs.last_hidden_state[:, 0, :] # [CLS] token
        
        # Fuse the Brains
        combined_features = torch.cat((text_features, vision_features), dim=1)
        logits = self.classifier(combined_features)
        return logits

# --- Training Loop ---
test_dataset = MultimodalDataset(test, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False)
test_predictions = []

os.makedirs("/kaggle/working/models", exist_ok=True)

for fold in range(CFG.n_folds):
    print(f"\n{'='*20} FOLD {fold+1}/{CFG.n_folds} {'='*20}")
    
    trn_df = train[train['fold'] != fold].reset_index(drop=True)
    val_df = train[train['fold'] == fold].reset_index(drop=True)
    
    trn_ds = MultimodalDataset(trn_df)
    val_ds = MultimodalDataset(val_df)
    
    trn_loader = DataLoader(trn_ds, batch_size=CFG.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False)
    
    model = MultimodalModel()
    
    # Enable T4 x2 Dual-GPU Training
    if torch.cuda.device_count() > 1:
        print(f"🚀 Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
        
    model = model.to(CFG.device)
    
    # Handle DataParallel .module wrapper for parameter groups
    base_model = model.module if hasattr(model, "module") else model
    
    # FIX 5: Parameter groups for different learning rates
    optimizer = torch.optim.AdamW([
        {'params': base_model.text_encoder.parameters(),   'lr': CFG.lr},
        {'params': base_model.vision_encoder.parameters(), 'lr': CFG.lr},
        {'params': base_model.classifier.parameters(),     'lr': 1e-4},
    ], weight_decay=0.01)
    
    total_steps = len(trn_loader) * CFG.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps*0.1), num_training_steps=total_steps)
    criterion = nn.CrossEntropyLoss(weight=LOSS_WEIGHTS)
    
    # FIX 3: Mixed Precision Scaler for T4 OOM/Speed
    scaler = torch.cuda.amp.GradScaler()
    
    best_f1 = 0
    best_model_path = f"/kaggle/working/models/best_fold_{fold}.pt"
    
    for epoch in range(CFG.epochs):
        model.train()
        train_loss = 0
        
        print(f"Epoch {epoch+1}/{CFG.epochs}")
        pbar = tqdm(trn_loader, total=len(trn_loader), desc="Training")
        
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(CFG.device)
            attention_mask = batch['attention_mask'].to(CFG.device)
            pixel_values = batch['pixel_values'].to(CFG.device)
            labels = batch['labels'].to(CFG.device)
            
            # FIX 3: Mixed Precision Autocast
            with torch.cuda.amp.autocast():
                logits = model(input_ids, attention_mask, pixel_values)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            
            # FIX 4: Gradient Clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        model.eval()
        val_preds = []
        val_labels = []
        with torch.no_grad():
            for batch in tqdm(val_loader, total=len(val_loader), desc="Validating"):
                input_ids = batch['input_ids'].to(CFG.device)
                attention_mask = batch['attention_mask'].to(CFG.device)
                pixel_values = batch['pixel_values'].to(CFG.device)
                labels = batch['labels'].to(CFG.device)
                
                logits = model(input_ids, attention_mask, pixel_values)
                val_preds.extend(logits.argmax(-1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
        _, _, f1, _ = precision_recall_fscore_support(val_labels, val_preds, average='weighted', zero_division=0)
        acc = accuracy_score(val_labels, val_preds)
        print(f"Epoch {epoch+1} - Loss: {train_loss/len(trn_loader):.4f} - Val Acc: {acc:.4f} - Val F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            # Save safely to disk so we don't blow up VRAM, but overwrite the same file to save disk space
            torch.save(model.state_dict(), best_model_path)
            
    # Load best model for predicting test
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    print(f"Predicting Test Set for Fold {fold+1}...")
    fold_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, total=len(test_loader), desc="Testing"):
            input_ids = batch['input_ids'].to(CFG.device)
            attention_mask = batch['attention_mask'].to(CFG.device)
            pixel_values = batch['pixel_values'].to(CFG.device)
            
            logits = model(input_ids, attention_mask, pixel_values)
            fold_preds.append(logits.cpu().numpy())
            
    test_predictions.append(np.vstack(fold_preds))
    
    # --- CRITICAL FIX: FREE DISK SPACE AND VRAM ---
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    if os.path.exists(best_model_path):
        os.remove(best_model_path)

# --- Bayesian Post-Processing ---
print("\n--- Applying Grandmaster Logit Biasing (Multimodal Post-Processing) ---")
final_logits = np.mean(test_predictions, axis=0)
bayesian_logits = np.zeros_like(final_logits)

# Claude's higher alpha for edge-case tie-breaking
alpha = 0.35

for i, row in test.iterrows():
    category = row['category']
    prior_probs = historical_probs_dict.get(category, np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
    log_priors = np.log(prior_probs + 1e-6)
    bayesian_logits[i] = final_logits[i] + (alpha * log_priors)

final_preds = np.argmax(bayesian_logits, axis=-1)
test['label'] = [reverse_label_map[p] for p in final_preds]

print("Applying Rule-Based Category Hack for Non Disaster...")
test.loc[test['category'] == 'Non Disaster', 'label'] = 'Minimal'

submission = test[['image_id', 'label']]
submission.to_csv("submission.csv", index=False)

with zipfile.ZipFile('submission_bayesian.zip', 'w') as zipf:
    zipf.write('submission.csv', arcname='submission.csv')

print("\n✅ Multimodal Bayesian Pipeline Complete! 'submission_bayesian.zip' is ready for upload.")
