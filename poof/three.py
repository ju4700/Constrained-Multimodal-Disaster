import os
import random
import warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoTokenizer, AutoModel, set_seed
import timm
from tqdm import tqdm
import zipfile

warnings.filterwarnings("ignore")

# ==========================================
# STEP 1: SETUP & REPRODUCIBILITY
# ==========================================
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
    text_model_name = "csebuetnlp/banglabert"
    image_model_name = "tf_efficientnetv2_s"
    base_path = "/kaggle/input/competitions/datathon-iiuc-cse-fest-2026/DisasterSeverity/"
    max_len = 128            # Keeping to 128 to save VRAM for the image model
    img_size = 224
    epochs = 4               # 4 Epochs per fold is usually enough for fine-tuning
    batch_size = 16          # Fits on T4x2
    lr = 2e-5
    n_folds = 5
    seed = 42
    
label_map = {"Minimal": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Catastrophic": 4}
reverse_label_map = {v: k for k, v in label_map.items()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# STEP 3: DATASET DEFINITION
# ==========================================
class MultimodalDisasterDataset(Dataset):
    def __init__(self, df, tokenizer, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Image
        img_path = str(self.df.loc[idx, 'img_path'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
            
        # Text
        text = str(self.df.loc[idx, 'context'])
        encoded = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=CFG.max_len,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        if self.is_test:
            return image, input_ids, attention_mask
        else:
            label = self.df.loc[idx, 'label_id']
            return image, input_ids, attention_mask, torch.tensor(label, dtype=torch.long)

train_transform = transforms.Compose([
    transforms.Resize((CFG.img_size, CFG.img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((CFG.img_size, CFG.img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# STEP 4: MULTIMODAL MODEL (EARLY FUSION)
# ==========================================
class MultimodalFusionModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        # Text Backbone
        self.text_model = AutoModel.from_pretrained(CFG.text_model_name)
        text_dim = self.text_model.config.hidden_size # Usually 768
        
        # Image Backbone (num_classes=0 removes the final classification head and returns pooled features)
        self.image_model = timm.create_model(CFG.image_model_name, pretrained=True, num_classes=0)
        img_dim = self.image_model.num_features # 1280 for efficientnetv2_s
        
        # Fusion Head
        self.fc = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(text_dim + img_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, images, input_ids, attention_mask):
        # Extract Text Features (CLS Token)
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state[:, 0, :] # Shape: (batch_size, 768)
        
        # Extract Image Features
        image_features = self.image_model(images) # Shape: (batch_size, 1280)
        
        # Fuse
        fused_features = torch.cat((text_features, image_features), dim=1) # Shape: (batch_size, 2048)
        logits = self.fc(fused_features)
        
        return logits

# ==========================================
# STEP 5: PREPARE DATA
# ==========================================
print("Loading Data...")
train = pd.read_csv(f"{CFG.base_path}train.csv")
test = pd.read_csv(f"{CFG.base_path}test.csv")
val = pd.read_csv(f"{CFG.base_path}validation.csv")

# Create absolute paths BEFORE concatenating because images are in different folders!
train['img_path'] = train['image_name'].apply(lambda x: f"{CFG.base_path}Train/{x}")
val['img_path'] = val['image_name'].apply(lambda x: f"{CFG.base_path}Validation/{x}")
test['img_path'] = test['image_name'].apply(lambda x: f"{CFG.base_path}Test/{x}")

# Merge train and val
train = pd.concat([train, val]).reset_index(drop=True)
train['label_id'] = train['label'].map(label_map)

# Handle missing text
train['context'] = train['context'].fillna("")
test['context'] = test['context'].fillna("")

# 🚨 THE DATASET HACK: Prepend category to the context text 🚨

# Stratified K-Fold
skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
train['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(train, train['label_id'])):
    train.loc[val_idx, 'fold'] = fold

tokenizer = AutoTokenizer.from_pretrained(CFG.text_model_name)
test_dataset = MultimodalDisasterDataset(test, tokenizer, val_transform, is_test=True)
test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False)

# ==========================================
# STEP 6: TRAINING LOOP
# ==========================================
test_predictions = []

for fold in range(CFG.n_folds):
    print(f"\n{'='*20} FOLD {fold+1}/{CFG.n_folds} {'='*20}")
    
    trn_df = train[train['fold'] != fold].reset_index(drop=True)
    val_df = train[train['fold'] == fold].reset_index(drop=True)
    
    train_dataset = MultimodalDisasterDataset(trn_df, tokenizer, train_transform)
    val_dataset = MultimodalDisasterDataset(val_df, tokenizer, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False)
    
    model = MultimodalFusionModel()
    
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)
    
    # Class Weights
    class_counts = np.bincount(trn_df['label_id'].values)
    total_samples = len(trn_df)
    weights = total_samples / (len(class_counts) * class_counts)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=0.01)
    
    best_f1 = 0
    best_model_weights = None
    
    for epoch in range(CFG.epochs):
        model.train()
        train_loss = 0
        for images, input_ids, attention_mask, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(images, input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for images, input_ids, attention_mask, labels in val_loader:
                images = images.to(device)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                
                logits = model(images, input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.numpy())
                
        _, _, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
        print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val F1: {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            # Save best weights
            best_model_weights = {k: v.cpu() for k, v in model.state_dict().items()}
            
    # Load best weights to predict on Test
    model.load_state_dict(best_model_weights)
    model.eval()
    print(f"Predicting Test Set for Fold {fold+1}...")
    fold_preds = []
    with torch.no_grad():
        for images, input_ids, attention_mask in test_loader:
            images = images.to(device)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            logits = model(images, input_ids, attention_mask)
            fold_preds.append(logits.cpu().numpy())
            
    test_predictions.append(np.vstack(fold_preds))

# ==========================================
# STEP 7: ENSEMBLING & POST-PROCESSING
# ==========================================
print("\nEnsembling 5-Fold Multimodal Predictions...")
final_logits = np.mean(test_predictions, axis=0)
final_preds = np.argmax(final_logits, axis=-1)

test['label'] = [reverse_label_map[p] for p in final_preds]

print("Applying Rule-Based Category Hack...")
test.loc[test['category'] == 'Non Disaster', 'label'] = 'Minimal'

submission = test[['image_id', 'label']]
submission.to_csv("submission.csv", index=False)

with zipfile.ZipFile('submission.zip', 'w') as zipf:
    zipf.write('submission.csv', arcname='submission.csv')

print("\n✅ Multimodal Training Complete! 'submission.zip' is ready.")
print(submission.head(10))
