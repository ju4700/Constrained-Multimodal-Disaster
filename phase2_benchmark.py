import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel
import timm
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

class CFG:
    seed = 42
    base_path = "/kaggle/input/datasets/seaninggg/crisismmd-multimodal-crisis-dataset/CrisisMMD_v2.0"
    tsv_file = os.path.join(base_path, "annotations", "hurricane_harvey_final_data.tsv")
    text_model = "xlm-roberta-large"
    vision_model = "swin_tiny_patch4_window7_224"
    max_len = 128
    batch_size = 4  # Reduced batch size for xlm-roberta-large
    epochs = 3
    lr = 2e-5
    weight_decay = 1e-4

def seed_everything(seed=CFG.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CrisisDataset(Dataset):
    def __init__(self, df, tokenizer, folder, transform=None):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.folder = folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row['tweet_text'])
        label = row.get('label', -1)
        image_path = str(row['image_path'])
        
        img_full_path = os.path.join(self.folder, image_path)
        
        try:
            image = Image.open(img_full_path).convert('RGB')
        except:
            image = Image.new('RGB', (224, 224), (255, 255, 255))
            
        if self.transform:
            image = self.transform(image)
            
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=CFG.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'pixel_values': image,
            'label': torch.tensor(label, dtype=torch.long)
        }

class MultimodalModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.text_model = AutoModel.from_pretrained(CFG.text_model)
        self.vision_model = timm.create_model(CFG.vision_model, pretrained=True, num_classes=0)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(self.text_model.config.hidden_size + self.vision_model.num_features, num_classes)
        
    def forward(self, input_ids, attention_mask, pixel_values):
        text_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_features = text_out.pooler_output if hasattr(text_out, 'pooler_output') and text_out.pooler_output is not None else text_out.last_hidden_state[:, 0, :]
        vision_features = self.vision_model(pixel_values)
        features = torch.cat([text_features, vision_features], dim=1)
        features = self.drop(features)
        out = self.fc(features)
        return out

def train_epoch(model, dataloader, optimizer, scaler, scheduler):
    model.train()
    losses = []
    criterion = nn.CrossEntropyLoss()
    for batch in tqdm(dataloader, desc="Training Phase 2"):
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['label'].to(device)
        
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(input_ids, attention_mask, pixel_values)
            loss = criterion(logits, labels)
            
        scaler.scale(loss).backward()
        
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()
        skip_lr_sched = (scale_after < scale_before)
        
        if not skip_lr_sched:
            scheduler.step()
            
        losses.append(loss.item())
    return np.mean(losses)

def valid_epoch(model, dataloader):
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation Phase 2"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask, pixel_values)
            pred = torch.argmax(logits, dim=1)
            
            preds.extend(pred.cpu().numpy())
            trues.extend(labels.cpu().numpy())
            
    return np.array(preds), np.array(trues)

def run_phase2():
    df = pd.read_csv(CFG.tsv_file, sep='\t')
    # Filter valid labels
    df = df[df['image_damage'].notnull() & (df['image_damage'] != 'None')]
    
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['image_damage'])
    num_classes = len(le.classes_)
    
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=CFG.seed, stratify=df['label'])
    
    tokenizer = AutoTokenizer.from_pretrained(CFG.text_model)
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = CrisisDataset(train_df, tokenizer, CFG.base_path, train_transform)
    test_dataset = CrisisDataset(test_df, tokenizer, CFG.base_path, val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False)
    
    print("Running Phase 2 Generalization Benchmark")
    model = MultimodalModel(num_classes).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs*len(train_loader))
    
    best_f1 = 0
    for epoch in range(CFG.epochs):
        train_epoch(model, train_loader, optimizer, scaler, scheduler)
        preds, trues = valid_epoch(model, test_loader)
        f1 = f1_score(trues, preds, average='macro')
        acc = accuracy_score(trues, preds)
        print(f"Phase 2 - Epoch {epoch+1}: Macro F1={f1:.4f}, Acc={acc:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            
    print(f"Phase 2 Best F1={best_f1:.4f}")
    
    results = pd.DataFrame({
        "Phase": ["Phase 2 (CrisisMMD)"],
        "Macro_F1": [best_f1]
    })
    results.to_csv("phase2_results.csv", index=False)

if __name__ == "__main__":
    run_phase2()
