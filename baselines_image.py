"""
baselines_image.py
Image-only baseline models on BanglaCalamityMMD dataset.
Models: ResNet-50, EfficientNet-B0, Swin-Tiny (image-only)
Saves: baselines_image_results.csv
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import timm
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
from tqdm.auto import tqdm
warnings.filterwarnings("ignore")

class CFG:
    seed       = 42
    base_path  = "/kaggle/input/banglaclamity/BanglaCalamityMMD"
    train_csv  = os.path.join(base_path, "Disaster_train.csv")
    valid_csv  = os.path.join(base_path, "Disaster_validation.csv")
    test_csv   = os.path.join(base_path, "Disaster_test.csv")
    batch_size = 16
    epochs     = 5
    lr         = 1e-4
    weight_decay = 1e-4
    output_csv = "baselines_image_results.csv"

def seed_everything(seed=CFG.seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_everything()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

train_df = pd.read_csv(CFG.train_csv)
valid_df = pd.read_csv(CFG.valid_csv)
test_df  = pd.read_csv(CFG.test_csv)

le = LabelEncoder()
train_df['label'] = le.fit_transform(train_df['category'].astype(str).str.strip())
valid_df['label'] = le.transform(valid_df['category'].astype(str).str.strip())
test_df['label']  = le.transform(test_df['category'].astype(str).str.strip())
num_classes = len(le.classes_)
print(f"Classes ({num_classes}): {le.classes_}")

def get_ext_map(folder):
    if not os.path.exists(folder): return {}
    return {os.path.splitext(f)[0]: f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))}

train_ext_map = get_ext_map(os.path.join(CFG.base_path, "Train"))
valid_ext_map = get_ext_map(os.path.join(CFG.base_path, "Validation"))
test_ext_map  = get_ext_map(os.path.join(CFG.base_path, "Test"))

results = []


def record(name, modality, test_preds, test_trues):
    f1  = f1_score(test_trues, test_preds, average='macro')
    acc = accuracy_score(test_trues, test_preds)
    print(f"[{name}] Test Macro F1={f1:.4f}, Acc={acc:.4f}")
    results.append({"Model": name, "Modality": modality,
                    "Test Macro F1": round(f1, 4), "Test Accuracy": round(acc, 4)})


class ImageOnlyDataset(Dataset):
    def __init__(self, df, folder, ext_map, transform=None):
        self.df        = df
        self.folder    = folder
        self.ext_map   = ext_map
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        label    = row.get('label', -1)
        image_id = str(row['image_id'])

        img_filename = self.ext_map.get(image_id, image_id + ".jpg")
        img_path     = os.path.join(self.folder, img_filename)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), (128, 128, 128))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_ds = ImageOnlyDataset(train_df, os.path.join(CFG.base_path, "Train"),      train_ext_map, train_transform)
test_ds  = ImageOnlyDataset(test_df,  os.path.join(CFG.base_path, "Test"),       test_ext_map,  val_transform)

train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True,  num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=CFG.batch_size, shuffle=False, num_workers=4)


def run_image_model(timm_name, display_name):
    print(f"\n=== Running {display_name} ===")
    model = timm.create_model(timm_name, pretrained=True, num_classes=num_classes)
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs * len(train_loader))
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(CFG.epochs):
        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CFG.epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(images)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                scheduler.step()

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            logits = model(images)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            trues.extend(labels.numpy())

    record(display_name, "Image-Only", np.array(preds), np.array(trues))


run_image_model("resnet50",         "ResNet-50")
run_image_model("efficientnet_b0",  "EfficientNet-B0")
run_image_model("swin_tiny_patch4_window7_224", "Swin-Tiny (Image-Only)")


results_df = pd.DataFrame(results)
print("\n=== IMAGE BASELINE RESULTS ===")
print(results_df.to_string(index=False))
results_df.to_csv(CFG.output_csv, index=False)
print(f"\nSaved to {CFG.output_csv}")
