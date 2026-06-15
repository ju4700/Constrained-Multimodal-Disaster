import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import timm
from tqdm import tqdm
import os

# 1. Setup
base_path = "/kaggle/input/competitions/datathon-iiuc-cse-fest-2026/DisasterSeverity/"
train_df = pd.read_csv(f"{base_path}train.csv")
label_map = {"Minimal": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Catastrophic": 4}
train_df['label'] = train_df['label'].map(label_map)

# Split into 80% train, 20% validation to test the signal
train_data, val_data = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=42)

# 2. PyTorch Dataset for Images
class DisasterImageDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Assumes image_name column has the exact file names
        img_name = str(self.df.loc[idx, 'image_name'])
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        label = self.df.loc[idx, 'label']
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.long)

# 3. Fast Image Augmentations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_loader = DataLoader(DisasterImageDataset(train_data, f"{base_path}Train/", train_transform), batch_size=32, shuffle=True)
val_loader = DataLoader(DisasterImageDataset(val_data, f"{base_path}Train/", val_transform), batch_size=32, shuffle=False)

# 4. Load EfficientNetV2-S (Lightning fast on Kaggle T4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('tf_efficientnetv2_s', pretrained=True, num_classes=5)

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)

model.to(device)

# Add Class Weights for Imbalanced Severities
class_counts = np.bincount(train_data['label'].values)
total_samples = len(train_data)
weights = total_samples / (len(class_counts) * class_counts)
weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=weights_tensor)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 5. Quick 3-Epoch Training Loop to check for a signal
print("Starting Image-Only Signal Test...")
for epoch in range(3):
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    # Evaluate
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    _, _, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted', zero_division=0)
    print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Validation F1: {f1:.4f}")

print("\nTEST COMPLETE. If Validation F1 is < 0.35, the images are noise. If F1 > 0.50, we should fuse the models.")