"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  DISASTER SEVERITY – VISION ONLY PIPELINE                                  ║
║  Model: EfficientNet-B4-NS (No Text)                                       ║
║  Purpose: Benchmark raw visual signal before multimodal fusion             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1: Imports & Seed
# ═══════════════════════════════════════════════════════════════════════════════
import os, random, warnings, zipfile, gc, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset, DataLoader

from PIL import Image
from scipy.special import softmax as scipy_softmax

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.optim.lr_scheduler import CosineAnnealingLR

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
START_TIME = time.time()

def elapsed():
    return f"{(time.time() - START_TIME) / 60:.1f} min"

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)
print(f"Device: {DEVICE}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2: Configuration
# ═══════════════════════════════════════════════════════════════════════════════
BASE_PATH = "/kaggle/input/competitions/datathon-iiuc-cse-fest-2026/DisasterSeverity/"
WORK_DIR  = "/kaggle/working/"

IMG_DIRS = {
    "train":      os.path.join(BASE_PATH, "Train"),
    "validation": os.path.join(BASE_PATH, "Validation"),
    "test":       os.path.join(BASE_PATH, "Test"),
}

label_map         = {"Minimal": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Catastrophic": 4}
reverse_label_map = {v: k for k, v in label_map.items()}
NUM_CLASSES       = 5

CFG = dict(
    model_name   = "tf_efficientnet_b4_ns",
    epochs       = 12,        # Slightly more epochs since there's no text model
    lr           = 3e-4,
    backbone_lr  = 3e-5,
    batch        = 16,
    img_size     = 380,
    mixup_alpha  = 0.4,
    cutmix_alpha = 0.4,
    drop_rate    = 0.30,
    tta_n        = 5,
    fp16         = True,
    label_smooth = 0.05,
    focal_gamma  = 2.0,
    n_folds      = 5,
    seed         = 42,
    bayesian_alpha = 0.15,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3: Data Loading & Bayesian Priors
# ═══════════════════════════════════════════════════════════════════════════════
print(f"[{elapsed()}] Loading data...")
train_raw = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
val_raw   = pd.read_csv(os.path.join(BASE_PATH, "validation.csv"))
test      = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))

train_raw["img_split"] = "train"
val_raw["img_split"]   = "validation"
test["img_split"]      = "test"

# Bayesian Prior based on tabular 'category' distribution
cross_all = pd.crosstab(
    pd.concat([train_raw, val_raw], ignore_index=True)["category"],
    pd.concat([train_raw, val_raw], ignore_index=True)["label"].map(label_map),
    normalize="index"
)
cross_val = pd.crosstab(
    val_raw["category"],
    val_raw["label"].map(label_map),
    normalize="index"
)

BAYESIAN_PRIORS = {}
for cat in cross_all.index:
    p_all = cross_all.loc[cat].reindex(range(5), fill_value=0).values
    p_val = cross_val.loc[cat].reindex(range(5), fill_value=0).values if cat in cross_val.index else p_all
    BAYESIAN_PRIORS[cat] = 0.40 * p_all + 0.60 * p_val
BAYESIAN_PRIORS["Non Disaster"] = np.array([1.00, 0.00, 0.00, 0.00, 0.00])

train = pd.concat([train_raw, val_raw]).reset_index(drop=True)
train["label_id"] = train["label"].map(label_map)

def get_img_path(row):
    return os.path.join(IMG_DIRS[row["img_split"]], row["image_name"])

train["image_path"] = train.apply(get_img_path, axis=1)
test["image_path"]  = test.apply(get_img_path, axis=1)

skf = StratifiedKFold(n_splits=CFG["n_folds"], shuffle=True, random_state=CFG["seed"])
train["fold"] = -1
for fold, (_, vi) in enumerate(skf.split(train, train["label_id"])):
    train.loc[vi, "fold"] = fold

counts = np.bincount(train["label_id"].values, minlength=NUM_CLASSES).astype(float)
CLASS_WEIGHTS = torch.tensor(len(train) / (NUM_CLASSES * counts), dtype=torch.float32)

print(f"Train samples: {len(train)} | Test samples: {len(test)}")
print(f"Class weights: {dict(zip(label_map.keys(), CLASS_WEIGHTS.numpy().round(3)))}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4: Image Augmentations & Dataset
# ═══════════════════════════════════════════════════════════════════════════════
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def get_train_transform(img_size):
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.1),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=20, p=0.4),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=15, p=0.4),
        A.CoarseDropout(
            max_holes=8, max_height=img_size // 8, max_width=img_size // 8,
            fill_value=0, p=0.3
        ),
        A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def get_val_transform(img_size):
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])

def get_tta_transforms(img_size):
    base_norm = [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return [
        A.Compose([A.Resize(height=img_size, width=img_size)] + base_norm),
        A.Compose([A.Resize(height=img_size, width=img_size), A.HorizontalFlip(p=1.0)] + base_norm),
        A.Compose([A.Resize(height=img_size, width=img_size), A.VerticalFlip(p=1.0)] + base_norm),
        A.Compose([A.Resize(height=img_size, width=img_size), A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0)] + base_norm),
        A.Compose([A.Resize(height=int(img_size * 1.1), width=int(img_size * 1.1)), A.CenterCrop(height=img_size, width=img_size)] + base_norm),
    ]

class DisasterImageDataset(TorchDataset):
    def __init__(self, df, transform, has_label=True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.has_label = has_label

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            img = np.array(Image.open(row["image_path"]).convert("RGB"))
        except Exception:
            img = np.zeros((CFG["img_size"], CFG["img_size"], 3), dtype=np.uint8)
        img = self.transform(image=img)["image"]
        if self.has_label:
            return img, torch.tensor(row["label_id"], dtype=torch.long)
        return img

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5: Model & Training Utilities
# ═══════════════════════════════════════════════════════════════════════════════
class ImageClassifier(nn.Module):
    def __init__(self, model_name, num_classes=5, drop_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=True, num_classes=0, global_pool="avg"
        )
        n_feat = self.backbone.num_features
        self.head = nn.Sequential(
            nn.BatchNorm1d(n_feat),
            nn.Dropout(drop_rate),
            nn.Linear(n_feat, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_rate * 0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

def focal_ce_loss(logits, labels, class_weights, label_smoothing=0.05, gamma=2.0):
    wt = class_weights.to(logits.device)
    ce = nn.CrossEntropyLoss(
        weight=wt, label_smoothing=label_smoothing, reduction="none"
    )(logits, labels)
    if gamma == 0:
        return ce.mean()
    pt = torch.exp(-ce)
    focal = ((1 - pt) ** gamma) * ce
    return focal.mean()

def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def cutmix_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    B, C, H, W = x.shape
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    x_new = x.clone()
    x_new[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return x_new, y, y[idx], lam

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6: Training Loop
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"  VISION MODEL: EfficientNet-B4-NS (Pure Images)")
print(f"{'='*60}")

fold_preds = []
cv_f1s = []
tta_transforms = get_tta_transforms(CFG["img_size"])

for fold in range(CFG["n_folds"]):
    t_start = time.time()
    print(f"\n  ── FOLD {fold+1}/{CFG['n_folds']} ──")

    seed_everything(CFG["seed"] + fold)

    trn_df = train[train["fold"] != fold]
    val_df = train[train["fold"] == fold]

    trn_loader = DataLoader(
        DisasterImageDataset(trn_df, get_train_transform(CFG["img_size"])),
        batch_size=CFG["batch"], shuffle=True, drop_last=True,
        num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        DisasterImageDataset(val_df, get_val_transform(CFG["img_size"])),
        batch_size=CFG["batch"], shuffle=False,
        num_workers=2, pin_memory=True,
    )

    model = ImageClassifier(
        CFG["model_name"], NUM_CLASSES, CFG["drop_rate"]
    ).to(DEVICE)

    cw = CLASS_WEIGHTS.to(DEVICE)

    optimizer = torch.optim.AdamW([
        {"params": model.backbone.parameters(), "lr": CFG["backbone_lr"]},
        {"params": model.head.parameters(), "lr": CFG["lr"]},
    ], weight_decay=1e-2)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    scheduler = CosineAnnealingLR(optimizer, T_max=CFG["epochs"] * len(trn_loader))
    scaler = torch.cuda.amp.GradScaler()

    best_f1 = 0.0
    best_weights = None

    for ep in range(CFG["epochs"]):
        # Train
        model.train()
        running_loss = 0.0

        for imgs, labels in trn_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            use_mix = random.random() < 0.5
            if use_mix:
                if random.random() < 0.5:
                    imgs, y_a, y_b, lam = mixup_data(imgs, labels, CFG["mixup_alpha"])
                else:
                    imgs, y_a, y_b, lam = cutmix_data(imgs, labels, CFG["cutmix_alpha"])

            with torch.cuda.amp.autocast(enabled=CFG["fp16"]):
                logits = model(imgs)
                if use_mix:
                    loss = (
                        lam * focal_ce_loss(logits, y_a, cw, CFG["label_smooth"], CFG["focal_gamma"]) +
                        (1 - lam) * focal_ce_loss(logits, y_b, cw, CFG["label_smooth"], CFG["focal_gamma"])
                    )
                else:
                    loss = focal_ce_loss(logits, labels, cw, CFG["label_smooth"], CFG["focal_gamma"])

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()

        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                with torch.cuda.amp.autocast(enabled=CFG["fp16"]):
                    logits = model(imgs.to(DEVICE))
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_labels.extend(labels.numpy())

        _, _, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted", zero_division=0
        )
        avg_loss = running_loss / len(trn_loader)
        print(f"    Ep {ep+1}/{CFG['epochs']} | loss={avg_loss:.4f} | val_f1={f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    cv_f1s.append(best_f1)

    # Load best weights for TTA prediction
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_weights.items()})
    model.eval()

    # TTA Prediction
    tta_logits_all = []
    for t_idx, tta_tfm in enumerate(tta_transforms):
        loader = DataLoader(
            DisasterImageDataset(test, tta_tfm, has_label=False),
            batch_size=CFG["batch"], shuffle=False,
            num_workers=2, pin_memory=True,
        )
        batch_logits = []
        with torch.no_grad():
            for imgs in loader:
                with torch.cuda.amp.autocast(enabled=CFG["fp16"]):
                    logits = model(imgs.to(DEVICE))
                batch_logits.append(logits.float().cpu().numpy())
        tta_logits_all.append(np.concatenate(batch_logits))

    fold_preds.append(np.mean(tta_logits_all, axis=0))
    print(f"  Fold {fold+1} F1: {best_f1:.4f} | Time: {(time.time() - t_start) / 60:.1f} min")

    del model, best_weights
    gc.collect()
    torch.cuda.empty_cache()

mean_f1 = np.mean(cv_f1s)
print(f"\n  Mean CV F1: {mean_f1:.4f} | Folds: {cv_f1s}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7: Post-Processing & Submission
# ═══════════════════════════════════════════════════════════════════════════════
# Average fold predictions
ensemble_logits = np.mean(fold_preds, axis=0)
vision_probs = scipy_softmax(ensemble_logits, axis=-1)

# Apply Bayesian Prior (since category is tabular, not text, we can use it!)
alpha = CFG["bayesian_alpha"]
test.reset_index(drop=True, inplace=True)
adjusted_logits = np.zeros_like(vision_probs)

for i, row in test.iterrows():
    cat = row["category"]
    prior = BAYESIAN_PRIORS.get(cat, np.ones(5) / 5)
    p_blend = vision_probs[i] + 1e-8
    p_prior = prior + 1e-8
    adjusted_logits[i] = np.log(p_blend) + alpha * np.log(p_prior)

final_preds = np.argmax(adjusted_logits, axis=-1)
test["label"] = [reverse_label_map[p] for p in final_preds]

# Hard Rules
test.loc[test["category"] == "Non Disaster", "label"] = "Minimal"
test.loc[(test["category"] == "Tropical Storm") & (test["label"] == "Catastrophic"), "label"] = "Severe"
test.loc[(test["category"] == "Flood") & (test["label"] == "Minimal"), "label"] = "Mild"

submission = test[["image_id", "label"]]
submission.to_csv("submission.csv", index=False)

with zipfile.ZipFile("submission_vision_only.zip", "w") as z:
    z.write("submission.csv", arcname="submission.csv")

print(f"\n{'='*60}")
print(f"  ✅ SUBMISSION READY: submission_vision_only.zip")
print(f"  Total runtime: {elapsed()}")
print(f"{'='*60}")
print(f"\nPrediction distribution:")
print(submission["label"].value_counts())
