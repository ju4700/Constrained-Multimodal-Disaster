import os, random, warnings, zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from scipy.special import softmax as scipy_softmax
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Vision imports
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# NLP imports
from transformers import (
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, get_cosine_schedule_with_warmup, set_seed
)
from datasets import Dataset as HFDataset

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)

seed_everything(42)

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
BASE_PATH = "/kaggle/input/competitions/datathon-iiuc-cse-fest-2026/DisasterSeverity/"
WORK_DIR  = "/kaggle/working/"

IMG_DIRS = {
    "train"      : os.path.join(BASE_PATH, "Train"),
    "validation" : os.path.join(BASE_PATH, "Validation"),
    "test"       : os.path.join(BASE_PATH, "Test"),
}

label_map         = {"Minimal": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Catastrophic": 4}
reverse_label_map = {v: k for k, v in label_map.items()}
NUM_CLASSES       = 5

SHARED = dict(
    n_folds       = 5,    # 5-fold to fit within Kaggle timeout limits
    seed          = 42,
    bayesian_alpha= 0.20, # Validation-Weighted Prior
    enable_fusion = False # Set True for MLP Feature Fusion (fast but needs memory)
)

MODEL_CFGS = [
    # ── Text Models ──
    dict(
        key        = "banglabert_large",
        model_name = "csebuetnlp/banglabert_large",
        type       = "text",
        epochs     = 5,
        lr         = 1e-5,
        batch      = 4,      # OOM SAFE
        grad_acc   = 4,      # Effective 16
        fp16       = True,
        grad_ckpt  = True,
        max_len    = 256,    # Prevents truncation of long landslides
        rdrop      = False,
        awp        = True,
        llrd       = True,
        ema        = True
    ),
    # ── Vision Model ──
    dict(
        key        = "efficientnet",
        model_name = "tf_efficientnet_b4_ns",
        type       = "vision",
        epochs     = 8,
        lr         = 3e-4,
        batch      = 16,     # Fits easily on T4
        img_size   = 380,
        mixup_alpha= 0.40,
        drop_rate  = 0.30,
        tta_n      = 5,
        fp16       = True,
    )
]

# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING & HACK INJECTIONS
# ═══════════════════════════════════════════════════════════════════════════
print("Loading data...")
train_raw = pd.read_csv(os.path.join(BASE_PATH, "train.csv"))
val_raw   = pd.read_csv(os.path.join(BASE_PATH, "validation.csv"))
test      = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))

train_raw["img_split"] = "train"
val_raw["img_split"]   = "validation"
test["img_split"]      = "test"

train = pd.concat([train_raw, val_raw]).reset_index(drop=True)
train["label_id"] = train["label"].map(label_map)

# Image Paths
def get_img_path(row):
    return os.path.join(IMG_DIRS[row["img_split"]], row["image_name"])

train["image_path"] = train.apply(get_img_path, axis=1)
test["image_path"]  = test.apply(get_img_path, axis=1)

# Death Keywords Hack
DEATH_KWS = ['মৃত', 'নিহত', 'নিখোঁজ', 'হতাহত', 'মৃতদেহ']
def has_death_kw(text):
    return any(w in str(text) for w in DEATH_KWS)

# Text Prefix Injection (Native Bengali to prevent Tokenizer UNK issues)
def build_text(row):
    cat = row["category"]
    ctx = str(row["context"])
    words = len(ctx.split())
    
    # Native Bengali length descriptors
    if words < 8: tier = "সংক্ষিপ্ত"      # Brief
    elif words <= 25: tier = "মাঝারি"     # Medium
    else: tier = "বিস্তারিত"            # Detailed
    
    # Native Bengali fatality flag
    fatal = "হ্যাঁ" if has_death_kw(ctx) else "না" # Yes/No
    
    # Example: "Flood । দৈর্ঘ্য: সংক্ষিপ্ত । মৃত্যু: হ্যাঁ । <context>"
    return f"{cat} । দৈর্ঘ্য: {tier} । মৃত্যু: {fatal} । {ctx}"

train["text"] = train.apply(build_text, axis=1)
test["text"]  = test.apply(build_text, axis=1)

# Folds
skf = StratifiedKFold(n_splits=SHARED["n_folds"], shuffle=True, random_state=SHARED["seed"])
train["fold"] = -1
for fold, (_, vi) in enumerate(skf.split(train, train["label_id"])):
    train.loc[vi, "fold"] = fold

counts        = np.bincount(train["label_id"].values, minlength=NUM_CLASSES).astype(float)
CLASS_WEIGHTS = torch.tensor(len(train) / (NUM_CLASSES * counts), dtype=torch.float32)

val_counts = np.bincount(val_raw["label"].map(label_map).values, minlength=NUM_CLASSES)
prior_val  = val_counts / val_counts.sum()

# ═══════════════════════════════════════════════════════════════════════════
# TEXT ENGINE: ADVANCED TRAINER V3
# ═══════════════════════════════════════════════════════════════════════════
class AWP:
    def __init__(self, model, adv_lr=0.1, adv_eps=1e-4):
        self.model = model
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.backup = {}

    def attack(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data)
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self.adv_lr * param.grad / norm1 * norm2
                    param.data.add_(r_at)
                    param.data = self._project(name, param.data)

    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

    def _project(self, param_name, param_data):
        r = param_data - self.backup[param_name]
        if torch.norm(r) > self.adv_eps:
            r = self.adv_eps * r / torch.norm(r)
        return self.backup[param_name] + r

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (1 - self.decay) * param.data + self.decay * self.shadow[name]

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}

def get_llrd_params(model, lr, weight_decay=0.01, llrd=0.9):
    no_decay = ["bias", "LayerNorm.weight"]
    if hasattr(model, 'bert'): base = model.bert
    elif hasattr(model, 'roberta'): base = model.roberta
    elif hasattr(model, 'electra'): base = model.electra
    else: base = model.base_model
    
    layers = list(base.encoder.layer)
    n = len(layers)
    
    emb_lr = lr * (llrd ** n)
    params = [
        {"params": [p for n, p in base.embeddings.named_parameters() if not any(nd in n for nd in no_decay)], "lr": emb_lr, "weight_decay": weight_decay},
        {"params": [p for n, p in base.embeddings.named_parameters() if any(nd in n for nd in no_decay)], "lr": emb_lr, "weight_decay": 0.0},
    ]
    for i, layer in enumerate(layers):
        l_lr = lr * (llrd ** (n - i - 1))
        params += [
            {"params": [p for n, p in layer.named_parameters() if not any(nd in n for nd in no_decay)], "lr": l_lr, "weight_decay": weight_decay},
            {"params": [p for n, p in layer.named_parameters() if any(nd in n for nd in no_decay)], "lr": l_lr, "weight_decay": 0.0},
        ]
    params += [
        {"params": [p for n, p in model.classifier.named_parameters() if not any(nd in n for nd in no_decay)], "lr": lr, "weight_decay": weight_decay},
        {"params": [p for n, p in model.classifier.named_parameters() if any(nd in n for nd in no_decay)], "lr": lr, "weight_decay": 0.0},
    ]
    return params

def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    _, _, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1}

class AdvancedTrainerV3(Trainer):
    def __init__(self, *args, mcfg=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.mcfg = mcfg
        self._ema = None
        self._step = 0

    def _ce(self, logits, labels):
        wt = CLASS_WEIGHTS.to(logits.device)
        return nn.CrossEntropyLoss(weight=wt, label_smoothing=0.05)(logits, labels)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        if model.training and self.mcfg["rdrop"] and not getattr(self, "_awp_mode", False):
            out1 = model(**inputs)
            out2 = model(**inputs)
            ce   = (self._ce(out1.logits, labels) + self._ce(out2.logits, labels)) / 2
            p1   = F.softmax(out1.logits, dim=-1)
            p2   = F.softmax(out2.logits, dim=-1)
            kl   = (F.kl_div(out1.logits.log_softmax(-1), p2, reduction="batchmean") +
                    F.kl_div(out2.logits.log_softmax(-1), p1, reduction="batchmean")) / 2
            loss = ce + 0.30 * kl
            return (loss, out1) if return_outputs else loss
        else:
            out  = model(**inputs)
            loss = self._ce(out.logits, labels)
            return (loss, out) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None, **kwargs):
        if self.mcfg["ema"] and self._ema is None:
            self._ema = EMA(self.model)
        self._step += 1
        model.train()
        inputs = self._prepare_inputs(inputs)
        labels = inputs["labels"].clone()

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        self.accelerator.backward(loss)

        if self.mcfg["awp"] and getattr(self.state, "epoch", 0.0) >= 2.0:
            awp = AWP(model)
            awp.attack()
            inputs["labels"] = labels
            self._awp_mode = True
            with self.compute_loss_context_manager():
                loss_adv = self.compute_loss(model, inputs)
            self._awp_mode = False
            if self.args.gradient_accumulation_steps > 1:
                loss_adv = loss_adv / self.args.gradient_accumulation_steps
            self.accelerator.backward(loss_adv)
            awp.restore()

        if self.mcfg["ema"] and self._ema is not None:
            self._ema.update()

        return loss.detach()

    def create_optimizer(self):
        if self.optimizer is None and self.mcfg["llrd"]:
            params = get_llrd_params(self.model, self.args.learning_rate)
            self.optimizer = torch.optim.AdamW(params)
        elif self.optimizer is None:
            self.optimizer = super().create_optimizer()
        return self.optimizer

    def evaluate(self, *args, **kwargs):
        if self.mcfg["ema"] and self._ema is not None: self._ema.apply_shadow()
        result = super().evaluate(*args, **kwargs)
        if self.mcfg["ema"] and self._ema is not None: self._ema.restore()
        return result

    def predict(self, *args, **kwargs):
        if self.mcfg["ema"] and self._ema is not None: self._ema.apply_shadow()
        result = super().predict(*args, **kwargs)
        if self.mcfg["ema"] and self._ema is not None: self._ema.restore()
        return result

def train_text_model(mcfg, train_df, test_df):
    key = mcfg["key"]
    tokenizer = AutoTokenizer.from_pretrained(mcfg["model_name"])
    def tok_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=mcfg["max_len"])

    tok_test = HFDataset.from_pandas(test_df[["text"]]).map(tok_fn, batched=True)
    fold_preds, cv_f1s = [], []

    for fold in range(SHARED["n_folds"]):
        print(f"\n{'='*15} [{key}] FOLD {fold+1} {'='*15}")
        trn_df = train_df[train_df["fold"] != fold].reset_index(drop=True)
        val_df = train_df[train_df["fold"] == fold].reset_index(drop=True)

        tok_trn = HFDataset.from_pandas(trn_df[["text", "label_id"]].rename(columns={"label_id": "label"})).map(tok_fn, batched=True)
        tok_val = HFDataset.from_pandas(val_df[["text", "label_id"]].rename(columns={"label_id": "label"})).map(tok_fn, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(mcfg["model_name"], num_labels=5)
        if mcfg["grad_ckpt"]: model.gradient_checkpointing_enable()

        args = TrainingArguments(
            output_dir=f"{WORK_DIR}{key}_f{fold}", eval_strategy="epoch", save_strategy="epoch",
            learning_rate=mcfg["lr"], per_device_train_batch_size=mcfg["batch"], per_device_eval_batch_size=mcfg["batch"]*2,
            gradient_accumulation_steps=mcfg["grad_acc"], num_train_epochs=mcfg["epochs"],
            warmup_ratio=0.10, weight_decay=0.01, fp16=mcfg["fp16"], lr_scheduler_type="cosine",
            load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
            report_to="none", save_total_limit=1
        )

        trainer = AdvancedTrainerV3(model=model, args=args, train_dataset=tok_trn, eval_dataset=tok_val, compute_metrics=compute_metrics, mcfg=mcfg)
        trainer.train()

        best_f1 = max(trainer.state.log_history, key=lambda x: x.get("eval_f1", -1)).get("eval_f1", 0)
        cv_f1s.append(best_f1)
        print(f"Fold {fold+1} F1: {best_f1:.4f}")

        fold_preds.append(trainer.predict(tok_test).predictions)
        del model, trainer; torch.cuda.empty_cache()

    return np.array(fold_preds), np.mean(cv_f1s)


# ═══════════════════════════════════════════════════════════════════════════
# VISION ENGINE: EFFICIENTNET
# ═══════════════════════════════════════════════════════════════════════════
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)

def get_train_transform(img_size):
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.15),
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=20, p=0.4),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=20, p=0.4),
        A.CoarseDropout(max_holes=8, max_height=img_size//8, max_width=img_size//8, fill_value=0, p=0.35),
        A.Normalize(mean=MEAN, std=STD), ToTensorV2(),
    ])

def get_val_transform(img_size):
    return A.Compose([A.Resize(img_size, img_size), A.Normalize(mean=MEAN, std=STD), ToTensorV2()])

def get_tta_transform(img_size, t):
    base = [A.Resize(img_size, img_size)]
    augs = [
        [], [A.HorizontalFlip(p=1.0)], [A.VerticalFlip(p=1.0)], [A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0)],
        [A.Resize(int(img_size*1.1), int(img_size*1.1)), A.CenterCrop(img_size, img_size)]
    ]
    return A.Compose(base + augs[t % 5] + [A.Normalize(mean=MEAN, std=STD), ToTensorV2()])

class DisasterImageDataset(Dataset):
    def __init__(self, df, transform, has_label=True):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.has_label = has_label
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try: img = np.array(Image.open(row["image_path"]).convert("RGB"))
        except: img = np.zeros((380, 380, 3), dtype=np.uint8)
        img = self.transform(image=img)["image"]
        if self.has_label: return img, torch.tensor(row["label_id"], dtype=torch.long)
        return img

class ImageClassifier(nn.Module):
    def __init__(self, model_name, num_classes=5, drop_rate=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, global_pool="avg")
        n_feat = self.backbone.num_features
        self.head = nn.Sequential(nn.BatchNorm1d(n_feat), nn.Dropout(drop_rate), nn.Linear(n_feat, num_classes))
    def get_features(self, x): return self.backbone(x)
    def forward(self, x): return self.head(self.get_features(x))

class FocalCELoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05, reduction="none")
    def forward(self, logits, labels):
        loss = self.ce(logits, labels)
        pt = torch.exp(-loss)
        return (((1 - pt) ** 2.0) * loss).mean()

def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam

def train_vision_model(mcfg, train_df, test_df):
    key = mcfg["key"]
    fold_preds, cv_f1s = [], []
    
    for fold in range(SHARED["n_folds"]):
        print(f"\n{'='*15} [{key}] FOLD {fold+1} {'='*15}")
        trn_df = train_df[train_df["fold"] != fold]
        val_df = train_df[train_df["fold"] == fold]
        
        trn_loader = DataLoader(DisasterImageDataset(trn_df, get_train_transform(mcfg["img_size"])), batch_size=mcfg["batch"], shuffle=True, drop_last=True)
        val_loader = DataLoader(DisasterImageDataset(val_df, get_val_transform(mcfg["img_size"])), batch_size=mcfg["batch"], shuffle=False)
        
        model = ImageClassifier(mcfg["model_name"], NUM_CLASSES, mcfg["drop_rate"]).to(DEVICE)
        criterion = FocalCELoss(CLASS_WEIGHTS.to(DEVICE))
        optimizer = torch.optim.AdamW([
            {"params": model.backbone.parameters(), "lr": mcfg["lr"] * 0.1},
            {"params": model.head.parameters(), "lr": mcfg["lr"]}
        ], weight_decay=1e-2)
        
        steps = len(trn_loader) * mcfg["epochs"]
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*steps), num_training_steps=steps)
        scaler = torch.cuda.amp.GradScaler()
        
        best_f1, best_w = 0.0, None
        
        for ep in range(mcfg["epochs"]):
            model.train()
            for imgs, labels in trn_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                use_mix = mcfg["mixup_alpha"] > 0 and random.random() < 0.5
                if use_mix: imgs, y_a, y_b, lam = mixup_data(imgs, labels, mcfg["mixup_alpha"])
                
                with torch.cuda.amp.autocast(enabled=mcfg["fp16"]):
                    logits = model(imgs)
                    loss = (lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)) if use_mix else criterion(logits, labels)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
            model.eval()
            all_p, all_l = [], []
            with torch.no_grad():
                for imgs, labels in val_loader:
                    with torch.cuda.amp.autocast(enabled=mcfg["fp16"]):
                        logits = model(imgs.to(DEVICE))
                    all_p.extend(logits.argmax(1).cpu().numpy())
                    all_l.extend(labels.numpy())
            _, _, f1, _ = precision_recall_fscore_support(all_l, all_p, average="weighted", zero_division=0)
            print(f"  Epoch {ep+1} | val_f1={f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                best_w = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                
        cv_f1s.append(best_f1)
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_w.items()})
        
        # TTA Predict
        model.eval()
        t_logits = []
        for t in range(mcfg["tta_n"]):
            loader = DataLoader(DisasterImageDataset(test_df, get_tta_transform(mcfg["img_size"], t), False), batch_size=mcfg["batch"])
            b_log = []
            with torch.no_grad():
                for imgs in loader:
                    with torch.cuda.amp.autocast(enabled=mcfg["fp16"]): b_log.append(model(imgs.to(DEVICE)).float().cpu().numpy())
            t_logits.append(np.concatenate(b_log))
        fold_preds.append(np.mean(t_logits, axis=0))
        del model; torch.cuda.empty_cache()
        
    return np.array(fold_preds), np.mean(cv_f1s)

# ═══════════════════════════════════════════════════════════════════════════
# EXECUTION
# ═══════════════════════════════════════════════════════════════════════════
all_logits, all_cv = {}, {}

for mcfg in MODEL_CFGS:
    seed_everything(SHARED["seed"])
    if mcfg["type"] == "text":
        preds, cv_f1 = train_text_model(mcfg, train, test)
    else:
        preds, cv_f1 = train_vision_model(mcfg, train, test)
    all_logits[mcfg["key"]] = preds.mean(axis=0) # Average across folds
    all_cv[mcfg["key"]] = cv_f1

total_score = sum(all_cv.values())
weights = {k: v / total_score for k, v in all_cv.items()}
print("Ensemble Weights:", weights)

ensemble_logits = sum(weights[k] * all_logits[k] for k in all_logits)

# ── Data-Driven Logit Hacking (Death Keywords) ──
adjusted_logits = ensemble_logits.copy()
for idx, row in test.iterrows():
    if not has_death_kw(row["context"]): continue
    cat = row["category"]
    if cat == "Wildfire":   adjusted_logits[idx, label_map["Catastrophic"]] += 2.6
    elif cat == "Flood":    adjusted_logits[idx, label_map["Catastrophic"]] += 1.5
    elif cat == "Earthquake": adjusted_logits[idx, label_map["Severe"]] += 0.8
    elif cat == "Landslides": adjusted_logits[idx, label_map["Severe"]] += 1.3

# ── Validation-Weighted Bayesian Prior ──
probs = scipy_softmax(adjusted_logits, axis=-1)
for i in range(len(probs)):
    probs[i] = probs[i] * (prior_val ** SHARED["bayesian_alpha"])
    probs[i] /= probs[i].sum()

final_preds = np.argmax(probs, axis=-1)
test["label"] = [reverse_label_map[p] for p in final_preds]

# ── Hard Rule ──
test.loc[test["category"] == "Non Disaster", "label"] = "Minimal"

submission = test[["image_id", "label"]]
submission.to_csv("submission.csv", index=False)
with zipfile.ZipFile("submission_multimodal_grandmaster.zip", "w") as z:
    z.write("submission.csv", arcname="submission.csv")

print("✅ submission_multimodal_grandmaster.zip ready.")
print(submission["label"].value_counts())
