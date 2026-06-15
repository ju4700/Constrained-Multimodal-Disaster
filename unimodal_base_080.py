# ==============================================================================
# DisasterSeverity - 0.82 Target (Base Model Edition)
# ==============================================================================
# This notebook targets an ambitious 0.82+ F1 score using ONLY the base 
# 'csebuetnlp/banglabert' model. Because the base model lacks the massive 
# capacity of large models, we use a "kitchen sink" regularization approach:
#   1. R-Drop (Symmetric KL Divergence)
#   2. FGM (Adversarial Embedding Perturbation)
#   3. EMA (Exponential Moving Average of Weights)
#   4. LLRD (Layer-wise Learning Rate Decay)
#   5. Focal Loss (γ=2.0)
#   6. Bayesian Prior Post-Processing
#   7. Soft Pseudo-Labeling
# ==============================================================================

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1: Imports & Setup
# ═══════════════════════════════════════════════════════════════════════════════
import os, gc, random, warnings, re
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import zipfile
from scipy.special import softmax as scipy_softmax
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed
)
from datasets import Dataset

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)

seed_everything(42)

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2: Configuration
# ═══════════════════════════════════════════════════════════════════════════════
class CFG:
    base_path   = "/kaggle/input/competitions/datathon-iiuc-cse-fest-2026/DisasterSeverity/"
    model_name  = "csebuetnlp/banglabert"  # Using base model for fast training
    max_len     = 256                      # Generous length to capture context
    n_folds     = 5
    epochs      = 5
    batch_size  = 16
    lr          = 2e-5
    
    # ── Regularization & Optimization ──
    fp16            = True
    use_rdrop       = True
    rdrop_alpha     = 0.50
    use_fgm         = True
    fgm_epsilon     = 0.50
    use_ema         = False     # Disabled to prevent shadow weight drag on 5-epoch runs
    ema_decay       = 0.999
    use_llrd        = False     # Disabled to allow bottom layers to train fast
    llrd_decay      = 0.90
    label_smooth    = 0.10
    focal_gamma     = 0.0       # Reverting to Standard CE loss (matches baseline)
    
    # ── Pseudo-Labeling ──
    use_pseudo       = True
    pseudo_threshold = 0.85
    pseudo_epochs    = 3
    pseudo_lr        = 1e-5
    pseudo_blend     = 0.35

label_map = {"Minimal": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Catastrophic": 4}
reverse_label_map = {v: k for k, v in label_map.items()}
NUM_CLASSES = len(label_map)

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3: Data Loading & Bayesian Priors
# ═══════════════════════════════════════════════════════════════════════════════
print("Loading data...")
train = pd.read_csv(f"{CFG.base_path}train.csv")
val   = pd.read_csv(f"{CFG.base_path}validation.csv")
test  = pd.read_csv(f"{CFG.base_path}test.csv")

train = pd.concat([train, val]).reset_index(drop=True)

# ── Bayesian Priors ──
category_priors = {}
for cat in train['category'].unique():
    subset = train[train['category'] == cat]
    counts = subset['label'].value_counts()
    probs = np.zeros(NUM_CLASSES)
    for lbl, count in counts.items():
        probs[label_map[lbl]] = count / len(subset)
    category_priors[cat] = probs

def build_text(row):
    cat = row["category"]
    ctx = str(row["context"])
    words = len(ctx.split())
    if words < 8: tier = "সংক্ষিপ্ত"
    elif words <= 25: tier = "মাঝারি"
    else: tier = "বিস্তারিত"
    
    return f"{cat} । দৈর্ঘ্য: {tier} । {ctx}"

train["text"] = train.apply(build_text, axis=1)
test["text"]  = test.apply(build_text, axis=1)
train["label_id"] = train["label"].map(label_map)

skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=42)
train["fold"] = -1
for fold, (_, vi) in enumerate(skf.split(train, train["label_id"])):
    train.loc[vi, "fold"] = fold

counts = np.bincount(train["label_id"].values, minlength=NUM_CLASSES).astype(float)
CLASS_WEIGHTS = torch.tensor(len(train) / (NUM_CLASSES * counts), dtype=torch.float32)

tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)
def tokenize_fn(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=CFG.max_len)

tokenized_test = Dataset.from_pandas(test[["text"]]).map(tokenize_fn, batched=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4: Training Utilities (FGM, EMA, LLRD, Focal Loss)
# ═══════════════════════════════════════════════════════════════════════════════
class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}
    def attack(self, epsilon=0.5, emb_name="embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    param.data.add_(epsilon * param.grad / norm)
    def restore(self, emb_name="embeddings"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

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
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

def get_llrd_params(model, base_lr, weight_decay=0.01, decay=0.90):
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    n_layers = getattr(model.config, "num_hidden_layers", 12)
    def depth(name):
        if any(h in name for h in ("classifier", "pooler", "head")): return n_layers + 1
        if "embeddings" in name: return 0
        m = re.search(r"\.layer\.(\d+)\.", name) or re.search(r"\.blocks\.(\d+)\.", name)
        return int(m.group(1)) + 1 if m else n_layers
    groups = defaultdict(lambda: {"decay": [], "no_decay": []})
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        d = depth(name)
        key = "no_decay" if any(nd in name for nd in no_decay) else "decay"
        groups[d][key].append(param)
    param_groups = []
    max_depth = n_layers + 1
    for d, ps in groups.items():
        lr = base_lr * (decay ** (max_depth - d))
        if ps["decay"]: param_groups.append({"params": ps["decay"], "lr": lr, "weight_decay": weight_decay})
        if ps["no_decay"]: param_groups.append({"params": ps["no_decay"], "lr": lr, "weight_decay": 0.0})
    return param_groups

def focal_ce_loss(logits, labels, class_weights, label_smoothing=0.10, gamma=2.0):
    wt = class_weights.to(logits.device)
    ce = nn.CrossEntropyLoss(weight=wt, label_smoothing=label_smoothing, reduction="none")(logits, labels)
    if gamma == 0: return ce.mean()
    pt = torch.exp(-ce)
    focal = ((1 - pt) ** gamma) * ce
    return focal.mean()

def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    _, _, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

class AdvancedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ema = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        if model.training and CFG.use_rdrop and not getattr(self, "_fgm_mode", False):
            out1 = model(**inputs)
            out2 = model(**inputs)
            ce = (focal_ce_loss(out1.logits, labels, CLASS_WEIGHTS, CFG.label_smooth, CFG.focal_gamma) +
                  focal_ce_loss(out2.logits, labels, CLASS_WEIGHTS, CFG.label_smooth, CFG.focal_gamma)) / 2
            p1 = F.softmax(out1.logits, dim=-1)
            p2 = F.softmax(out2.logits, dim=-1)
            kl = (F.kl_div(out1.logits.log_softmax(-1), p2, reduction="batchmean") +
                  F.kl_div(out2.logits.log_softmax(-1), p1, reduction="batchmean")) / 2
            loss = ce + CFG.rdrop_alpha * kl
            return (loss, out1) if return_outputs else loss
        else:
            out = model(**inputs)
            loss = focal_ce_loss(out.logits, labels, CLASS_WEIGHTS, CFG.label_smooth, CFG.focal_gamma)
            return (loss, out) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None, **kwargs):
        if CFG.use_ema and self._ema is None:
            self._ema = EMA(self.model, decay=CFG.ema_decay)
        model.train()
        inputs = self._prepare_inputs(inputs)
        labels = inputs["labels"].clone()
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        self.accelerator.backward(loss)

        if CFG.use_fgm:
            fgm = FGM(model)
            fgm.attack(epsilon=CFG.fgm_epsilon)
            inputs["labels"] = labels
            self._fgm_mode = True
            with self.compute_loss_context_manager():
                loss_adv = self.compute_loss(model, inputs)
            self._fgm_mode = False
            if self.args.gradient_accumulation_steps > 1:
                loss_adv = loss_adv / self.args.gradient_accumulation_steps
            self.accelerator.backward(loss_adv)
            fgm.restore()

        if CFG.use_ema and self._ema is not None:
            self._ema.update()
        return loss.detach()

    def create_optimizer(self):
        if self.optimizer is None and CFG.use_llrd:
            params = get_llrd_params(self.model, self.args.learning_rate, self.args.weight_decay, CFG.llrd_decay)
            self.optimizer = torch.optim.AdamW(params, eps=1e-6, betas=(0.9, 0.999))
        elif self.optimizer is None:
            self.optimizer = super().create_optimizer()
        return self.optimizer

    def evaluate(self, *args, **kwargs):
        if CFG.use_ema and self._ema is not None: self._ema.apply_shadow()
        res = super().evaluate(*args, **kwargs)
        if CFG.use_ema and self._ema is not None: self._ema.restore()
        return res

    def predict(self, *args, **kwargs):
        if CFG.use_ema and self._ema is not None: self._ema.apply_shadow()
        res = super().predict(*args, **kwargs)
        if CFG.use_ema and self._ema is not None: self._ema.restore()
        return res

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5: 5-Fold Training
# ═══════════════════════════════════════════════════════════════════════════════
cv_f1s = []
fold_logits = []

for fold in range(CFG.n_folds):
    print(f"\n{'='*20} FOLD {fold+1}/{CFG.n_folds} {'='*20}")
    trn_df = train[train["fold"] != fold].reset_index(drop=True)
    val_df = train[train["fold"] == fold].reset_index(drop=True)

    tok_trn = Dataset.from_pandas(trn_df[["text", "label_id"]].rename(columns={"label_id": "label"})).map(tokenize_fn, batched=True)
    tok_val = Dataset.from_pandas(val_df[["text", "label_id"]].rename(columns={"label_id": "label"})).map(tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(CFG.model_name, num_labels=NUM_CLASSES)

    args = TrainingArguments(
        output_dir                  = f"/kaggle/working/fold_{fold}",
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        learning_rate               = CFG.lr,
        per_device_train_batch_size = CFG.batch_size,
        per_device_eval_batch_size  = CFG.batch_size,
        num_train_epochs            = CFG.epochs,
        warmup_ratio                = 0.10,
        lr_scheduler_type           = "cosine",
        weight_decay                = 0.01,
        fp16                        = CFG.fp16,
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        greater_is_better           = True,
        report_to                   = "none",
        save_total_limit            = 1,
    )

    trainer = AdvancedTrainer(model=model, args=args, train_dataset=tok_trn, eval_dataset=tok_val, compute_metrics=compute_metrics)
    trainer.train()

    best_f1 = max(trainer.state.log_history, key=lambda x: x.get("eval_f1", -1)).get("eval_f1", 0)
    cv_f1s.append(best_f1)
    
    print(f"Fold {fold+1} F1: {best_f1:.4f}")
    fold_logits.append(trainer.predict(tokenized_test).predictions)
    
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

mean_f1 = np.mean(cv_f1s)
print(f"\nMean CV F1: {mean_f1:.4f}")

# Ensemble probabilities
test_probs = scipy_softmax(np.mean(fold_logits, axis=0), axis=-1)

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6: Soft Pseudo-Labeling
# ═══════════════════════════════════════════════════════════════════════════════
if CFG.use_pseudo:
    print("\n── Soft Pseudo-Labeling ──")
    max_probs = test_probs.max(axis=-1)
    confident = max_probs >= CFG.pseudo_threshold
    
    if confident.sum() > 50:
        pseudo_preds = np.argmax(test_probs, axis=-1)
        pseudo_df = test[confident].copy()
        pseudo_df["label_id"] = pseudo_preds[confident]
        
        full_train = pd.concat([train[["text", "label_id"]], pseudo_df[["text", "label_id"]]], ignore_index=True)
        tok_full = Dataset.from_pandas(full_train.rename(columns={"label_id": "label"})).map(tokenize_fn, batched=True)
        
        # Start from pre-trained weights to avoid overfitting to a specific fold
        pseudo_model = AutoModelForSequenceClassification.from_pretrained(CFG.model_name, num_labels=NUM_CLASSES)
        
        p_args = TrainingArguments(
            output_dir                  = "/kaggle/working/pseudo",
            num_train_epochs            = CFG.pseudo_epochs,
            per_device_train_batch_size = CFG.batch_size,
            learning_rate               = CFG.pseudo_lr,
            warmup_ratio                = 0.10,
            lr_scheduler_type           = "cosine",
            weight_decay                = 0.01,
            fp16                        = CFG.fp16,
            save_strategy               = "no",
            report_to                   = "none",
        )
        
        p_trainer = AdvancedTrainer(model=pseudo_model, args=p_args, train_dataset=tok_full)
        p_trainer.train()
        
        pseudo_logits = p_trainer.predict(tokenized_test).predictions
        pseudo_probs = scipy_softmax(pseudo_logits, axis=-1)
        
        test_probs = (1 - CFG.pseudo_blend) * test_probs + CFG.pseudo_blend * pseudo_probs
        
        del pseudo_model, p_trainer
        gc.collect()
        torch.cuda.empty_cache()

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7: Bayesian Prior Post-Processing & Hard Rules
# ═══════════════════════════════════════════════════════════════════════════════
print("\nApplying Bayesian Priors & Hard Rules...")
alpha = 0.20
final_probs = test_probs.copy()

for i, row in test.iterrows():
    cat = row["category"]
    prior = category_priors.get(cat, np.ones(NUM_CLASSES)/NUM_CLASSES)
    
    # Bayesian adjustment (Multiply Probability by Prior^Alpha)
    final_probs[i] = final_probs[i] * (prior ** alpha)
    
    # Re-normalize
    final_probs[i] = final_probs[i] / np.sum(final_probs[i])

final_preds = np.argmax(final_probs, axis=-1)

# Apply Hard Rules
for i, row in test.iterrows():
    cat = row["category"]
    pred_class = reverse_label_map[final_preds[i]]
    
    # 1. Non Disaster -> Minimal
    if cat == "Non Disaster":
        final_preds[i] = label_map["Minimal"]
    # 2. Tropical Storm + Catastrophic -> Severe
    elif cat == "Tropical Storm" and pred_class == "Catastrophic":
        final_preds[i] = label_map["Severe"]
    # 3. Flood + Minimal -> Mild
    elif cat == "Flood" and pred_class == "Minimal":
        final_preds[i] = label_map["Mild"]

test["label"] = [reverse_label_map[p] for p in final_preds]
submission = test[["image_id", "label"]]
submission.to_csv("submission.csv", index=False)

with zipfile.ZipFile("submission.zip", "w") as z:
    z.write("submission.csv", arcname="submission.csv")

print("\n 'submission.zip' is ready for upload.")
print("\nFinal prediction distribution:")
print(submission["label"].value_counts())
print()
print(submission.head(10))
