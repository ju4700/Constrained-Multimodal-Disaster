"""
targeted_fix_v4.py — The REAL fix.

Strategy: Use the EXACT original 0.784 model (which works!) and add ONLY the
hard rules from our EDA. No sqrt weights, no temperature calibration, no LLRD, 
no AWP. Those "improvements" over-regularized the model and killed pseudo-labeling.

Expected score: 0.79-0.81 (original 0.784 + free points from hard rules)
"""

# ── Cell 1: Imports & Seed ─────────────────────────────────────────────────
import os, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from scipy.special import softmax as scipy_softmax
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import zipfile

warnings.filterwarnings("ignore")

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

# ── Cell 2: Configuration ──────────────────────────────────────────────────
# EXACT SAME config as the original 0.784 model. DO NOT CHANGE.
class CFG:
    model_name = "csebuetnlp/banglabert"
    base_path  = "/kaggle/input/datathon-iiuc-cse-fest-2026/DisasterSeverity/"
    max_len = 128
    epochs       = 5
    batch_size   = 16
    lr           = 2e-5
    warmup_ratio = 0.10
    weight_decay = 0.01
    label_smoothing = 0.10
    use_rdrop   = True
    rdrop_alpha = 0.50
    use_fgm     = True
    fgm_epsilon = 0.50
    n_folds = 5
    seed    = 42
    use_pseudo       = True
    pseudo_threshold = 0.90
    pseudo_epochs    = 3
    pseudo_lr        = 1e-5

label_map         = {"Minimal": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Catastrophic": 4}
reverse_label_map = {v: k for k, v in label_map.items()}

# ── Cell 3: Data Loading ───────────────────────────────────────────────────
print("Loading data...")
train = pd.read_csv(f"{CFG.base_path}train.csv")
test  = pd.read_csv(f"{CFG.base_path}test.csv")
val   = pd.read_csv(f"{CFG.base_path}validation.csv")

train = pd.concat([train, val]).reset_index(drop=True)

# Category-aware text (same as original)
train["text"] = train["category"] + ": " + train["context"].fillna("")
test["text"]  = test["category"]  + ": " + test["context"].fillna("")

train["label_id"] = train["label"].map(label_map)

# Stratified K-Fold
skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
train["fold"] = -1
for fold, (_, val_idx) in enumerate(skf.split(train, train["label_id"])):
    train.loc[val_idx, "fold"] = fold

tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=CFG.max_len,
    )

test_dataset   = Dataset.from_pandas(test[["text"]])
tokenized_test = test_dataset.map(tokenize_fn, batched=True)

# ORIGINAL class weights — NOT sqrt. These make the model confident enough
# for pseudo-labeling to work.
class_counts  = np.bincount(train["label_id"].values, minlength=5).astype(float)
CLASS_WEIGHTS = (len(train) / (5 * class_counts)).tolist()
print("Class weights:", {k: round(CLASS_WEIGHTS[v], 3) for k, v in label_map.items()})
print(f"Train: {len(train)} samples | Test: {len(test)} samples")

# ── Cell 4: FGM + AdvancedTrainer (exact original) ─────────────────────────
class FGM:
    def __init__(self, model):
        self.model  = model
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


def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    _, _, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


class AdvancedTrainer(Trainer):
    def _ce(self, logits, labels):
        wt = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(logits.device)
        return nn.CrossEntropyLoss(weight=wt, label_smoothing=CFG.label_smoothing)(
            logits, labels
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")

        if model.training and CFG.use_rdrop and not getattr(self, "_fgm_mode", False):
            out1 = model(**inputs)
            out2 = model(**inputs)
            ce = (self._ce(out1.logits, labels) + self._ce(out2.logits, labels)) / 2
            p1  = F.softmax(out1.logits, dim=-1)
            p2  = F.softmax(out2.logits, dim=-1)
            kl  = (
                F.kl_div(out1.logits.log_softmax(-1), p2, reduction="batchmean") +
                F.kl_div(out2.logits.log_softmax(-1), p1, reduction="batchmean")
            ) / 2
            loss = ce + CFG.rdrop_alpha * kl
            return (loss, out1) if return_outputs else loss
        else:
            out  = model(**inputs)
            loss = self._ce(out.logits, labels)
            return (loss, out) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None, **kwargs):
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

        return loss.detach()


# ── Cell 5: 5-Fold Training ────────────────────────────────────────────────
test_predictions = []

for fold in range(CFG.n_folds):
    print(f"\n{'='*22} FOLD {fold+1}/{CFG.n_folds} {'='*22}")

    trn_df = train[train["fold"] != fold].reset_index(drop=True)
    val_df = train[train["fold"] == fold].reset_index(drop=True)

    tok_trn = (
        Dataset.from_pandas(trn_df[["text", "label_id"]].rename(columns={"label_id": "label"}))
        .map(tokenize_fn, batched=True)
    )
    tok_val = (
        Dataset.from_pandas(val_df[["text", "label_id"]].rename(columns={"label_id": "label"}))
        .map(tokenize_fn, batched=True)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        CFG.model_name, num_labels=5
    )

    args = TrainingArguments(
        output_dir                  = f"/kaggle/working/fold_{fold}",
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        learning_rate               = CFG.lr,
        per_device_train_batch_size = CFG.batch_size,
        per_device_eval_batch_size  = CFG.batch_size,
        num_train_epochs            = CFG.epochs,
        warmup_ratio                = CFG.warmup_ratio,
        lr_scheduler_type           = "cosine",
        weight_decay                = CFG.weight_decay,
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        greater_is_better           = True,
        report_to                   = "none",
        save_total_limit            = 1,
    )

    trainer = AdvancedTrainer(
        model           = model,
        args            = args,
        train_dataset   = tok_trn,
        eval_dataset    = tok_val,
        compute_metrics = compute_metrics,
    )

    trainer.train()
    print(f"Predicting test set for fold {fold + 1}...")
    preds = trainer.predict(tokenized_test).predictions
    test_predictions.append(preds)

# ── Cell 6: Ensemble + Hard Rules ──────────────────────────────────────────
print("\nEnsembling 5-fold predictions...")
final_logits = np.mean(test_predictions, axis=0)
final_preds  = np.argmax(final_logits, axis=-1)

test["label"] = [reverse_label_map[p] for p in final_preds]

# ── HARD RULES (the ONLY new thing vs original 0.784) ─────────────────────
# Rule 1: Non Disaster → Minimal (100% in training data, same as original)
test.loc[test["category"] == "Non Disaster", "label"] = "Minimal"

# Rule 2: Tropical Storm + Catastrophic → Severe (only 0.4% in train = 2/450)
test.loc[
    (test["category"] == "Tropical Storm") & (test["label"] == "Catastrophic"),
    "label"
] = "Severe"

# Rule 3: Human Damage + Catastrophic → Severe (only 1.1% in train = 5/450)
test.loc[
    (test["category"] == "Human Damage") & (test["label"] == "Catastrophic"),
    "label"
] = "Severe"

# Rule 4: Flood + Minimal → Mild (only 0.2% in train = 1/450)
test.loc[
    (test["category"] == "Flood") & (test["label"] == "Minimal"),
    "label"
] = "Mild"

print("Hard rules applied.")
print("\nPre-pseudo prediction distribution:")
print(test["label"].value_counts())

# ── Cell 7: Pseudo-Labelling (exact original logic) ────────────────────────
if CFG.use_pseudo:
    print("\n── Pseudo-Labelling ──")
    probs     = scipy_softmax(final_logits, axis=-1)
    max_probs = probs.max(axis=-1)

    confident = max_probs >= CFG.pseudo_threshold
    print(f"High-confidence test samples: {confident.sum()}/{len(test)} "
          f"(threshold={CFG.pseudo_threshold})")

    if confident.sum() > 50:
        pseudo_df = test[confident].copy()
        pseudo_df["label_id"] = final_preds[confident]

        full_train = pd.concat(
            [train[["text", "label_id"]], pseudo_df[["text", "label_id"]]],
            ignore_index=True,
        )
        print(f"Expanded train size: {len(train)} → {len(full_train)} samples")

        tok_full = (
            Dataset.from_pandas(full_train.rename(columns={"label_id": "label"}))
            .map(tokenize_fn, batched=True)
        )

        pseudo_model = AutoModelForSequenceClassification.from_pretrained(
            CFG.model_name, num_labels=5
        )
        pseudo_args = TrainingArguments(
            output_dir                  = "/kaggle/working/pseudo",
            num_train_epochs            = CFG.pseudo_epochs,
            per_device_train_batch_size = CFG.batch_size,
            per_device_eval_batch_size  = CFG.batch_size,
            learning_rate               = CFG.pseudo_lr,
            warmup_ratio                = CFG.warmup_ratio,
            lr_scheduler_type           = "cosine",
            weight_decay                = CFG.weight_decay,
            save_strategy               = "no",
            report_to                   = "none",
        )
        pseudo_trainer = AdvancedTrainer(
            model           = pseudo_model,
            args            = pseudo_args,
            train_dataset   = tok_full,
            compute_metrics = compute_metrics,
        )
        pseudo_trainer.train()

        pseudo_logits = pseudo_trainer.predict(tokenized_test).predictions

        # Blend: 70% CV ensemble + 30% pseudo model (exact original)
        blended      = 0.70 * final_logits + 0.30 * pseudo_logits
        final_preds  = np.argmax(blended, axis=-1)
        test["label"] = [reverse_label_map[p] for p in final_preds]

        # ── Re-apply ALL hard rules after pseudo-blending ──────────────
        test.loc[test["category"] == "Non Disaster", "label"] = "Minimal"
        test.loc[
            (test["category"] == "Tropical Storm") & (test["label"] == "Catastrophic"),
            "label"
        ] = "Severe"
        test.loc[
            (test["category"] == "Human Damage") & (test["label"] == "Catastrophic"),
            "label"
        ] = "Severe"
        test.loc[
            (test["category"] == "Flood") & (test["label"] == "Minimal"),
            "label"
        ] = "Mild"

        print("Blended predictions updated (70% CV + 30% pseudo).")
        print("Hard rules re-applied after blending.")
    else:
        print("Too few high-confidence samples; skipping pseudo-labelling.")

# ── Cell 8: Save Submission ────────────────────────────────────────────────
print("\nFinal prediction distribution:")
print(test["label"].value_counts())

submission = test[["image_id", "label"]]
submission.to_csv("submission.csv", index=False)

with zipfile.ZipFile("submission.zip", "w") as z:
    z.write("submission.csv", arcname="submission.csv")

print("\n✅ submission.zip ready.")
print("\nSample predictions:")
print(submission.head(10))
