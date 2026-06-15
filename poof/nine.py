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
import shutil
import gc

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

# ── Configuration ──────────────────────────────────────────────────────────
class CFG:
    # 1. THE LARGE MODEL UPGRADE
    model_name = "csebuetnlp/banglabert_large"
    base_path  = "/kaggle/input/competitions/datathon-iiuc-cse-fest-2026/DisasterSeverity/"

    # 2. THE TRUNCATION FIX
    max_len = 256            # Landslides are long. 128 was truncating them.

    # 3. TRAINING & HARDWARE
    epochs       = 5
    batch_size   = 8         # Halved to fit banglabert_large on T4
    grad_accum   = 2         # Restores effective batch_size to 16
    lr           = 2e-5
    warmup_ratio = 0.10
    weight_decay = 0.01

    # 4. REGULARISATION
    label_smoothing = 0.10
    use_rdrop   = True
    rdrop_alpha = 0.50
    use_fgm     = True
    fgm_epsilon = 0.50

    n_folds = 5
    seed    = 42

label_map         = {"Minimal": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Catastrophic": 4}
reverse_label_map = {v: k for k, v in label_map.items()}

# ── Data Loading & The Prepend Hack ────────────────────────────────────────
print("Loading data...")
train = pd.read_csv(f"{CFG.base_path}train.csv")
test  = pd.read_csv(f"{CFG.base_path}test.csv")
val   = pd.read_csv(f"{CFG.base_path}validation.csv")

# Extract the Validation-Weighted Bayesian Priors BEFORE concatenation
cross_all = pd.crosstab(pd.concat([train, val], ignore_index=True)['category'], pd.concat([train, val], ignore_index=True)['label'].map(label_map), normalize='index')
cross_val = pd.crosstab(val['category'], val['label'].map(label_map), normalize='index')

historical_probs_dict = {}
for cat in cross_all.index:
    p_all = cross_all.loc[cat].reindex(range(5), fill_value=0).values
    p_val = cross_val.loc[cat].reindex(range(5), fill_value=0).values if cat in cross_val.index else p_all
    # 50% Train+Val Prior, 50% Val Prior (chronologically closer to test set)
    historical_probs_dict[cat] = 0.5 * p_all + 0.5 * p_val
historical_probs_dict['Non Disaster'] = np.array([1.00, 0.00, 0.00, 0.00, 0.00])

train = pd.concat([train, val]).reset_index(drop=True)

# THE LENGTH HACK: Adding Length Tier explicitly to the text string
def build_text(row):
    length = len(str(row['context']))
    tier = "short" if length < 60 else ("medium" if length < 130 else "long")
    return f"[{row['category']}] [{tier}] {row['context']}"

train["text"] = train.apply(build_text, axis=1)
test["text"]  = test.apply(build_text, axis=1)

train["label_id"] = train["label"].map(label_map)

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

# Pre-compute class weights
class_counts  = np.bincount(train["label_id"].values, minlength=5).astype(float)
CLASS_WEIGHTS = (len(train) / (5 * class_counts)).tolist()

# ── FGM + AdvancedTrainer ──────────────────────────────────────────────────
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

# ── 5-Fold Training ────────────────────────────────────────────────────────
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
        gradient_accumulation_steps = CFG.grad_accum,
        num_train_epochs            = CFG.epochs,
        warmup_ratio                = CFG.warmup_ratio,
        lr_scheduler_type           = "cosine",
        weight_decay                = CFG.weight_decay,
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1",
        greater_is_better           = True,
        report_to                   = "none",
        save_total_limit            = 1,
        fp16                        = True,  # Mixed precision to save memory on T4
        logging_steps               = 50
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
    
    # Cleanup memory
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()
    if os.path.exists(f"/kaggle/working/fold_{fold}"):
        shutil.rmtree(f"/kaggle/working/fold_{fold}")

# ── Grandmaster Bayesian Post-Processing ────────────────────────────────────
print("\nEnsembling 5-Fold Predictions & Applying Validation-Weighted Bayesian Priors...")
final_logits = np.mean(test_predictions, axis=0)
bayesian_logits = np.zeros_like(final_logits)
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

with zipfile.ZipFile('submission_unimodal_grandmaster.zip', 'w') as zipf:
    zipf.write('submission.csv', arcname='submission.csv')

print("\n🚀 Unimodal Grandmaster Pipeline Complete! 'submission_unimodal_grandmaster.zip' is ready for upload.")
