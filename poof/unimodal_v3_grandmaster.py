import os, re, random, warnings, zipfile, gc
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from collections import defaultdict
from scipy.special import softmax as scipy_softmax
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, set_seed,
)
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

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

BASE_PATH = "/kaggle/input/competitions/datathon-iiuc-cse-fest-2026/DisasterSeverity/"

SHARED = dict(
    max_len         = 256,     # PATCH 1: Truncation Fix
    n_folds         = 5,
    seed            = 42,
    weight_decay    = 0.01,
    warmup_ratio    = 0.10,
    label_smoothing = 0.10,
    use_rdrop       = True,
    rdrop_alpha     = 0.50,
    use_awp         = True,
    awp_lr          = 1e-4,
    awp_eps         = 1e-3,
    use_llrd        = True,
    llrd_decay      = 0.90,
    focal_gamma     = 2.0,
    pseudo_threshold = 0.85,
    pseudo_epochs    = 3,
    pseudo_lr        = 8e-6,
    pseudo_blend     = 0.35,
    bayesian_alpha   = 0.20    # PATCH 4: Bayesian Prior
)

MODEL_CFGS = [
    dict(
        key        = "banglabert",
        model_name = "csebuetnlp/banglabert",
        epochs     = 5,
        lr         = 2e-5,
        batch      = 16,
        grad_acc   = 1,
        fp16       = True,
        grad_ckpt  = False,
    ),
    dict(
        key        = "xlmr_large",
        model_name = "xlm-roberta-large",
        epochs     = 6,      # PATCH 3: More epochs
        lr         = 8e-6,   # PATCH 3: Lower LR
        batch      = 4,      # OOM FIX: Prevent T4 crash
        grad_acc   = 4,      # OOM FIX: Keep effective batch 16
        fp16       = True,
        grad_ckpt  = True,
    ),
]

label_map         = {"Minimal": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Catastrophic": 4}
reverse_label_map = {v: k for k, v in label_map.items()}
NUM_LABELS        = 5

print("Loading data...")
train = pd.read_csv(f"{BASE_PATH}train.csv")
test  = pd.read_csv(f"{BASE_PATH}test.csv")
val   = pd.read_csv(f"{BASE_PATH}validation.csv")

# Extract Bayesian Prior before concatenation
cross_all = pd.crosstab(pd.concat([train, val], ignore_index=True)['category'], pd.concat([train, val], ignore_index=True)['label'].map(label_map), normalize='index')
cross_val = pd.crosstab(val['category'], val['label'].map(label_map), normalize='index')

historical_probs_dict = {}
for cat in cross_all.index:
    p_all = cross_all.loc[cat].reindex(range(5), fill_value=0).values
    p_val = cross_val.loc[cat].reindex(range(5), fill_value=0).values if cat in cross_val.index else p_all
    historical_probs_dict[cat] = 0.5 * p_all + 0.5 * p_val
historical_probs_dict['Non Disaster'] = np.array([1.00, 0.00, 0.00, 0.00, 0.00])

train = pd.concat([train, val]).reset_index(drop=True)

# PATCH 2: THE LENGTH TIER HACK (Bengali Word Count)
def build_text(row):
    words = len(str(row['context']).split())
    tier = "brief" if words < 8 else ("medium" if words <= 25 else "detailed")
    return f"[{row['category']}] [{tier}] {row['context']}"

train["text"] = train.apply(build_text, axis=1)
test["text"]  = test.apply(build_text, axis=1)

train["label_id"] = train["label"].map(label_map)

skf = StratifiedKFold(n_splits=SHARED["n_folds"], shuffle=True, random_state=SHARED["seed"])
train["fold"] = -1
for fold, (_, vi) in enumerate(skf.split(train, train["label_id"])):
    train.loc[vi, "fold"] = fold

counts        = np.bincount(train["label_id"].values, minlength=NUM_LABELS).astype(float)
CLASS_WEIGHTS = (len(train) / (NUM_LABELS * counts)).tolist()

class AWP:
    SKIP = ("bias", "LayerNorm", "layer_norm", "classifier", "pooler")
    def __init__(self, model):
        self.model  = model
        self.backup = {}
    def attack(self, adv_lr=1e-4, adv_eps=1e-3):
        for name, param in self.model.named_parameters():
            if not param.requires_grad or param.grad is None: continue
            if any(s in name for s in self.SKIP): continue
            self.backup[name] = param.data.clone()
            norm = torch.norm(param.grad)
            if norm != 0 and not torch.isnan(norm):
                r = adv_lr * param.grad / norm
                param.data.add_(r)
                param.data = torch.clamp(param.data, self.backup[name] - adv_eps, self.backup[name] + adv_eps)
    def restore(self):
        for name, param in self.model.named_parameters():
            if name in self.backup: param.data = self.backup[name]
        self.backup = {}

def get_llrd_params(model, base_lr, weight_decay=0.01, decay=0.90):
    no_decay = ("bias", "LayerNorm.weight", "LayerNorm.bias", "layer_norm.weight", "layer_norm.bias")
    n_layers = getattr(model.config, "num_hidden_layers", 12)
    def depth(name):
        if any(h in name for h in ("classifier", "pooler")): return n_layers + 1
        if "embeddings" in name: return 0
        m = re.search(r"\.layer\.(\d+)\.", name)
        return int(m.group(1)) + 1 if m else n_layers
    groups = defaultdict(lambda: {"decay": [], "no_decay": []})
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        d, key = depth(name), "no_decay" if any(nd in name for nd in no_decay) else "decay"
        groups[d][key].append(param)
    param_groups = []
    max_depth = n_layers + 1
    for d, ps in groups.items():
        lr = base_lr * (decay ** (max_depth - d))
        if ps["decay"]: param_groups.append({"params": ps["decay"], "lr": lr, "weight_decay": weight_decay})
        if ps["no_decay"]: param_groups.append({"params": ps["no_decay"], "lr": lr, "weight_decay": 0.0})
    return param_groups

def focal_ce_loss(logits, labels, class_weights, label_smoothing=0.1, gamma=2.0):
    wt = torch.tensor(class_weights, dtype=torch.float32).to(logits.device)
    ce = nn.CrossEntropyLoss(weight=wt, label_smoothing=label_smoothing, reduction="none")(logits, labels)
    if gamma == 0: return ce.mean()
    pt = torch.exp(-ce)
    focal = ((1 - pt) ** gamma) * ce
    return focal.mean()

def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    _, _, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1}

class AdvancedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        sh = SHARED
        is_rdrop = (model.training and sh["use_rdrop"] and not getattr(self, "_awp_mode", False))
        if is_rdrop:
            out1, out2 = model(**inputs), model(**inputs)
            ce = (focal_ce_loss(out1.logits, labels, CLASS_WEIGHTS, sh["label_smoothing"], sh["focal_gamma"]) +
                  focal_ce_loss(out2.logits, labels, CLASS_WEIGHTS, sh["label_smoothing"], sh["focal_gamma"])) / 2
            p1, p2 = F.softmax(out1.logits, -1), F.softmax(out2.logits, -1)
            kl = (F.kl_div(out1.logits.log_softmax(-1), p2, reduction="batchmean") +
                  F.kl_div(out2.logits.log_softmax(-1), p1, reduction="batchmean")) / 2
            loss = ce + sh["rdrop_alpha"] * kl
            return (loss, out1) if return_outputs else loss
        else:
            out = model(**inputs)
            loss = focal_ce_loss(out.logits, labels, CLASS_WEIGHTS, sh["label_smoothing"], sh["focal_gamma"])
            return (loss, out) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None, **kwargs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        labels = inputs["labels"].clone()
        with self.compute_loss_context_manager(): loss = self.compute_loss(model, inputs)
        scale = self.args.gradient_accumulation_steps
        self.accelerator.backward(loss / scale if scale > 1 else loss)
        if SHARED["use_awp"]:
            awp = AWP(model)
            awp.attack(adv_lr=SHARED["awp_lr"], adv_eps=SHARED["awp_eps"])
            inputs["labels"] = labels
            self._awp_mode = True
            with self.compute_loss_context_manager(): loss_adv = self.compute_loss(model, inputs)
            self._awp_mode = False
            self.accelerator.backward(loss_adv / scale if scale > 1 else loss_adv)
            awp.restore()
        return loss.detach()

    def create_optimizer(self):
        if SHARED["use_llrd"]:
            grouped = get_llrd_params(self.model, self.args.learning_rate, self.args.weight_decay, SHARED["llrd_decay"])
            self.optimizer = torch.optim.AdamW(grouped, eps=1e-6, betas=(0.9, 0.999))
        else: self.optimizer = super().create_optimizer()
        return self.optimizer

def train_model(model_cfg, train_df, test_df):
    key = model_cfg["key"]
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_name"])
    def tok_fn(examples): return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=SHARED["max_len"])
    tok_test = Dataset.from_pandas(test_df[["text"]]).map(tok_fn, batched=True)
    
    fold_preds = []
    cv_f1s = []

    for fold in range(SHARED["n_folds"]):
        print(f"\n{'='*18} [{key.upper()}] FOLD {fold+1}/{SHARED['n_folds']} {'='*18}")
        trn_df = train_df[train_df["fold"] != fold].reset_index(drop=True)
        val_df = train_df[train_df["fold"] == fold].reset_index(drop=True)

        tok_trn = Dataset.from_pandas(trn_df[["text", "label_id"]].rename(columns={"label_id": "label"})).map(tok_fn, batched=True)
        tok_val = Dataset.from_pandas(val_df[["text", "label_id"]].rename(columns={"label_id": "label"})).map(tok_fn, batched=True)

        model = AutoModelForSequenceClassification.from_pretrained(model_cfg["model_name"], num_labels=NUM_LABELS)
        if model_cfg["grad_ckpt"]: model.gradient_checkpointing_enable()

        args = TrainingArguments(
            output_dir                  = f"/kaggle/working/{key}_fold{fold}",
            eval_strategy               = "epoch",
            save_strategy               = "epoch",
            learning_rate               = model_cfg["lr"],
            per_device_train_batch_size = model_cfg["batch"],
            per_device_eval_batch_size  = model_cfg["batch"],
            gradient_accumulation_steps = model_cfg["grad_acc"],
            num_train_epochs            = model_cfg["epochs"],
            warmup_ratio                = SHARED["warmup_ratio"],
            lr_scheduler_type           = "cosine",
            weight_decay                = SHARED["weight_decay"],
            fp16                        = model_cfg["fp16"],
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
        fold_preds.append(trainer.predict(tok_test).predictions)
        
        del model, trainer
        gc.collect()
        torch.cuda.empty_cache()

    mean_f1 = np.mean(cv_f1s)
    return fold_preds, mean_f1

all_logits = {}
all_cv_f1 = {}

for mcfg in MODEL_CFGS:
    seed_everything(SHARED["seed"])
    preds, cv_f1 = train_model(mcfg, train, test)
    all_logits[mcfg["key"]] = np.array(preds)
    all_cv_f1[mcfg["key"]] = cv_f1

total_score = sum(all_cv_f1.values())
weights = {k: v / total_score for k, v in all_cv_f1.items()}
ensemble_logits = sum(weights[k] * all_logits[k].mean(axis=0) for k in all_logits)
final_preds = np.argmax(ensemble_logits, axis=-1)

# ═══════════════════════════════════════════════════════════════════════════
# Soft Pseudo-labelling
# ═══════════════════════════════════════════════════════════════════════════
print("\n── Soft Pseudo-Labelling ──")
probs = scipy_softmax(ensemble_logits, axis=-1)
max_probs = probs.max(axis=-1)
confident = max_probs >= SHARED["pseudo_threshold"]

if confident.sum() > 50:
    pseudo_df = test[confident].copy()
    pseudo_df["label_id"] = final_preds[confident]
    full_train = pd.concat([train[["text", "label_id"]], pseudo_df[["text", "label_id"]]], ignore_index=True)

    class SoftPseudoTrainer(AdvancedTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            out = model(**inputs)
            loss = focal_ce_loss(out.logits, labels, CLASS_WEIGHTS, SHARED["label_smoothing"], SHARED["focal_gamma"])
            return (loss, out) if return_outputs else loss

    best_key = max(all_cv_f1, key=all_cv_f1.get)
    best_cfg = next(c for c in MODEL_CFGS if c["key"] == best_key)
    tokenizer_p = AutoTokenizer.from_pretrained(best_cfg["model_name"])
    def tok_p(examples): return tokenizer_p(examples["text"], padding="max_length", truncation=True, max_length=SHARED["max_len"])
    
    tok_full = Dataset.from_pandas(full_train.rename(columns={"label_id": "label"})).map(tok_p, batched=True)
    tok_tst  = Dataset.from_pandas(test[["text"]]).map(tok_p, batched=True)

    pseudo_model = AutoModelForSequenceClassification.from_pretrained(best_cfg["model_name"], num_labels=NUM_LABELS)
    if best_cfg["grad_ckpt"]: pseudo_model.gradient_checkpointing_enable()

    p_args = TrainingArguments(
        output_dir                  = "/kaggle/working/pseudo",
        num_train_epochs            = SHARED["pseudo_epochs"],
        per_device_train_batch_size = best_cfg["batch"],
        per_device_eval_batch_size  = best_cfg["batch"],
        gradient_accumulation_steps = best_cfg["grad_acc"],
        learning_rate               = SHARED["pseudo_lr"],
        warmup_ratio                = SHARED["warmup_ratio"],
        lr_scheduler_type           = "cosine",
        weight_decay                = SHARED["weight_decay"],
        fp16                        = best_cfg["fp16"],
        save_strategy               = "no",
        report_to                   = "none",
    )
    
    pseudo_trainer = SoftPseudoTrainer(model=pseudo_model, args=p_args, train_dataset=tok_full, compute_metrics=compute_metrics)
    pseudo_trainer.train()
    pseudo_logits = pseudo_trainer.predict(tok_tst).predictions

    pw = SHARED["pseudo_blend"]
    blended_logits = (1 - pw) * ensemble_logits + pw * pseudo_logits
else:
    print("Too few confident samples; skipping pseudo-labelling.")
    blended_logits = ensemble_logits

# ═══════════════════════════════════════════════════════════════════════════
# PATCH 4: Bayesian Prior Post-Processing
# ═══════════════════════════════════════════════════════════════════════════
test.reset_index(drop=True, inplace=True)
final_logits = np.zeros_like(blended_logits)
alpha = SHARED["bayesian_alpha"]

blended_probs = scipy_softmax(blended_logits, axis=-1)

for i, row in test.iterrows():
    category = row['category']
    prior_probs = historical_probs_dict.get(category, np.array([0.2, 0.2, 0.2, 0.2, 0.2]))
    p_blend = blended_probs[i] + 1e-6
    p_prior = prior_probs + 1e-6
    log_final = np.log(p_blend) + (alpha * np.log(p_prior))
    final_logits[i] = log_final

final_preds = np.argmax(final_logits, axis=-1)
test["label"] = [reverse_label_map[p] for p in final_preds]
test.loc[test["category"] == "Non Disaster", "label"] = "Minimal"

submission = test[["image_id", "label"]]
submission.to_csv("submission.csv", index=False)
with zipfile.ZipFile("submission_unimodal_v3_grandmaster.zip", "w") as z:
    z.write("submission.csv", arcname="submission.csv")

print("\n✅ Unimodal v3 Grandmaster Pipeline Complete!")
