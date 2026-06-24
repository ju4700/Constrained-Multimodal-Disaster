import os, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from scipy.special import softmax as scipy_softmax
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from datasets import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
import lightgbm as lgb
import optuna
import zipfile
import gc

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")

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

class CFG:
    base_path = "/kaggle/input/competitions/datathon-iiuc-cse-fest-2026/DisasterSeverity/"
    
    model_configs = [
        {"tag": "bbert",   "name": "csebuetnlp/banglabert",       "max_len": 128, "batch_size": 16, "grad_accum": 1},
        {"tag": "xlmr_b",  "name": "xlm-roberta-base",            "max_len": 128, "batch_size": 16, "grad_accum": 1},
    ]

    fp16 = True
    epochs       = 5
    lr           = 2e-5
    warmup_ratio = 0.10
    weight_decay = 0.01
    label_smoothing = 0.10
    n_folds      = 5
    seed         = 42

    use_rdrop    = True
    rdrop_alpha  = 0.50

    use_fgm      = True
    fgm_epsilon  = 1.0

    use_pseudo          = True
    dips_conf_threshold = 0.85
    dips_var_threshold  = 0.15

train = pd.read_csv(CFG.base_path + "train.csv")
test  = pd.read_csv(CFG.base_path + "test.csv")
val   = pd.read_csv(CFG.base_path + "validation.csv")

train = pd.concat([train, val]).reset_index(drop=True)

train["text"] = train["category"] + ": " + train["context"].fillna("")
test["text"]  = test["category"]  + ": " + test["context"].fillna("")

label_map = {"Minimal": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Catastrophic": 4}
reverse_label_map = {v: k for k, v in label_map.items()}
train["label_id"] = train["label"].map(label_map)
N_CLASSES = 5

skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
train["fold"] = -1
for fold, (_, val_idx) in enumerate(skf.split(train, train["label_id"])):
    train.loc[val_idx, "fold"] = fold

class_counts = np.bincount(train["label_id"].values)
CLASS_WEIGHTS = len(train) / (N_CLASSES * class_counts)

class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}
    def attack(self, epsilon=1.0, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and param.grad is not None:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)
    def restore(self, emb_name='word_embeddings'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

class SafeMoEGating(nn.Module):
    def __init__(self, hidden_size, num_experts=3, k=2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)
        
        router_logits = self.router(x_flat)
        gate_probs = F.softmax(router_logits, dim=-1)
        
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        final_output = torch.zeros_like(x_flat)
        
        for i in range(self.k):
            expert_indices = top_k_indices[:, i]
            expert_weights = top_k_probs[:, i].unsqueeze(-1)
            
            for expert_id, expert in enumerate(self.experts):
                token_mask = (expert_indices == expert_id)
                if token_mask.any():
                    routed_tokens = x_flat[token_mask]
                    expert_out = expert(routed_tokens)
                    final_output[token_mask] += expert_out * expert_weights[token_mask]
                    
        # Shazeer Load-Balancing Auxiliary Loss
        importance = gate_probs.mean(dim=0)
        load_mask = F.one_hot(top_k_indices, num_classes=self.num_experts).float()
        load = load_mask.sum(dim=1).mean(dim=0) 
        aux_loss = self.num_experts * torch.sum(importance * load)

        return final_output.view(batch_size, seq_len, d_model), aux_loss

class TextMoEModel(nn.Module):
    def __init__(self, model_name, num_labels=5):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.moe = SafeMoEGating(hidden_size, num_experts=3, k=2)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.config = self.backbone.config
        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        moe_output, aux_loss = self.moe(sequence_output)
        
        cls_output = moe_output[:, 0, :]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        
        output = SequenceClassifierOutput(
            loss=None,
            logits=logits,
        )
        output.aux_loss = aux_loss
        return output

class DIPSTrainerCallback(TrainerCallback):
    def __init__(self, collector: list, tokenized_test):
        self.collector      = collector
        self.tokenized_test = tokenized_test
        self.trainer        = None
    def on_epoch_end(self, args, state, control, **kwargs):
        if self.trainer is None: return
        print(f"\\n[DIPS] Extracting test probabilities at epoch {state.epoch:.0f}")
        raw = self.trainer.predict(self.tokenized_test).predictions
        self.collector.append(scipy_softmax(raw, axis=-1))

def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    _, _, f1_w, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_w}

class AdvancedTrainer(Trainer):
    def _ce(self, logits, labels):
        wt = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(logits.device)
        return nn.CrossEntropyLoss(weight=wt, label_smoothing=CFG.label_smoothing)(logits, labels)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        if model.training and CFG.use_rdrop and not getattr(self, "_fgm_mode", False):
            out1 = model(**inputs)
            out2 = model(**inputs)
            ce   = (self._ce(out1.logits, labels) + self._ce(out2.logits, labels)) / 2
            p1, p2 = F.softmax(out1.logits, dim=-1), F.softmax(out2.logits, dim=-1)
            kl   = (F.kl_div(out1.logits.log_softmax(-1), p2, reduction="batchmean") +
                    F.kl_div(out2.logits.log_softmax(-1), p1, reduction="batchmean")) / 2
            aux = (out1.aux_loss + out2.aux_loss) / 2
            loss = ce + CFG.rdrop_alpha * kl + 0.1 * aux
            return (loss, out1) if return_outputs else loss
        out  = model(**inputs)
        loss = self._ce(out.logits, labels) + 0.1 * out.aux_loss
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

def make_tok_fn(tokenizer, max_len):
    def tokenize_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_len)
    return tokenize_fn

oof_logits_per_model  = {}
test_logits_per_model = {}
dips_snapshots        = {}

for mc in CFG.model_configs:
    tag, model_name = mc["tag"], mc["name"]
    batch_size = mc.get("batch_size", 16)
    grad_accum = mc.get("grad_accum",  1)
    max_len    = mc.get("max_len",   128)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenize  = make_tok_fn(tokenizer, max_len)
    tok_test = Dataset.from_pandas(test[["text"]]).map(tokenize, batched=True)

    oof_logits = np.zeros((len(train), N_CLASSES), dtype=np.float32)
    fold_test_logits = []
    model_dips = []

    for fold in range(CFG.n_folds):
        trn_df = train[train["fold"] != fold].reset_index(drop=True)
        val_df = train[train["fold"] == fold].reset_index(drop=True)

        tok_trn = Dataset.from_pandas(trn_df[["text", "label_id"]].rename(columns={"label_id": "label"})).map(tokenize, batched=True)
        tok_val = Dataset.from_pandas(val_df[["text", "label_id"]].rename(columns={"label_id": "label"})).map(tokenize, batched=True)

        model = TextMoEModel(model_name, num_labels=N_CLASSES)

        args = TrainingArguments(
            output_dir                  = f"/kaggle/working/{tag}_fold{fold}",
            eval_strategy               = "epoch",
            save_strategy               = "epoch",
            learning_rate               = CFG.lr,
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size  = batch_size,
            gradient_accumulation_steps = grad_accum,
            num_train_epochs            = CFG.epochs,
            warmup_ratio                = CFG.warmup_ratio,
            lr_scheduler_type           = "cosine",
            weight_decay                = CFG.weight_decay,
            fp16                        = CFG.fp16,
            load_best_model_at_end      = True,
            metric_for_best_model       = "f1",
            greater_is_better           = True,
            report_to                   = "none",
            save_total_limit            = 1,
        )

        dips_cb = DIPSTrainerCallback(collector=model_dips, tokenized_test=tok_test)
        trainer = AdvancedTrainer(model=model, args=args, train_dataset=tok_trn, eval_dataset=tok_val, compute_metrics=compute_metrics, callbacks=[dips_cb])
        dips_cb.trainer = trainer

        trainer.train()

        val_original_idx = train[train["fold"] == fold].index.values
        oof_logits[val_original_idx] = trainer.predict(tok_val).predictions
        fold_test_logits.append(trainer.predict(tok_test).predictions)

        del model
        torch.cuda.empty_cache()

    oof_logits_per_model[tag]  = oof_logits
    test_logits_per_model[tag] = np.mean(fold_test_logits, axis=0)
    dips_snapshots[tag]        = model_dips

print("==================================================")
print("🚀 Launching DIPS Global Pre-Training Pass on banglabert")
print("==================================================")

if CFG.use_pseudo:
    all_snaps = []
    for snaps in dips_snapshots.values():
        all_snaps.extend(snaps)
    snap_arr  = np.stack(all_snaps)
    max_probs = snap_arr.max(axis=-1)
    mean_conf = max_probs.mean(axis=0)
    std_conf  = max_probs.std(axis=0)

    useful = (mean_conf >= CFG.dips_conf_threshold) & (std_conf <= CFG.dips_var_threshold)
    print(f"\\n[DIPS] Identified {useful.sum()} 'Useful' pseudo-labels out of {len(test)} test samples.")

    if useful.sum() > 50:
        pseudo_test_probs = np.mean(list(test_logits_per_model.values()), axis=0)
        pseudo_df = test[useful].copy()
        pseudo_df["label_id"] = np.argmax(pseudo_test_probs[useful], axis=-1)

        full_train = pd.concat([train[["text", "label_id"]], pseudo_df[["text", "label_id"]]], ignore_index=True)
        mc0 = CFG.model_configs[0]
        tok0 = AutoTokenizer.from_pretrained(mc0["name"])
        tok_fn0 = make_tok_fn(tok0, mc0["max_len"])
        tok_full = Dataset.from_pandas(full_train.rename(columns={"label_id": "label"})).map(tok_fn0, batched=True)
        tok_tst  = Dataset.from_pandas(test[["text"]]).map(tok_fn0, batched=True)

        pseudo_model = TextMoEModel(mc0["name"], num_labels=N_CLASSES)
        pseudo_args = TrainingArguments(
            output_dir="/kaggle/working/pseudo", num_train_epochs=3, per_device_train_batch_size=mc0["batch_size"],
            per_device_eval_batch_size=mc0["batch_size"], learning_rate=1e-5, warmup_ratio=CFG.warmup_ratio,
            lr_scheduler_type="cosine", weight_decay=CFG.weight_decay, fp16=CFG.fp16, save_strategy="no", report_to="none",
        )
        pseudo_trainer = AdvancedTrainer(model=pseudo_model, args=pseudo_args, train_dataset=tok_full)
        pseudo_trainer.train()
        pseudo_logits = pseudo_trainer.predict(tok_tst).predictions
        test_logits_per_model["dips_pseudo"] = pseudo_logits
        del pseudo_model
        torch.cuda.empty_cache()

print("\\n── Training Level-2 LightGBM Stacker ──")
oof_probs_concat = np.hstack([scipy_softmax(logits, axis=-1) for logits in oof_logits_per_model.values()])
lgb_train = lgb.Dataset(oof_probs_concat, label=train["label_id"].values)

lgb_params = {
    'objective': 'multiclass', 'num_class': N_CLASSES, 'metric': 'multi_logloss',
    'boosting_type': 'gbdt', 'learning_rate': 0.05, 'num_leaves': 15, 'max_depth': 4,
    'feature_fraction': 0.8, 'verbose': -1, 'seed': CFG.seed
}
lgb_model = lgb.train(lgb_params, lgb_train, num_boost_round=300)

lgb_oof_probs = lgb_model.predict(oof_probs_concat)
test_probs_concat = np.hstack([scipy_softmax(logits, axis=-1) for k, logits in test_logits_per_model.items() if k != "dips_pseudo"])
lgb_test_probs = lgb_model.predict(test_probs_concat)

oof_labels = train["label_id"].values
preds = np.argmax(lgb_oof_probs, axis=-1)
_, _, f1_stacker, _ = precision_recall_fscore_support(oof_labels, preds, average="weighted", zero_division=0)
print(f"LightGBM Stacker OOF F1 Score (before Optuna): {f1_stacker:.4f}")

print("\\n── Running Optuna Log Bias Correction ──")
def objective(trial):
    shifts = [trial.suggest_float(f'shift_class_{c}', -2.0, 2.0) for c in range(N_CLASSES)]
    shifted_logits = np.log(np.clip(lgb_oof_probs, 1e-8, 1.0)) + np.array(shifts)
    preds = np.argmax(shifted_logits, axis=-1)
    _, _, f1, _ = precision_recall_fscore_support(oof_labels, preds, average="weighted", zero_division=0)
    return f1

study = optuna.create_study(direction="maximize", pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=30)
best_shifts = np.array([study.best_params[f'shift_class_{c}'] for c in range(N_CLASSES)])
print("Found Optimal Log Bias Shifts:", best_shifts)
print("Calibrated OOF F1 Score:", study.best_value)

print("\\n── Generating Final Predictions ──")
shifted_test_logits = np.log(np.clip(lgb_test_probs, 1e-8, 1.0)) + best_shifts
if "dips_pseudo" in test_logits_per_model:
    blended = 0.70 * shifted_test_logits + 0.30 * test_logits_per_model["dips_pseudo"]
    final_preds = np.argmax(blended, axis=-1)
else:
    final_preds = np.argmax(shifted_test_logits, axis=-1)

test["label"] = [reverse_label_map[p] for p in final_preds]

test.loc[test["category"] == "Non Disaster", "label"] = "Minimal"
deadly_mask = test['context'].str.contains('নিহত|মৃত্যু|মৃত|নিহতদের|প্রাণহানি|হত্যা', na=False)
low_severity_mask = test['label'].isin(['Minimal', 'Mild'])
upgraded_count = (deadly_mask & low_severity_mask).sum()
test.loc[deadly_mask & low_severity_mask, 'label'] = 'Moderate'
print(f"Applied Lexical Anchor Override: Upgraded {upgraded_count} fatal events to Moderate.")

submission = test[["image_id", "label"]]
submission.to_csv("submission_grandmaster_text.csv", index=False)

with zipfile.ZipFile("submission_grandmaster_text.zip", "w") as z:
    z.write("submission_grandmaster_text.csv", arcname="submission.csv")

print("\\n✅ 'submission_grandmaster_text.zip' is ready for upload.")
print("\\nFinal Prediction Distribution:")
print(submission["label"].value_counts())
