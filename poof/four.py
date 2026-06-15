import os
import random
import warnings
import numpy as np
import pandas as pd
import torch
from torch import nn
import safetensors.torch
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    Trainer, 
    TrainingArguments, 
    set_seed
)
from transformers.modeling_outputs import SequenceClassifierOutput
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
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)

seed_everything(42)

class CFG:
    model_name = "csebuetnlp/banglabert"
    base_path = "/kaggle/input/competitions/datathon-iiuc-cse-fest-2026/DisasterSeverity/"
    max_len = 128            # Exact baseline match
    epochs = 4               # Exact baseline match
    batch_size = 16
    lr = 2e-5
    n_folds = 5
    seed = 42
    
label_map = {"Minimal": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Catastrophic": 4}
reverse_label_map = {v: k for k, v in label_map.items()}

category_list = [
    'Landslides', 'Wildfire', 'Tropical Storm', 'Drought', 
    'Flood', 'Earthquake', 'Non Disaster', 'Human Damage'
]
cat2id = {c: i for i, c in enumerate(category_list)}

class BanglaBertWithCategoryFusion(nn.Module):
    def __init__(self, model_name, num_labels=5, num_categories=8):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = self.bert.config
        self.config.num_labels = num_labels
        
        # Deep MLP Fusion Head
        self.fusion = nn.Sequential(
            nn.Linear(self.config.hidden_size + num_categories, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask, category_features, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Electra uses [CLS] token
        text_features = outputs.last_hidden_state[:, 0, :]
        
        fused_features = torch.cat((text_features, category_features), dim=1)
        logits = self.fusion(fused_features)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return SequenceClassifierOutput(loss=loss, logits=logits)

print("Loading Data...")
train = pd.read_csv(f"{CFG.base_path}train.csv")
test = pd.read_csv(f"{CFG.base_path}test.csv")

val = pd.read_csv(f"{CFG.base_path}validation.csv")
train = pd.concat([train, val]).reset_index(drop=True)

train['label_id'] = train['label'].map(label_map)
train['context'] = train['context'].fillna("")
test['context'] = test['context'].fillna("")

skf = StratifiedKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed)
train['fold'] = -1
for fold, (train_idx, val_idx) in enumerate(skf.split(train, train['label_id'])):
    train.loc[val_idx, 'fold'] = fold

tokenizer = AutoTokenizer.from_pretrained(CFG.model_name)

def tokenize_fn(examples):
    tokenized = tokenizer(examples["context"], padding="max_length", truncation=True, max_length=CFG.max_len)
    
    cat_features = []
    for cat in examples['category']:
        one_hot = [0.0] * len(category_list)
        if cat in cat2id:
            one_hot[cat2id[cat]] = 1.0
        cat_features.append(one_hot)
        
    tokenized['category_features'] = cat_features
    return tokenized

test_dataset = Dataset.from_pandas(test[['context', 'category']])
tokenized_test = test_dataset.map(tokenize_fn, batched=True)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1}

class BalancedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        class_counts = np.bincount(train['label_id'].values)
        total_samples = len(train)
        weights = total_samples / (len(class_counts) * class_counts)
        weights_tensor = torch.tensor(weights, dtype=torch.float32).to(self.args.device)
        
        loss_fct = nn.CrossEntropyLoss(weight=weights_tensor)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        if state_dict is None:
            state_dict = self.model.state_dict()
        
        # Memory-Safe Copy: Prevents mutating the optimizer's active references
        contiguous_state_dict = {k: v.contiguous() for k, v in state_dict.items()}
        
        safetensors.torch.save_file(
            contiguous_state_dict, 
            os.path.join(output_dir, "model.safetensors"), 
            metadata={"format": "pt"}
        )

test_predictions = []

for fold in range(CFG.n_folds):
    print(f"\n{'='*20} FOLD {fold+1}/{CFG.n_folds} {'='*20}")
    
    trn_df = train[train['fold'] != fold].reset_index(drop=True)
    val_df = train[train['fold'] == fold].reset_index(drop=True)
    
    trn_ds = Dataset.from_pandas(trn_df[['context', 'category', 'label_id']].rename(columns={'label_id': 'label'}))
    val_ds = Dataset.from_pandas(val_df[['context', 'category', 'label_id']].rename(columns={'label_id': 'label'}))
    
    tok_trn = trn_ds.map(tokenize_fn, batched=True)
    tok_val = val_ds.map(tokenize_fn, batched=True)
    
    model = BanglaBertWithCategoryFusion(CFG.model_name, num_labels=5, num_categories=8)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    training_args = TrainingArguments(
        output_dir=f"/kaggle/working/fold_{fold}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=CFG.lr,
        per_device_train_batch_size=CFG.batch_size,
        per_device_eval_batch_size=CFG.batch_size,
        num_train_epochs=CFG.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        report_to="none",
        save_total_limit=1
    )
    
    trainer = BalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=tok_trn,
        eval_dataset=tok_val,
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    print(f"Predicting Test Set for Fold {fold+1}...")
    preds = trainer.predict(tokenized_test).predictions
    test_predictions.append(preds)

print("\nEnsembling 5-Fold Predictions...")
final_logits = np.mean(test_predictions, axis=0)
final_preds = np.argmax(final_logits, axis=-1)

test['label'] = [reverse_label_map[p] for p in final_preds]

print("Rule-Based Category Hack...")
test.loc[test['category'] == 'Non Disaster', 'label'] = 'Minimal'

submission = test[['image_id', 'label']]
submission.to_csv("submission.csv", index=False)

with zipfile.ZipFile('submission.zip', 'w') as zipf:
    zipf.write('submission.csv', arcname='submission.csv')

print("\nTraining Complete! 'submission.zip' is ready for upload.")
