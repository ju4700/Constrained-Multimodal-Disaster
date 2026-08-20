"""
baselines_text.py
Text-only baseline models on BanglaCalamityMMD dataset.
Models: SVM, Random Forest, CNN, RNN, BiLSTM, BanglaBERT (Sagor Sarkar),
        BanglaBERT (CSEBUET), IndicBERT
Saves: baselines_text_results.csv
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import warnings
from tqdm.auto import tqdm
warnings.filterwarnings("ignore")

class CFG:
    seed         = 42
    base_path    = "/kaggle/input/banglaclamity/BanglaCalamityMMD"
    train_csv    = os.path.join(base_path, "Disaster_train.csv")
    valid_csv    = os.path.join(base_path, "Disaster_validation.csv")
    test_csv     = os.path.join(base_path, "Disaster_test.csv")
    max_len      = 128
    batch_size   = 16
    epochs       = 3
    lr           = 2e-5
    embed_dim    = 128    
    hidden_dim   = 256
    output_csv   = "baselines_text_results.csv"

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

results = []

def record(name, modality, test_preds, test_trues):
    f1  = f1_score(test_trues, test_preds, average='macro')
    acc = accuracy_score(test_trues, test_preds)
    print(f"[{name}] Test Macro F1={f1:.4f}, Acc={acc:.4f}")
    results.append({"Model": name, "Modality": modality,
                    "Test Macro F1": round(f1, 4), "Test Accuracy": round(acc, 4)})

print("\n=== Classical ML Baselines ===")

train_text = train_df['context'].astype(str).tolist()
valid_text = valid_df['context'].astype(str).tolist()
test_text  = test_df['context'].astype(str).tolist()
train_labels = train_df['label'].values
test_labels  = test_df['label'].values

tfidf = TfidfVectorizer(max_features=20000, analyzer='char_wb', ngram_range=(2, 5))
X_train_tfidf = tfidf.fit_transform(train_text)
X_test_tfidf  = tfidf.transform(test_text)

svm = SVC(kernel='linear', C=1.0, probability=True, random_state=CFG.seed)
svm.fit(X_train_tfidf, train_labels)
record("SVM (TF-IDF)", "Text-Only", svm.predict(X_test_tfidf), test_labels)

rf = RandomForestClassifier(n_estimators=200, random_state=CFG.seed, n_jobs=-1)
rf.fit(X_train_tfidf, train_labels)
record("Random Forest (TF-IDF)", "Text-Only", rf.predict(X_test_tfidf), test_labels)

print("\n=== Deep Learning Baselines (trained from scratch) ===")

from collections import Counter
all_words = [w for t in train_text for w in t.split()]
word_counts = Counter(all_words)
vocab = {'<PAD>': 0, '<UNK>': 1}
for w, _ in word_counts.most_common(30000):
    vocab[w] = len(vocab)

def encode_text(texts, max_len=CFG.max_len):
    out = []
    for t in texts:
        ids = [vocab.get(w, 1) for w in t.split()[:max_len]]
        ids += [0] * (max_len - len(ids))
        out.append(ids)
    return np.array(out, dtype=np.int64)

X_train_enc = encode_text(train_text)
X_test_enc  = encode_text(test_text)
vocab_size   = len(vocab)


class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

train_dl = DataLoader(TextDataset(X_train_enc, train_labels), batch_size=CFG.batch_size, shuffle=True)
test_dl  = DataLoader(TextDataset(X_test_enc,  test_labels),  batch_size=CFG.batch_size, shuffle=False)


def train_and_eval(model, name):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    for epoch in range(CFG.epochs):
        model.train()
        for X_b, y_b in tqdm(train_dl, desc=f"{name} Epoch {epoch+1}"):
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            optimizer.step()
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for X_b, y_b in test_dl:
            X_b = X_b.to(device)
            out = model(X_b)
            preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            trues.extend(y_b.numpy())
    record(name, "Text-Only", np.array(preds), np.array(trues))


class TextCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, CFG.embed_dim, padding_idx=0)
        self.conv1 = nn.Conv1d(CFG.embed_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.pool  = nn.AdaptiveMaxPool1d(1)
        self.drop  = nn.Dropout(0.3)
        self.fc    = nn.Linear(256, num_classes)
    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.fc(self.drop(x))

print("\nRunning Text CNN...")
train_and_eval(TextCNN(), "CNN (Text)")

class TextRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, CFG.embed_dim, padding_idx=0)
        self.rnn   = nn.GRU(CFG.embed_dim, CFG.hidden_dim, batch_first=True)
        self.drop  = nn.Dropout(0.3)
        self.fc    = nn.Linear(CFG.hidden_dim, num_classes)
    def forward(self, x):
        x = self.embed(x)
        _, h = self.rnn(x)
        return self.fc(self.drop(h.squeeze(0)))

print("\nRunning RNN (GRU)...")
train_and_eval(TextRNN(), "RNN (GRU)")

class TextBiLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, CFG.embed_dim, padding_idx=0)
        self.lstm  = nn.LSTM(CFG.embed_dim, CFG.hidden_dim, batch_first=True, bidirectional=True)
        self.drop  = nn.Dropout(0.3)
        self.fc    = nn.Linear(CFG.hidden_dim * 2, num_classes)
    def forward(self, x):
        x = self.embed(x)
        _, (h, _) = self.lstm(x)
        h = torch.cat([h[0], h[1]], dim=1)
        return self.fc(self.drop(h))

print("\nRunning BiLSTM...")
train_and_eval(TextBiLSTM(), "BiLSTM")

print("\n=== Transformer Baselines ===")

class TransformerTextDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df        = df
        self.tokenizer = tokenizer
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tokenizer(
            str(row['context']),
            max_length=CFG.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].flatten(),
            'attention_mask': enc['attention_mask'].flatten(),
            'label':          torch.tensor(row['label'], dtype=torch.long)
        }


class TransformerClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size  = self.encoder.config.hidden_size
        self.drop    = nn.Dropout(0.3)
        self.fc      = nn.Linear(hidden_size, num_classes)
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.pooler_output if hasattr(out, 'pooler_output') and out.pooler_output is not None \
              else out.last_hidden_state[:, 0, :]
        return self.fc(self.drop(cls))


def run_transformer(model_name, display_name):
    print(f"\nRunning {display_name} ({model_name})...")
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    train_ds   = TransformerTextDataset(train_df, tokenizer)
    test_ds    = TransformerTextDataset(test_df,  tokenizer)
    train_load = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True)
    test_load  = DataLoader(test_ds,  batch_size=CFG.batch_size, shuffle=False)

    model     = TransformerClassifier(model_name, num_classes).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(CFG.epochs):
        model.train()
        for batch in tqdm(train_load, desc=f"Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            labels    = batch['label'].to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = model(input_ids, attn_mask)
                loss   = criterion(logits, labels)
            scaler.scale(loss).backward()
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                pass  

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in tqdm(test_load, desc="Testing"):
            input_ids = batch['input_ids'].to(device)
            attn_mask = batch['attention_mask'].to(device)
            logits    = model(input_ids, attn_mask)
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            trues.extend(batch['label'].numpy())
    record(display_name, "Text-Only", np.array(preds), np.array(trues))


run_transformer("sagorsarker/bangla-bert-base", "BanglaBERT (Sagor Sarkar)")
run_transformer("csebuetnlp/banglabert",        "BanglaBERT (CSEBUET)")
run_transformer("ai4bharat/indic-bert",         "IndicBERT")


results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
results_df.to_csv(CFG.output_csv, index=False)
print(f"\nSaved to {CFG.output_csv}")
