import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "0")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, f1_score)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification,
                          AutoTokenizer, get_linear_schedule_with_warmup)

DATA = "/home/zbw/nlp/HW-1/nyt.csv"
OUT = "/home/zbw/nlp/HW-1"
MODEL_ID = "bert-base-uncased"
SEED = 42
MAX_LEN = 64
BATCH = 32
EPOCHS = 3
LR = 2e-5
DEV = "npu:0"

np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device(DEV)
print(f"torch={torch.__version__} torch_npu={__import__('torch_npu').__version__} device={DEV}")


def load_nyt():
    df = pd.read_csv(DATA)
    df["text"] = df["text"].astype(str)
    classes = sorted(df["label"].unique())
    label2id = {c: i for i, c in enumerate(classes)}
    y = df["label"].map(label2id).to_numpy()
    return df["text"].to_numpy(), y, classes, label2id


def split_811(texts, y):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        texts, y, test_size=0.2, stratify=y, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)
    return X_tr, y_tr, X_val, y_val, X_te, y_te


class NytDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(
            texts.tolist(), max_length=MAX_LEN, truncation=True,
            padding="max_length", return_tensors="pt")
        self.labels = torch.tensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()} | {"labels": self.labels[i]}


def to_dev(batch):
    return {k: v.to(device) for k, v in batch.items()}


def eval_model(model, loader):
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for b in loader:
            logits = model(**to_dev(b)).logits
            preds.append(logits.argmax(-1).cpu().numpy())
            ys.append(b["labels"].numpy())
    yp, yt = np.concatenate(preds), np.concatenate(ys)
    return (accuracy_score(yt, yp), f1_score(yt, yp, average="macro"),
            confusion_matrix(yt, yp), yt, yp)


def train(model, tr_ld, va_ld, freeze_encoder):
    if freeze_encoder:
        for n, p in model.named_parameters():
            if "classifier" not in n:
                p.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"  trainable params: {n_train:,} "
          f"({'classifier only' if freeze_encoder else 'full fine-tune'})")

    counts = np.bincount(tr_ld.dataset.labels.numpy())
    n_tr = len(tr_ld.dataset)
    class_weight = torch.tensor(
        n_tr / (len(counts) * counts), dtype=torch.float32).to(device)

    opt = AdamW(trainable, lr=LR, weight_decay=0.01)
    total_steps = len(tr_ld) * EPOCHS
    sched = get_linear_schedule_with_warmup(
        opt, num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps)

    history = {"train_loss": [], "val_acc": [], "val_f1": []}
    best_f1, best_state = -1.0, None
    for ep in range(1, EPOCHS + 1):
        model.train()
        ep_loss, n = 0.0, 0
        ep_t0 = time.time()
        for b in tr_ld:
            b = to_dev(b)
            logits = model(**b).logits
            loss = F.cross_entropy(logits, b["labels"], weight=class_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            ep_loss += loss.item()
            n += 1
        acc, mf1, _, _, _ = eval_model(model, va_ld)
        history["train_loss"].append(ep_loss / n)
        history["val_acc"].append(acc)
        history["val_f1"].append(mf1)
        print(f"  epoch {ep}: loss={ep_loss/n:.4f} val_acc={acc:.4f} "
              f"val_f1={mf1:.4f} ({time.time()-ep_t0:.0f}s)")
        if mf1 > best_f1:
            best_f1, best_state = mf1, model.state_dict()
    model.load_state_dict(best_state)
    return model, best_f1, history


def main():
    texts, y, classes, label2id = load_nyt()
    X_tr, y_tr, X_val, y_val, X_te, y_te = split_811(texts, y)
    print(f"train={len(X_tr)} val={len(X_val)} test={len(X_te)} classes={classes}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tr_ld = DataLoader(NytDataset(X_tr, y_tr, tokenizer), batch_size=BATCH,
                       shuffle=True, num_workers=8)
    va_ld = DataLoader(NytDataset(X_val, y_val, tokenizer), batch_size=BATCH,
                       num_workers=8)
    te_ld = DataLoader(NytDataset(X_te, y_te, tokenizer), batch_size=BATCH,
                       num_workers=8)

    results = {}
    for name, freeze in [("BERT 微调 (fine-tune)", False),
                         ("BERT 线性探测 (frozen)", True)]:
        print(f"\n=== T3 {name} ===")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, num_labels=len(classes)).to(device)
        model, bf1, hist = train(model, tr_ld, va_ld, freeze)
        acc, mf1, cm, yt, yp = eval_model(model, te_ld)
        results[name] = (acc, mf1)
        print(f"  test Acc={acc:.4f} MacroF1={mf1:.4f} (best val F1={bf1:.4f})")
        if not freeze:
            print(classification_report(yt, yp, target_names=classes, digits=4))
            torch.save(model.state_dict(), f"{OUT}/bert_finetuned.pt")
            np.save(f"{OUT}/cm_bert.npy", cm)
            np.save(f"{OUT}/history_bert.npy",
                    np.array([hist["train_loss"], hist["val_acc"],
                              hist["val_f1"]], dtype=object))
        else:
            np.save(f"{OUT}/cm_bert_lp.npy", cm)

    print("\n=== T3 comparison ===")
    for k, (a, f) in results.items():
        print(f"  {k}: Acc={a:.4f} MacroF1={f:.4f}")


if __name__ == "__main__":
    main()