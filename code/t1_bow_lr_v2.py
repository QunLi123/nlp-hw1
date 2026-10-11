import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             f1_score, classification_report)
from sklearn.model_selection import train_test_split

plt.rcParams["font.size"] = 10

DATA = "/home/zbw/nlp/HW-1/nyt.csv"
OUT = "/home/zbw/nlp/HW-1"
SEED = 42
np.random.seed(SEED)


def load_data(path):
    df = pd.read_csv(path)
    df["text"] = df["text"].astype(str)
    classes = sorted(df["label"].unique())
    label2id = {c: i for i, c in enumerate(classes)}
    y = df["label"].map(label2id).to_numpy()
    return df["text"].to_numpy(), y, classes


def split_811(texts, y):
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        texts, y, test_size=0.2, stratify=y, random_state=SEED)
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)
    return X_tr, y_tr, X_val, y_val, X_te, y_te


def run_variant(name, binary, use_idf, X_tr, y_tr, X_val, y_val, X_te, y_te,
                classes):
    t0 = time.time()
    if use_idf:
        vec = TfidfVectorizer(lowercase=True, stop_words="english",
                              max_features=50000, sublinear_tf=True)
    else:
        vec = CountVectorizer(binary=binary, lowercase=True,
                              stop_words="english", max_features=50000)
    X_tr_v = vec.fit_transform(X_tr)
    X_val_v, X_te_v = vec.transform(X_val), vec.transform(X_te)
    print(f"[{name}] vocab={len(vec.vocabulary_)} "
          f"fit={time.time()-t0:.1f}s nnz/row={X_tr_v.nnz/X_tr_v.shape[0]:.0f}")

    best_C, best_va = 1.0, -1.0
    for C in [0.1, 0.5, 1.0, 5.0, 10.0]:
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf.fit(X_tr_v, y_tr)
        va = accuracy_score(y_val, clf.predict(X_val_v))
        if va > best_va:
            best_C, best_va = C, va

    clf = LogisticRegression(C=best_C, max_iter=2000, solver="lbfgs")
    clf.fit(X_tr_v, y_tr)
    rows = []
    for split, X, y in [("train", X_tr_v, y_tr),
                        ("val", X_val_v, y_val),
                        ("test", X_te_v, y_te)]:
        p = clf.predict(X)
        acc = accuracy_score(y, p)
        mf1 = f1_score(y, p, average="macro")
        rows.append((split, acc, mf1))
        print(f"  [{split}] Acc={acc:.4f} MacroF1={mf1:.4f}")
    print(f"[{name}] best_C={best_C} total={time.time()-t0:.1f}s")

    if use_idf or binary:
        save_cm(f"{OUT}/fig_t1_{name}_cm.png",
                confusion_matrix(y_te, clf.predict(X_te_v)), classes)
    return best_C, rows, clf, vec


def save_cm(path, cm, classes):
    fig, ax = plt.subplots(figsize=(5, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"saved {path}")


def top_words_per_class(vec, clf, classes, n=12):
    words = np.array(vec.get_feature_names_out())
    coef = clf.coef_
    print("\nTop discriminative words per class:")
    out = {}
    for i, c in enumerate(classes):
        idx = np.argsort(coef[i])[::-1][:n]
        out[c] = words[idx].tolist()
        print(f"  {c}: {', '.join(words[idx])}")
    return out


def main():
    texts, y, classes = load_data(DATA)
    X_tr, y_tr, X_val, y_val, X_te, y_te = split_811(texts, y)
    print(f"train={len(X_tr)} val={len(X_val)} test={len(X_te)} "
          f"classes={classes}")

    results = {}
    print("\n=== T1 (1) Binary BoW ===")
    c, rows, clf, vec = run_variant("binary", True, False,
                                    X_tr, y_tr, X_val, y_val, X_te, y_te, classes)
    results["binary"] = rows
    top_words_per_class(vec, clf, classes)

    print("\n=== T1 (2) Word-frequency BoW ===")
    c, rows, clf, vec = run_variant("tf", False, False,
                                    X_tr, y_tr, X_val, y_val, X_te, y_te, classes)
    results["tf"] = rows

    print("\n=== T1 (bonus) TF-IDF BoW ===")
    c, rows, clf, vec = run_variant("tfidf", False, True,
                                    X_tr, y_tr, X_val, y_val, X_te, y_te, classes)
    results["tfidf"] = rows

    print("\n=== T1 test summary ===")
    for k, v in results.items():
        t = dict(v)["test"]
        print(f"  {k}: Acc={t[1]:.4f} MacroF1={t[2]:.4f}")


if __name__ == "__main__":
    main()