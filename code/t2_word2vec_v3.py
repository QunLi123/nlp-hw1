import json
import logging
import os
import time

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("t2v3")

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

NYT = "/home/zbw/nlp/HW-1/nyt.csv"
AG = "/home/zbw/nlp/HW-1/ag.csv"
GLOVE = "/home/zbw/nlp/HW-1/glove.6B.100d.txt"
OUT = "/home/zbw/nlp/HW-1"
SEED = 42
DIM = 100
np.random.seed(SEED)


def tokenize(text):
    return simple_preprocess(str(text), deacc=True, min_len=2)


def load_nyt():
    df = pd.read_csv(NYT)
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


def doc_vector(tokens, wv):
    v = [wv[t] for t in tokens if t in wv]
    return np.mean(v, axis=0) if v else np.zeros(DIM, np.float32)


def embed(docs, wv):
    return np.stack([doc_vector(tok, wv) for tok in map(tokenize, docs)])


def load_glove(path):
    wv = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            wv[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
    log.info("loaded GloVe: %d vectors (dim=%d)", len(wv),
             len(next(iter(wv.values()))))
    return wv


def train_w2v(corpus_texts, path, name):
    if os.path.exists(path):
        model = Word2Vec.load(path)
        log.info("loaded %s w2v: vocab=%d", name, len(model.wv))
    else:
        corpus = [tokenize(t) for t in corpus_texts]
        t0 = time.time()
        model = Word2Vec(sentences=corpus, vector_size=DIM, window=5,
                         min_count=5, sg=1, epochs=5, workers=8, seed=SEED)
        model.save(path)
        log.info("trained %s w2v: vocab=%d in %.1fs", name, len(model.wv),
                 time.time() - t0)
    return model.wv


def oov_coverage(wv, docs):
    toks = [t for doc in map(tokenize, docs) for t in doc]
    known = sum(1 for t in toks if t in wv)
    return known, len(toks)


def evaluate(X_tr, y_tr, X_val, y_val, X_te, y_te, name):
    best_C, best_va = 1.0, -1.0
    for C in [0.1, 0.5, 1.0, 5.0, 10.0]:
        clf = LogisticRegression(C=C, max_iter=2000, solver="lbfgs")
        clf.fit(X_tr, y_tr)
        va = accuracy_score(y_val, clf.predict(X_val))
        if va > best_va:
            best_C, best_va = C, va
    clf = LogisticRegression(C=best_C, max_iter=2000, solver="lbfgs")
    clf.fit(X_tr, y_tr)
    p = clf.predict(X_te)
    acc, mf1 = accuracy_score(y_te, p), f1_score(y_te, p, average="macro")
    log.info("[%s] test Acc=%.4f MacroF1=%.4f (best_C=%s)", name, acc, mf1,
             best_C)
    return {"method": name, "acc": float(acc), "macro_f1": float(mf1),
            "best_C": best_C, "dim": DIM}


def main():
    texts, y, classes = load_nyt()
    X_tr, y_tr, X_val, y_val, X_te, y_te = split_811(texts, y)
    log.info("split: train=%d val=%d test=%d", len(X_tr), len(X_val),
             len(X_te))

    results = []

    print("\n=== T2 (1) Pre-trained GloVe glove.6B 100d (mean) ===")
    glove = load_glove(GLOVE)
    k, n = oov_coverage(glove, X_te)
    log.info("GloVe OOV coverage on test: %d/%d = %.1f%%", k, n,
             100.0 * k / n)
    results.append(evaluate(embed(X_tr, glove), y_tr, embed(X_val, glove),
                            y_val, embed(X_te, glove), y_te,
                            "GloVe 6B 100d (mean)"))

    print("\n=== T2 (2) Word2Vec trained on AG News 100d (mean) ===")
    ag = pd.read_csv(AG)["text"].astype(str).tolist()
    wv_ag = train_w2v(ag, f"{OUT}/w2v_ag_100.model", "AG-News")
    k, n = oov_coverage(wv_ag, X_te)
    log.info("w2v-AG OOV coverage on test: %d/%d = %.1f%%", k, n,
             100.0 * k / n)
    results.append(evaluate(embed(X_tr, wv_ag), y_tr, embed(X_val, wv_ag),
                            y_val, embed(X_te, wv_ag), y_te,
                            "word2vec AG-News 100d (mean)"))

    print("\n=== T2 (3) Word2Vec trained on NYT 100d (mean) ===")
    wv_nyt = train_w2v(X_tr, f"{OUT}/w2v_nyt_100.model", "NYT")
    k, n = oov_coverage(wv_nyt, X_te)
    log.info("w2v-NYT OOV coverage on test: %d/%d = %.1f%%", k, n,
             100.0 * k / n)
    results.append(evaluate(embed(X_tr, wv_nyt), y_tr, embed(X_val, wv_nyt),
                            y_val, embed(X_te, wv_nyt), y_te,
                            "word2vec NYT 100d (mean)"))

    print("\n=== T2 summary (NYT test) ===")
    for r in results:
        print(f"  {r['method']}: Acc={r['acc']:.4f} "
              f"MacroF1={r['macro_f1']:.4f}")
    with open(f"{OUT}/t2_results.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info("saved %s/t2_results.json", OUT)


if __name__ == "__main__":
    main()
