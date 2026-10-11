import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

OUT = "/home/zbw/nlp/HW-1"
SEED = 42
plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False})

# ---------------- overall comparison ----------------
methods = [
    ("T1\nbinary BoW", 0.9913, 0.9778),
    ("T1\nTF BoW", 0.9931, 0.9834),
    ("T1\nTF-IDF BoW", 0.9939, 0.9857),
    ("T2\nGloVe 100d", 0.9861, 0.9669),
    ("T2\nw2v AG 100d", 0.9844, 0.9613),
    ("T2\nw2v NYT 100d", 0.9887, 0.9727),
    ("T3\nBERT (NPU)", 0.9826, 0.9624),
    ("T3\nBERT frozen", 0.5182, 0.4129),
]
labels = [m[0] for m in methods]
accs = [m[1] for m in methods]
f1s = [m[2] for m in methods]

fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(methods))
w = 0.38
b1 = ax.bar(x - w / 2, accs, w, label="Accuracy", color="#4C72B0")
b2 = ax.bar(x + w / 2, f1s, w, label="Macro F1", color="#DD8452")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
            f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x, labels, fontsize=9)
ax.set_ylim(0.3, 1.05)
ax.set_ylabel("score")
ax.set_title("NYT text classification: all methods (test set)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/fig_overall.png", dpi=150)
print("saved fig_overall.png")

# ---------------- T3 learning curve ----------------
hist = np.load(f"{OUT}/history_bert.npy", allow_pickle=True)
losses, v_acc, v_f1 = hist[0], hist[1], hist[2]
eps = np.arange(1, len(losses) + 1)
fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(eps, losses, "o-", color="#C44E52", label="train loss")
ax1.set_xlabel("epoch")
ax1.set_ylabel("train loss", color="#C44E52")
ax1.tick_params(axis="y", labelcolor="#C44E52")
ax2 = ax1.twinx()
ax2.plot(eps, v_acc, "s--", color="#4C72B0", label="val Acc")
ax2.plot(eps, v_f1, "^--", color="#55A868", label="val Macro F1")
ax2.set_ylabel("val score", color="#4C72B0")
ax2.set_ylim(0.9, 1.0)
ax1.set_xticks(eps)
fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=9)
fig.tight_layout(rect=[0, 0, 1, 0.92])
fig.savefig(f"{OUT}/fig_t3_curve.png", dpi=150)
print("saved fig_t3_curve.png")

# ---------------- T3 confusion matrix ----------------
cm = np.load(f"{OUT}/cm_bert.npy")
classes = ["business", "politics", "sports"]
fig, ax = plt.subplots(figsize=(5, 4.2))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(3), classes)
ax.set_yticks(range(3), classes)
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("BERT confusion matrix (test)")
for i in range(3):
    for j in range(3):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
fig.colorbar(im, shrink=0.8)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_t3_cm.png", dpi=150)
print("saved fig_t3_cm.png")

# ---------------- T2 t-SNE (word2vec NYT 100d mean) ----------------
df = pd.read_csv(f"{OUT}/nyt.csv")
df["text"] = df["text"].astype(str)
classes3 = ["business", "politics", "sports"]
label2id = {c: i for i, c in enumerate(classes3)}
y = df["label"].map(label2id).to_numpy()
texts = df["text"].to_numpy()
_, X_tmp, _, y_tmp = train_test_split(texts, y, test_size=0.2,
                                      stratify=y, random_state=SEED)
_, X_te, _, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5,
                                    stratify=y_tmp, random_state=SEED)


def tokenize(t):
    return simple_preprocess(str(t), deacc=True, min_len=2)


wv = Word2Vec.load(f"{OUT}/w2v_nyt_100.model").wv
X = np.stack([
    np.mean([wv[t] for t in tok if t in wv], axis=0)
    if any(t in wv for t in tok) else np.zeros(100, np.float32)
    for tok in map(tokenize, X_te)])
Z = TSNE(n_components=2, perplexity=30, random_state=SEED,
         init="pca", learning_rate="auto").fit_transform(X)
fig, ax = plt.subplots(figsize=(6.5, 5.5))
cmap = plt.get_cmap("tab10")
for i, c in enumerate(classes3):
    m = y_te == i
    ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.7,
               label=f"{c} (n={m.sum()})", color=cmap(i))
ax.set_title("t-SNE of NYT test docs (word2vec NYT 100d mean)")
ax.legend(fontsize=9, markerscale=3)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_t2_tsne.png", dpi=150)
print("saved fig_t2_tsne.png")

print("all figures done")
