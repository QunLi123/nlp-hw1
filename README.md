# NLP 作业一：NYT 新闻文本分类（T1 词袋 → T2 Word2Vec → T3 BERT）

南开大学 计算机学院 邹博闻 2312251

在 NYT 新闻数据集（business / politics / sports 三分类）上，用三种文本表示方法完成同一分类任务，
统一在 8:1:1（分层抽样，seed=42）划分的测试集上报告 **Accuracy** 与 **Macro F1**。

## 结果一览（test set）

| 方法 | 特征维度 | Accuracy | Macro F1 |
|---|---|---|---|
| T1 binary BoW + LR | 50,000 稀疏 | 0.9913 | 0.9778 |
| T1 word-frequency BoW + LR | 50,000 稀疏 | 0.9931 | 0.9834 |
| T1 TF-IDF BoW + LR（加分） | 50,000 稀疏 | **0.9939** | **0.9857** |
| T2 预训练 GloVe 100d + 均值 + LR | 100 稠密 | 0.9861 | 0.9669 |
| T2 word2vec(AG News) 100d + 均值 + LR | 100 稠密 | 0.9844 | 0.9613 |
| T2 word2vec(NYT) 100d + 均值 + LR | 100 稠密 | 0.9887 | 0.9727 |
| T3 BERT-base-uncased 微调（max_len=64, 3 epochs, NPU） | 768 | 0.9826 | 0.9624 |
| T3 BERT 线性探测（frozen） | 768 | 0.5182 | 0.4129 |

详细分析见 `HW-1.md` / `REPORT.md` / `report/main.pdf`（PDF 报告）。

## 目录结构

```
.
├── code/
│   ├── t1_bow_lr_v2.py        # T1：binary / word-frequency / TF-IDF 三种词袋 + LR
│   ├── t2_word2vec_v3.py      # T2：GloVe / word2vec(AG) / word2vec(NYT)，100 维均值 + LR
│   ├── t3_compare.py          # T3：BERT 微调 + 线性探测（max_len=64, 3 epochs）
│   ├── make_figures.py        # 画总对比 / t-SNE / 学习曲线 / 混淆矩阵
│   └── run_t3_npu.sh          # 在 Ascend NPU 上跑 T3 的环境脚本
├── report/                    # LaTeX 报告（main.tex）与编译好的 main.pdf
│   └── figures/
├── HW-1.md                    # 交付用 Markdown 报告
├── REPORT.md                  # 实验记录
└── README.md
```

## 环境

T1 / T2 / 画图（CPU）：

```bash
python -m venv venv && source venv/bin/activate
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    scikit-learn gensim pandas numpy matplotlib
# CPU 版 torch（避免拉到 CUDA 依赖）：
pip install --index-url https://download.pytorch.org/whl/cpu torch
```

T3（Ascend NPU，950）：系统 Python + `torch_npu`，运行前 `source run_t3_npu.sh` 里配置的 CANN 环境。

## 数据与预训练词向量准备

代码里的路径以 `~/nlp/HW-1/` 为根，请把下列文件放到该目录（或修改脚本顶部的路径常量）：

| 文件 | 说明 |
|---|---|
| `nyt.csv` | NYT 数据集（列：`text,label`） |
| `ag.csv` | AG News 文本（列：`text`，无标签，9 万条） |
| `glove.6B.100d.txt` | GloVe 6B 100 维词向量，来自 http://nlp.stanford.edu/data/glove.6B.zip |

```bash
# 下载并解压 GloVe 100d（只需 100d 一个文件）
curl -L -o glove.6B.zip https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
python -c "import zipfile; zipfile.ZipFile('glove.6B.zip').extract('glove.6B.100d.txt')"
```

> 仓库未包含 `*.csv`、`*.model`、`glove*.txt`、`*.pt` 等大文件，按需自行下载/生成。

## 运行

```bash
# T1：三种词袋
python code/t1_bow_lr_v2.py

# T2：GloVe / word2vec(AG) / word2vec(NYT)，100 维
python code/t2_word2vec_v3.py

# T3：BERT 微调 + 线性探测（需 NPU 环境）
xrun --device ascend -g 1 -n t3 bash code/run_t3_npu.sh

# 画图
python code/make_figures.py
```

## 复现要点

- 三任务共用同一 8:1:1 划分：`train_test_split(..., test_size=0.2, stratify=y, random_state=42)`
  再对剩下 20% 做 1:1 切分，保证 val/test 各 10%。
- T2 词向量统一 **100 维**，文档表示统一为「文档内有效词向量取平均」。
- T2 的 NYT 自训 word2vec 只用训练集文本，避免信息泄漏。
- T3 按作业要求 **max_length=64、3 epochs**；950 上 torch_npu 存在长序列反向梯度溢出
  （256 必炸、128 稳定），64 远低于阈值，训练稳定。
