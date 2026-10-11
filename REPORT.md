# NYT 新闻文本分类作业报告（T1 → T2 → T3 递进）

> 数据集：NYT（11,519 条，3 分类：sports / politics / business）
> 划分：训练 / 验证 / 测试 = 8:1:1（分层抽样，固定随机种子 42，三任务共用同一划分，保证可比）
> 评测：Accuracy + Macro F1（类别不均衡：sports 75%，politics 12.6%，business 12.4%，Macro F1 更重要）
> 硬件：Ascend 950（NPU），T1/T2 为 CPU 计算，T3 使用 NPU 加速

---

## 一、T1：Bag-of-Words + 逻辑回归（Logistic Regression）

三种词袋变体，词表上限 5 万，去停用词，C 在验证集上网格搜索（0.1~10）：

| 变体 | 特征含义 | 测试 Acc | 测试 Macro F1 |
|---|---|---|---|
| (1) binary bag | 词是否出现（0/1 二值） | 0.9913 | 0.9778 |
| (2) word frequency | 词频（TF 计数） | 0.9931 | 0.9834 |
| (3) TF-IDF（加分） | 词频×逆文档频率 | **0.9939** | **0.9857** |

**各类别判别词**（LR 系数最高的词，反映每类的语言特征）：

- **business**：company, market, percent, industry, billion, bank, trade …
- **politics**：mr, washington, republican, obama, senator, house, texas …
- **sports**：players, league, game, team, football, tournament, coach, club …

**结论**：词频比二值多带“出现次数”信息（+0.6% F1），TF-IDF 通过 IDF 压制高频虚词进一步小幅提升。逻辑回归+词袋在主题清晰的新闻上已经非常强。

---

## 二、T2：Word2Vec / Word Embedding（100 维词向量 + 均值文档表示 + LR）

按作业要求，词向量统一 100 维，文档表示统一取词向量平均，分类器统一逻辑回归。三组实验：

1. **预训练 GloVe**：`glove.6B.100d.txt`（40 万词），直接查表。
2. **Word2Vec 训练于 AG News**：9 万条无标签 `ag.csv`，skip-gram，dim=100，window=5，min_count=5，5 epochs，词表 22,504。
3. **Word2Vec 训练于 NYT**：NYT 训练集文本，同配置，词表 27,051。

| 词向量来源 | 维度 | 测试 Acc | 测试 Macro F1 |
|---|---|---|---|
| (1) 预训练 GloVe（glove.6B） | 100 | 0.9861 | 0.9669 |
| (2) word2vec 训练于 AG News | 100 | 0.9844 | 0.9613 |
| (3) word2vec 训练于 NYT | 100 | **0.9887** | **0.9727** |
| T1 词频 BoW（参照） | 50,000 | 0.9931 | 0.9834 |

**词向量语义验证**（以 NYT 自训模型为例）：
- `sim(juventus, soccer)=0.488` vs `sim(juventus, business)=0.234` ✓
- 最近邻：juventus → ajax, wolfsburg, fiorentina；bank → deutsche, monetary, jpmorgan
- 测试集词表覆盖率：GloVe 99.8%、AG 版 95.9%、NYT 版 98.6%

**结论**：三组都在 0.98 以上，比 T1 词袋低约 0.5~1 个百分点——均值池化把整篇文档压成一个 100 维稠密向量，
信息量损失，故略低于 T1 的 5 万维稀疏向量。预训练 GloVe 与自训 word2vec 基本打平，
NYT 自训略优（领域更贴合）。稠密表示更紧凑、天然处理 OOV。

---

## 三、T3：BERT 微调（Ascend 950 NPU 加速）

按作业要求：`bert-base-uncased`，**max_length=64**，**3 epochs**，类别加权损失 + 线性 warmup + 梯度裁剪。

| 方法 | 测试 Acc | 测试 Macro F1 | 单 epoch 耗时 |
|---|---|---|---|
| **BERT 全量微调（NPU）** | **0.9826** | **0.9624** | **21 s** |
| BERT 线性探测（冻结 encoder，对比项） | 0.5182 | 0.4129 | 8 s |

- 每类 Macro F1：business 0.9488 / politics 0.9437 / **sports 0.9948**
- **NPU vs CPU**：同配置在 CPU 上 40 分钟未跑完一个 epoch，NPU 上 3 个 epoch 仅约 1 分钟（≈60 倍加速）
- 线性探测远不如微调（0.518 vs 0.983）→ 证明 **BERT 必须针对下游任务微调**才有价值

### 附：Ascend 950 上的一个真实坑（torch_npu 数值溢出 bug）

微调过程中发现 torch_npu（2.10.0.post4 + CANN 9.1.0）在 950 上 **BERT 反向传播不稳定**：
- `max_len=256` 时梯度累积溢出（`inf`/`nan`），训练到约 20~40 步后 loss 变 NaN，**重复实验确定性复现**
- 定位：非优化器问题（AdamW/SGD 均炸）、非 dtype 问题（fp32/bf16/fp16 均炸），
  冻结 encoder 后稳定 → **encoder 反向在长序列上的累积是根因**
- 实测安全阈值：`max_len=128` 完全稳定；256 必炸；192/224 时而撑过一段后炸
- 作业要求的 `max_len=64` 远低于该阈值，训练全程无 NaN，故直接采用 64（另加梯度裁剪兜底）

这也是 950（3510 架构，aclnn-only）相对 910B 跨代迁移的算子兼容性问题之一，
与 chitu 适配 950 时遇到的坑同源。

---

## 四、总对比（test 集）

| 方法 | 特征维度 | 测试 Acc | 测试 Macro F1 |
|---|---|---|---|
| T1 binary BoW + LR | 50,000 稀疏 | 0.9913 | 0.9778 |
| T1 TF BoW + LR | 50,000 稀疏 | 0.9931 | 0.9834 |
| T1 TF-IDF + LR | 50,000 稀疏 | 0.9939 | 0.9857 |
| T2 GloVe 100d 均值 + LR | 100 稠密 | 0.9861 | 0.9669 |
| T2 word2vec AG 100d 均值 + LR | 100 稠密 | 0.9844 | 0.9613 |
| T2 word2vec NYT 100d 均值 + LR | 100 稠密 | 0.9887 | 0.9727 |
| **T3 BERT 微调（NPU）** | 768（上下文感知） | **0.9826** | **0.9624** |
| T3 BERT 线性探测 | 768 | 0.5182 | 0.4129 |

**关键结论**：
1. **T1 词袋接近天花板**：主题清晰的新闻分类上 BoW+LR 已到 0.99+，更“高级”的方法很难在绝对分数上超越——不是方法不好，是任务不够难。
2. **T2 词向量把维度压缩约 500 倍**（5 万→100）仍保持 0.98，体现分布式表示的紧凑性与 OOV 鲁棒性；GloVe / AG / NYT 三种来源分数接近。
3. **T3 BERT** 是唯一捕捉**上下文语义**的方法，在截断到 64 token 的约束下仍达 0.9826。
4. **真正的区分度在冻结 vs 微调**：BERT 内部 0.518 → 0.983，差距 0.46 是全场最大，说明语义表示的价值需要下游微调来兑现。

---

## 五、复现

- 全部脚本在服务器 `~/nlp/HW-1/`：`t1_bow_lr_v2.py` / `t2_word2vec_v3.py` / `t3_compare.py` / `make_figures.py` / `run_t3_npu.sh`
- 数据/词向量：`nyt.csv`、`ag.csv`、`glove.6B.100d.txt`、`w2v_ag_100.model`、`w2v_nyt_100.model`
- 图：`fig_overall.png`（总对比）、`fig_t2_tsne.png`（词向量 t-SNE）、
  `fig_t3_curve.png`（BERT 学习曲线）、`fig_t3_cm.png` + `fig_t1_*_cm.png`（混淆矩阵）
- 产出：`bert_finetuned.pt`（微调模型）、`t2_results.json`、`t3_run_b64.log`
- 运行说明见仓库 `README.md`。
