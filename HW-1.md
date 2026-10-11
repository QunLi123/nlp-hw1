# 新闻文本分类：从词袋到 BERT

> **自然语言处理作业报告**　|　南开大学 计算机学院　|　邹博闻 2312251
> 数据集：NYT（11,519 条，business / politics / sports 三分类）
> 划分：训练 / 验证 / 测试 = 8:1:1（分层抽样，固定种子 42，三任务共用同一划分）
> 评测：Accuracy + Macro F1　|　T3 使用 Ascend 950 NPU 加速

---

## 一、任务描述

本次作业的目标是在 NYT 数据集上做新闻文本分类，把每篇文章分到 business、politics、sports 三个类别之一。任务要求按三个步骤递进完成，每步用不同的文本表示方法做同一个分类任务：

1. **T1**：Bag-of-Words + 逻辑回归。特征有两种做法：binary bag（只记录词是否出现）、word frequency（记录词频）。
2. **T2**：word2vec / 词嵌入。用预训练 GloVe 或自行训练的词向量，把文档表示成词向量的聚合，再训练分类器。
3. **T3**：BERT 微调。

笔者在每条路径上还做了对照实验：T1 加 TF-IDF 变体；T2 按作业要求比较预训练 GloVe、AG News 自训 word2vec、NYT 自训 word2vec 三种来源；T3 比较全量微调和冻结 encoder 的线性探测。

## 二、数据集分析

数据给了两个 csv：`nyt.csv`（text, label，有标签，用于分类）和 `ag.csv`（只有 text，9 万条 AG News 文本）。ag.csv 的用途是 T2 无监督训练 word2vec——NYT 自己只有一万多篇，直接训词向量词表会稀疏，9 万条新闻文本量足够。这个判断在实验里得到验证：用 ag.csv 训出的词向量在 NYT 测试集上词表覆盖率达 95.9%。

| 指标 | nyt.csv | ag.csv |
|---|---|---|
| 样本数 | 11,519 | 90,000 |
| 类别 | sports 8,639（75.0%）/ politics 1,451（12.6%）/ business 1,429（12.4%） | 无标签 |

两个注意点：

1. **类别严重不均衡**。sports 占四分之三。全猜 sports 的模型 Accuracy 也有 75%，所以必须同时看 Macro F1（三类同等对待）。
2. **文本很长**。T3 受 BERT 序列长度限制要截断，会损失上下文，后面能看到它对准确率的影响。

## 三、T1：Bag-of-Words + 逻辑回归

向量每一维对应词表里的一个词，取值方式区分两个变体：

- binary bag：$v_i = \mathbb{I}[\text{词 } i \text{ 出现}]$
- word frequency：$v_i = \text{count}(i)$

两个版本都去英文停用词，词表上限 5 万，惩罚系数 $C$ 在验证集网格搜索，测试集报数。**加分项 TF-IDF**：$\text{tfidf}(i) = \text{count}(i) \cdot \log(N/\text{df}(i))$。

| 变体 | Accuracy | Macro F1 |
|---|---|---|
| binary bag | 0.9913 | 0.9778 |
| word frequency | 0.9931 | 0.9834 |
| TF-IDF（加分） | **0.9939** | **0.9857** |

三个变体都到 0.99 以上。从二值到词频多了"出现几次"的信息（+0.56% F1），TF-IDF 再小幅提升。

**各类别判别词**（逻辑回归系数最高的词）：

```
business: company, market, percent, industry, billion, bank, trade ...
politics : mr, washington, republican, obama, senator, house, texas ...
sports   : players, league, game, team, football, tournament, coach ...
```

词和类别对应很干净，说明 NYT 新闻主题特征明显，词袋模型够用。

## 四、T2：Word2Vec 词向量

按作业要求，词向量统一 **100 维**，文档表示统一取词向量平均，分类器统一用逻辑回归。三组实验：

1. **预训练 GloVe**：`glove.6B.100d.txt`（40 万词），直接查表。
2. **Word2Vec 训练于 AG News**：9 万条无标签 `ag.csv`，skip-gram，dim=100，window=5，min_count=5，5 epochs，词表 22,504。
3. **Word2Vec 训练于 NYT**：NYT 训练集文本，同配置，词表 27,051。

文档表示取文档内所有有效词向量的平均：

$$\vec{d} = \frac{1}{n}\sum_{i=1}^{n} \vec{w}_i$$

| 词向量来源 | 维度 | Accuracy | Macro F1 |
|---|---|---|---|
| (1) 预训练 GloVe（glove.6B） | 100 | 0.9861 | 0.9669 |
| (2) word2vec 训练于 AG News | 100 | 0.9844 | 0.9613 |
| (3) word2vec 训练于 NYT | 100 | **0.9887** | **0.9727** |
| T1 词频 BoW（参照） | 50,000 | 0.9931 | 0.9834 |

三组都在 0.98 以上，比 T1 低约 0.5~1 个百分点——均值池化丢信息，稀疏高维的词袋保留更完整。预训练 GloVe（覆盖率 99.8%）与 AG 自训（95.9%）基本打平，NYT 自训略优（98.6%），说明这条稠密路线上词向量来源不是主要瓶颈。

**词向量语义验证**：`sim(juventus, soccer)=0.488` vs `sim(juventus, business)=0.234`；最近邻 juventus→ajax/wolfsburg/fiorentina，bank→deutsche/monetary/jpmorgan。t-SNE 可视化三类基本可分（见 `fig_t2_tsne.png`）。

## 五、T3：BERT 微调（Ascend 950 NPU）

`bert-base-uncased` 序列分类微调，按作业要求 **max_length=64、3 epochs**，类别加权交叉熵，AdamW（lr=2e-5）+ 线性 warmup + 梯度裁剪，保留验证 Macro F1 最优权重。

训练在 Ascend 950 NPU 上进行：3 epochs 共约 1 分钟（21~23s/epoch），同样配置 CPU 上 40 分钟连一个 epoch 都跑不完，**NPU 约快 60 倍**。

| 方法 | Accuracy | Macro F1 | 每 epoch 耗时 |
|---|---|---|---|
| BERT 全量微调（NPU） | **0.9826** | **0.9624** | 21 s |
| BERT 线性探测（冻结 encoder） | 0.5182 | 0.4129 | 8 s |

每类 Macro F1：business 0.9488 / politics 0.9437 / sports 0.9948。

**线性探测 vs 微调**：只训分类头只有 0.5182，全量微调 0.9826，差距 0.46 是全场最大。预训练特征如果不针对下游任务微调，直接用来分类效果很差——**微调是发挥 BERT 价值的关键**。

### 踩坑记录：torch_npu 在 950 上的梯度溢出

首次 NPU 微调，训练到 20~40 步左右 loss 突然变 nan。定位过程：

1. 排除优化器和 dtype：AdamW / SGD、fp32 / fp16 / bf16 组合全炸；
2. 排除数据：同数据 CPU 训练正常，锁定是 torch_npu 在 950 上的反向不稳定；
3. 定位参数：在梯度首次出现 inf 处打断点，问题在 encoder 内部——冻结 encoder 只训分类头时稳定；
4. 确定触发条件：**序列长度是关键**。`max_len=128` 完全稳定，`max_len=256` 必炸（复现 4 次全挂），192/224 也会中途随机炸。判断是长序列某算子的反向梯度累加溢出。

作业要求的 `max_len=64` 远低于该阈值，训练全程无 nan，故直接采用 64（另加梯度裁剪兜底）。这属于 950（3510 架构）上 torch_npu 算子兼容性问题的一例。

## 六、实验结果对比

| 方法 | 特征维度 | Accuracy | Macro F1 |
|---|---|---|---|
| T1 binary BoW + LR | 50,000 稀疏 | 0.9913 | 0.9778 |
| T1 TF BoW + LR | 50,000 稀疏 | 0.9931 | 0.9834 |
| T1 TF-IDF + LR | 50,000 稀疏 | **0.9939** | **0.9857** |
| T2 GloVe 100d 均值 + LR | 100 稠密 | 0.9861 | 0.9669 |
| T2 word2vec AG 100d 均值 + LR | 100 稠密 | 0.9844 | 0.9613 |
| T2 word2vec NYT 100d 均值 + LR | 100 稠密 | 0.9887 | 0.9727 |
| T3 BERT 微调（NPU） | 768 上下文 | 0.9826 | 0.9624 |
| T3 BERT 线性探测 | 768 | 0.5182 | 0.4129 |

**为什么分数拉不开**：

1. **任务不够难**。NYT 三类在词面上区分度极高，词袋已接近上限；
2. **BERT 被截断拖累**。按作业只能看 64 token，词袋看全文；
3. **T2 均值、BERT 池化都丢信息**。稠密表示路线天然比不过"看全文"的词袋——新闻主题分类本身对顺序不敏感；
4. **真正的区分度在冻结 vs 微调**。差距最大的不是方法之间，而是 BERT 内部（0.518 vs 0.983）。

## 七、总结

1. 词袋 + 逻辑回归最朴素也最稳，三个变体都到 0.99，TF-IDF 略优于词频、词频略优于二值；
2. word2vec 把 5 万维稀疏表示压成 100 维稠密表示，损失约一个百分点，天然处理 OOV；GloVe / AG / NYT 三种来源分数接近；
3. BERT 在 NPU 上微调达 0.9826，训练约 1 分钟；冻结 encoder 只有 0.5182，充分说明微调的必要性；
4. Ascend 950 的 torch_npu 存在长序列反向梯度溢出，用受控序列长度（本例 64）+ 梯度裁剪规避。

代码、模型、图片均保存在服务器 `~/nlp/HW-1/`，运行说明见仓库 `README.md`。完整代码与报告仓库：<https://github.com/QunLi123/nlp-hw1>
