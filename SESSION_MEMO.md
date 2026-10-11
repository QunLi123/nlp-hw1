# NLP 作业（NYT 文本分类 T1→T2→T3）会话记忆

> 最后更新：2026-09-22（明天接着干，先读这个文件）
> 服务器：fuse2v（jumper 跳板），用户 zbw；本地：C:\Users\lwzbw\Desktop\projects\nlp\HW-1\

---

## 一、任务还原

图片信息 + 数据还原出的作业：**NYT 新闻文本分类**（business/politics/sports 3 类），8:1:1 划分，评测 Accuracy + Macro F1，三个递进步骤：

- **T1**：Bag-of-Words + 逻辑回归（binary bag / word frequency 两种特征）
- **T2**：word2vec（在 9 万条无标签 ag.csv 上训词向量 → 文档表示 → 分类）
- **T3**：BERT 微调

## 二、最终结果（test 集，同一 8:1:1 划分 seed=42）

| 方法 | Acc | Macro F1 |
|---|---|---|
| T1 binary BoW + LR | 0.9913 | 0.9778 |
| T1 词频 BoW + LR | 0.9931 | 0.9834 |
| T1 TF-IDF（加分） | 0.9939 | 0.9857 |
| T2 w2v 均值 + LR | 0.9826 | 0.9577 |
| T2 w2v TF-IDF 加权 | 0.9818 | 0.9554 |
| T2 w2v SIF 加权 | 0.9731 | 0.9409 |
| **T3 BERT 微调（NPU, max_len=128）** | **0.9852** | **0.9658** |
| T3 BERT 线性探测（冻结 encoder） | 0.5651 | 0.4743 |

- T2 附加：词表 22,504，OOV 覆盖率 95.9%，sim(juventus,soccer)=0.554
- T3 每类 F1：sports 0.9971 / business 0.9492 / politics 0.9510；NPU 3 epoch 仅 2 分钟（39s/epoch，≈CPU 60 倍）

## 三、服务器环境（关键，别踩坑）

### 两个 Python 环境
1. **`~/nlp/HW-1/venv`**（本项目 venv，T1/T2/画图用）：python 3.12.3，装 sklearn 1.9.1 / gensim 4.4.0 / pandas 3.0.6 / numpy 2.5.3 / torch 2.14.0+cpu / transformers 5.17.0 / matplotlib
2. **系统 python3**（=chitu 的 ~/.local 环境，T3 NPU 用）：torch 2.10.0+cpu + torch_npu 2.10.0.post4 + transformers 5.2.0；另往里装了 sklearn/joblib/threadpoolctl/narwhals/cloudpickle（`--no-deps` 避免动 numpy 1.26）

### venv 创建要点
- 系统 Python 缺 ensurepip，`python3 -m venv` 会"失败"但目录已建好 → 用 `curl bootstrap.pypa.io/get-pip.py` 引导进 venv
- pip 慢 → 用清华镜像：`-i https://pypi.tuna.tsinghua.edu.cn/simple`
- **教训**：PyPI 默认 torch wheel 会拉一堆 nvidia CUDA 依赖（好几 GB），把 home 盘（11G 剩余）塞满失败过。CPU torch 必须用 `--index-url https://download.pytorch.org/whl/cpu`
- HF 下载模型：设 `HF_ENDPOINT=https://hf-mirror.com`

### NPU 使用（T3）
- 跑 T3 用系统 python3 + chitu 那套环境变量（source set_env.sh + LD_LIBRARY_PATH 等，见 run_t3_npu.sh）
- 用 xlurm 独占设备：`xrun --device ascend -g 1 <cmd>`；**用 xlurm 时不要手动 export ASCEND_RT_VISIBLE_DEVICES**（用 `${ASCEND_RT_VISIBLE_DEVICES:-0}`）
- ⚠️ **950 的 torch_npu BERT 训练 bug**（已定位）：
  - 现象：全量微调 loss 突然变 nan，梯度 inf（2.10.0.post4 + CANN 9.1.0）
  - 触发条件 = **序列长度**：`max_len=256` 必炸（复现 4/4），192/224 随机炸，**`max_len=128` 完全稳定**
  - 非优化器/dtype 问题（AdamW/SGD、fp32/16/bf16 全试过）；冻结 encoder 只训分类头稳定；forward 单独正常
  - 规避：`max_len=128` + `clip_grad_norm_(1.0)`
  - 若想 max_len 更大 → 等 torch_npu 升级或用 CPU 全上下文训练

## 四、文件位置

### 服务器 `~/nlp/HW-1/`
- 主脚本：`t1_bow_lr_v2.py`、`t2_word2vec_v2.py`、`t3_compare.py`（T3 全量微调+线性探测）、`t3_bert_npu.py`、`make_figures.py`
- 运行包装：`run_t3_npu.sh`（source 环境 + xlurm 用）
- 产出：`w2v_ag.model`、`bert_finetuned.pt`、`cm_bert.npy`、`cm_bert_lp.npy`、`history_bert.npy`
- 图：`fig_overall.png`、`fig_t2_tsne.png`、`fig_t3_curve.png`、`fig_t3_cm.png`、`fig_t1_binary_cm.png`、`fig_t1_tfidf_cm.png`
- 报告：`REPORT.md`；调试脚本已挪到 `debug/`

### 本地 `C:\Users\lwzbw\Desktop\projects\nlp\HW-1\`
- `HW-1.md`（交付用 md 报告）、`REPORT.md`
- `report/main.tex` + `NKU.png` + `figures/` → 已编译 `main.pdf`（9 页，风格参照 BigData 推荐系统作业那两个 tex）
- 6 张实验图 png

### 跑 T3 的命令（记住）
```bash
xrun --device ascend -g 1 -n t3 /home/zbw/nlp/HW-1/run_t3_npu.sh
```

## 五、未完成 / 可续点
- 当前报告/PDF 已交付，无明确待办
- 可能的下一步：T3 想提分需绕开 128 截断（torch_npu 修复 or CPU 全上下文）；或把 PDF 传服务器；或下一题作业
- 磁盘：home 剩 ~2.8G（torch CPU 装完后的状态），大文件放共享盘