# Umami-MRNN

Code and data for:

[Umami-MRNN: Deep learning-based prediction of umami peptide using RNN and MLP](https://doi.org/10.1016/j.foodchem.2022.134935)
Lulu Qi, [Jialuo Du](https://daibiaoxuwu.github.io), Yue Sun, Yongzhao Xiong, Xinyao Zhao, Daodong Pan, Yueru Zhi, Yali Dang, Xinchang Gao.
*Food Chemistry*, 2023.

**Just want predictions?** Use the [online demo](https://daibiaoxuwu.github.io/umami-mrnn/) (runs in your browser, nothing to install; [source](https://github.com/daibiaoxuwu/umami-mrnn)).

[中文说明见下方](#中文说明)

## Quick start

Requires Python 3.9 or newer. No GPU is needed; everything runs on an ordinary laptop in under a minute.

```sh
pip install -r requirements.txt

python predict.py EGS DQR GFP      # predict with the published models
python train.py                    # retrain from data/peptides.csv
python tests/check.py              # optional: check everything is consistent
```

`predict.py` prints one line per peptide:

```
name  sequence  prediction  threshold_mmol_L
EGS   EGS       umami       28.30
GFP   GFP       non-umami   47.08
```

It also accepts a FASTA file (`python predict.py my_peptides.fasta`). Sequences must be 2–39 residues of the 20 standard amino acids.

## How the model works

Each network predicts a peptide's umami **taste threshold** in mmol/L (lower = stronger umami).

1. **MLP** (multilayer perceptron) reads 1,093 numbers describing the whole sequence: amino-acid composition (AAC), dipeptide composition (DPC), dipeptide deviation from expected mean (DDE), and composition / transition / distribution of physicochemical groups (CTD). Definitions follow [iFeature](https://github.com/Superzchen/iFeature). Layers: 1093 → 256 → 32 → 1.
2. **RNN** (recurrent neural network) reads the peptide one residue at a time; each residue is a 10-number code of physicochemical properties. Two layers of 32 units.
3. **Combined prediction** = 0.67 × MLP + 0.33 × RNN. Below **40 mmol/L** the peptide is classed as umami.

Bitter peptides (no umami taste) are trained towards a threshold of 50, so that their predictions land clearly above the cutoff.

## Files

| File | What it is |
|---|---|
| `data/peptides.csv` | The dataset: 129 umami peptides with thresholds and 289 bitter peptides |
| `data/original/` | The original spreadsheets from the paper |
| `features.py` | Turns a sequence into the numbers the networks read |
| `model.py` | The two networks and how they are combined |
| `train.py` | Trains both networks and reports accuracy on held-out peptides |
| `predict.py` | Predicts new peptides |
| `published/` | Weights of the models used in the paper's demo |
| `web_demo.py` | Exports trained models for the web demo (`python web_demo.py export <folder>`) |
| `tests/check.py` | Consistency checks (see below) |
| `tools/excel_to_csv.py` | Rebuilds `data/peptides.csv` from the spreadsheets |

## Results and reproducibility

- `published/` holds the exact models behind the paper's demo. `tests/check.py` confirms that the PyTorch code loads them and reproduces the original TensorFlow predictions (within 0.0003 mmol/L on 109 peptides).
- Retraining with `train.py` gives **about 80–83% accuracy** on held-out peptides (20% of each class; seeds 0–2). It does not recreate the published weights exactly: the original random split and seed are unknown.
- The original spreadsheet lists most umami peptides two or three times. `data/peptides.csv` keeps each peptide once, and `train.py` splits by peptide, so no test peptide is also seen in training. Accuracy measured on a split that contains repeats will be higher.

## Changes from the first version

This repository was rewritten in 2026 to be easier to run: PyTorch instead of TensorFlow 2.x, a single CSV dataset, features computed from sequences rather than copied from spreadsheets, and a training script that matches the published models. The previous TensorFlow scripts (`main.py`, `Merge_BP.py`, `Merge_RNN.py`) are in the git history at commit `f64bf12`. While rewriting we found:

- row 77 of `AllUmami.xlsx` is labelled VDV but its descriptors are those of VAV (fixed in the CSV);
- in `AllUmami.xlsx`, the dipeptide composition (DPC) of 27 peptides divides by sequence length instead of the number of residue pairs (`features.py` uses the number of pairs, as the published models do);
- `PeptideSequence.xlsx` stores the 10-digit residue codes as numbers, which drops their leading zeros (`features.py` defines the codes directly).

---

## 中文说明

**只需要预测？** 请使用[在线演示](https://daibiaoxuwu.github.io/umami-mrnn/)（在浏览器中运行，无需安装；[源代码](https://github.com/daibiaoxuwu/umami-mrnn)）。

### 快速开始

需要 Python 3.9 或更高版本，无需 GPU，普通笔记本电脑一分钟内即可完成。

```sh
pip install -r requirements.txt

python predict.py EGS DQR GFP      # 使用论文中的模型进行预测
python train.py                    # 用 data/peptides.csv 重新训练
python tests/check.py              # 可选：一致性检查
```

`predict.py` 也可以读取 FASTA 文件（`python predict.py my_peptides.fasta`）。序列须由 20 种标准氨基酸组成，长度 2–39。

### 模型原理

每个网络预测肽的鲜味**阈值**（mmol/L，数值越低鲜味越强）。

1. **MLP**（多层感知机）读取描述整条序列的 1,093 个特征：氨基酸组成（AAC）、二肽组成（DPC）、二肽偏离期望均值（DDE）以及理化性质分组的组成/转换/分布（CTD），定义参照 [iFeature](https://github.com/Superzchen/iFeature)。
2. **RNN**（循环神经网络）逐个读取残基，每个残基用 10 位理化性质编码表示，两层、每层 32 个单元。
3. **最终预测** = 0.67 × MLP + 0.33 × RNN；低于 **40 mmol/L** 判定为鲜味肽。

训练时苦味肽（非鲜味）的目标阈值设为 50，使其预测值明显高于判定阈值。

### 结果与可复现性

- `published/` 是论文演示所用的模型。`tests/check.py` 验证 PyTorch 代码可加载这些模型，并复现原 TensorFlow 的预测结果（109 条肽，误差小于 0.0003 mmol/L）。
- 用 `train.py` 重新训练，在留出的测试肽上准确率**约为 80–83%**（每类各留 20%，随机种子 0–2）。由于原始的数据划分和随机种子未知，重新训练不会得到与论文完全相同的权重。
- 原始表格中多数鲜味肽重复出现两到三次。`data/peptides.csv` 中每条肽只保留一次，`train.py` 按肽划分训练/测试集，因此测试肽不会出现在训练集中；若划分时包含重复样本，测得的准确率会偏高。

### 与第一版的区别

本仓库于 2026 年重写，以便更容易运行：使用 PyTorch 代替 TensorFlow 2.x，数据整理为一个 CSV 文件，特征直接由序列计算，训练脚本与论文模型结构一致。旧的 TensorFlow 脚本（`main.py`、`Merge_BP.py`、`Merge_RNN.py`）保留在 git 历史的 `f64bf12` 提交中。重写时发现：

- `AllUmami.xlsx` 第 77 行标为 VDV，但其特征对应 VAV（已在 CSV 中更正）；
- `AllUmami.xlsx` 中有 27 条肽的二肽组成（DPC）除以了序列长度而非残基对数（`features.py` 按残基对数计算，与论文模型一致）；
- `PeptideSequence.xlsx` 将 10 位残基编码存为数字，丢失了前导零（`features.py` 直接定义了这些编码）。
