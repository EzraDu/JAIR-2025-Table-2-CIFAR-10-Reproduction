# JAIR 2025 Table 2（CIFAR-10）复现

复现 Yu 等，*Detecting Generative Model Inversion Attacks*（JAIR 2025）的 Table 2。受害者模型是 CIFAR-10 上的 WRN-40-2。表中数字是检出率：被判为反演图的样本比例。

所有反演图都由 [CMI 仓库](https://github.com/zju-vipa/CMI) 的源码生成，没有修改该仓库。

## 结果

原始数字在 [results/table2.csv](results/table2.csv)。

| 测试对象 | 论文 (%) | 本次 (%) | 张数 |
| --- | ---: | ---: | ---: |
| CIFAR-10 测试集（误报） | 4.74 | 8.60 | 10000 |
| CMI | 94.48 | 90.97 | 10240 |
| DAFL | 95.95 | 99.55 | 10240 |
| DeepInv | 79.98 | 73.37 | 10240 |
| DFQ | 97.25 | 80.48 | 10240 |
| IPR | 98.26 | 95.67 | 10240 |
| ZSKT | 88.77 | 91.05 | 10240 |
| 六种反演平均 | 92.45±6.35 | 88.52±8.94 | — |

误报用完整 CIFAR-10 测试集。六种反演各 10240 张。平均和标准差只算这六种反演，标准差是六个数的总体标准差（平方和除以 6 再开方）。留出集准确率 0.9181（训练 24576 张，留出 6144 张）。

## 反演

入口是 CMI 的 `datafree_kd.py`，系数与仓库里 `scripts/<方法>/*_wrn402_wrn161.sh` 一致。学生网络是 WRN-16-1，教师是 WRN-40-2。

| 方法 | 做法 | 学生最高准确率 |
| --- | --- | ---: |
| CMI | `CMISynthesizer` | — |
| DAFL | 官方 dafl 脚本，200 轮，保留最后 10240 张 | 66.98% |
| DFQ | 官方 dfq 脚本，200 轮，保留最后 10240 张 | 85.42% |
| DeepInv | 官方 deepinv 脚本，40 轮；每张 1000 步生成器更新 | 61.36% |
| IPR | `CMISynthesizer`（scratch 系数，40 轮）合成后，用同仓库 `vanilla_kd.py` 蒸馏 WRN-16-1（`T=1`） | 合成 77.94%；蒸馏后 71.52% |
| ZSKT | 官方 zskt 脚本，`--epochs 200`。取第 101 轮合成调用的第 40–79 步，共 10240 张，用已经训好的 Table 2 SVM 打分 | 该轮 65.48%（学习率 0.0492） |

Windows 上 CMI 内部 DataLoader 的 `num_workers=4` 会在第一批之后卡死。采集进程里把它设为 0，CMI 源码不动。

## 检测器

PCA 不重拟合。

- 受害者特征：WRN-40-2 的 `ReLU(BN(block3))`，128×8×8，展平 8192 维，用已有的 `[['cmi', 'ipr']]relu_512` PCA 降到 512 维（拟合样本数 126400）。
- 图像特征：ImageNet ResNet18 去掉全连接层后的 512 维。输入用 CIFAR-10 的均值和方差，不用 ImageNet 归一化。
- 拼成 1024 维，`SVC(kernel="rbf", class_weight="balanced")`。
- 训练池：CIFAR-10 训练集、CMI、IPR 各 10240 张，种子 0 做 80/20。没有使用 `--match-single`。

PCA 经 sklearnex 读出后 `transform` 会缺 `n_jobs`。`reproduce_table2.py` 在导入 `function` 之前改用 scikit-learn 原版 `transform`。

`reproduce_table2.py` 需要和`function.py`、`registry.py` 放在同一目录运行，并从该目录读取 `Invdata` 与 `checkpoints`。评 ZSKT 时不要重跑，否则会重新拟合 SVM。ZSKT 的 10240 张由 `zskt_score_epoch101.py` 用已经保存的 SVM 打分。

## 脚本

运行前设置 `CMI_ROOT`（CMI 仓库根目录）和 `INVDET_ROOT`（检测代码根目录）。未设置时按相对路径 `CMI`、`invdet` 查找。

| 文件 | 作用 |
| --- | --- |
| [scripts/reproduce_table2.py](scripts/reproduce_table2.py) | 训练 Table 2 的 SVM，并评 noInv、CMI、DAFL、DeepInv、DFQ、IPR、ZSKT。 |
| [scripts/collect_cmi_methods.py](scripts/collect_cmi_methods.py) | 用 CMI 源码收集 DAFL、DFQ、DeepInv、IPR。 |
| [scripts/zskt_collect_epoch101.py](scripts/zskt_collect_epoch101.py) | 用 CMI 的 zskt 脚本跑到第 101 轮，保存该轮合成图后停止，并还原学生权重。 |
| [scripts/zskt_score_epoch101.py](scripts/zskt_score_epoch101.py) | 取第 101 轮第 40–79 步共 10240 张，用已有 SVM 打分。不重拟合 PCA 或 SVM。 |
| [scripts/zskt_epoch101_run.py](scripts/zskt_epoch101_run.py) | 先采集第 101 轮，再打分。 |

环境：Python 3.11，torch 2.11+cu128，torchvision 0.26。
