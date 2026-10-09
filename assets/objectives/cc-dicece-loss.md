# cc-dicece-loss —— 基于连通域实例级 CC-Metrics 的 Dice+CE 复合损失

## 论文支撑（统一四问）

- **哪篇论文用了它**：Luc Bouteille、Alexander Jaus、Jens Kleesiek、Rainer Stiefelhagen、Lukas Heine（德国 TIO/IKIM 等团队），《Learning to Look Closer: A New Instance-Wise Loss for Small Cerebral Lesion Segmentation》，IEEE ISBI 2026，https://arxiv.org/abs/2511.17146 。本卡依据 arXiv **v3 全文**（2026-04-06）与官方仓库 TIO-IKIM/Learning-to-Look-Closer 代码（2026-10-09 复核）整理。
- **目的**：脑内小病灶（腔隙、微出血、脑转移等）体积占比可忽略，全局 Dice/DiceCE 中单个病灶的贡献趋零，网络系统性漏检小病灶。论文在 CC-Metrics 框架上构造实例级损失 CC-DiceCE：对真值每个连通域（实例）取 Voronoi 区域，逐实例计算 Dice 与 CE，让**每个真实病灶在损失中权重相等**、与体积无关。
- **效果**：5 个脑 MRI 数据集 5 折交叉验证（nnU-Net 框架，CC-DiceCE 对 DiceCE 基线）：instance recall LAC 0.3211→0.3698、CMB 0.5417→0.5780、SBM 0.7994→0.8115、WMH 0.6564→0.6432（微降）、BraTS 0.4513→0.5286；全局 Dice 几乎不变（最差 LAC −0.0113）；recall 增益跨病灶大小四分位（不只最小四分位受益，BraTS 最小四分位 0.0057→0.0767）。与同期的 blob loss 相比，25 个"指标-数据集"组合中 22 个更优。
- **结论**：CC-DiceCE 提升 detection（recall）而分割质量基本不动；precision 代价随数据集而异——BraTS 上 precision 0.7708→0.0517，小标注伪影被逐实例项放大。论文实验将本损失与全局 DiceCE **1:1 相加**使用；病灶定义模糊的数据（WMH）需缩短训练日程（1000→150 epoch）抑制过拟合。

## 类别标签

`[loss]`（可导损失；真值侧的连通域标记与 Voronoi 划分是离散预处理、不参与梯度，预测侧 softmax→统计路径全程自动微分）

## 形式

真值前景 K 的 26 连通分量集合 𝒦，对每个分量 C（论文第 II 节）：

```
Voronoi 区域：  R_C = { t ∈ T : d(t,C) < d(t,C′), ∀ C′ ∈ 𝒦∖{C} }   （平局任意取向）
CC-Metrics：    V_m(P,K) = (1/|𝒦|) · Σ_{C∈𝒦} m(P ∩ R_C, C)          （逐实例等权平均）
```

取 m 为 DiceCE、指标变损失，即逐实例项（对应官方 `VectorizedCCLoss` mode="dice_ce"）：

```
Dice_C  = 1 − 2·Σ_{R_C}(p·g) / ( Σ_{R_C} p + Σ_{R_C} g )      （分母下限 ε=1e-7）
CE_C    = (1/|R_C|) · Σ_{R_C} −log p_true
CC-DiceCE = (1/B) Σ_b (1/|𝒦_b|) Σ_{C∈𝒦_b} [ Dice_C + CE_C ]
```

关键性质：真值某实例被完全漏检时，其 Voronoi 区域内交集 Σ(p·g)≈0，该实例 Dice 项 = 1（最差值），以 1/|𝒦| 权重直接计入；而单个假阳体素只污染它落在的那一格 Voronoi 区域——**结构性地偏置 recall**。

## 适用

- 3D 二类分割（背景+前景双通道 softmax）；实例可定义为连通域的病灶，尤其**多发小病灶**（微出血、腔隙、小转移瘤）。
- 漏检是主要矛盾、且单实例体积占全局比例可忽略（全局 Dice/CE 梯度几乎推不动新病灶）的场景。
- nnU-Net 用户可直接用官方训练器（`nnUNetTrainerGlobalCCDiceCE` 等）；本仓库参考实现剥离了 nnU-Net 管线依赖，供自研管线取数。

## 行为特征

- 漏检强惩罚、假阳弱惩罚（见"形式"）：参考实现自测中，漏检一个 r=2 小实例使 CC 损失 +0.304，而同期全局 DiceCE 仅 +0.021（约 14.7 倍差距）——这是实例级与全局损失的核心行为差异。
- recall 增益不限于最小病灶：论文中增益跨大小四分位；全局 Dice 曲线与基线几乎重合、instance recall 曲线抬升是本损失的典型训练曲线特征。
- CE 项按 Voronoi 区域求均值：区域内的背景体素也参与记账，假阳只有落在某实例"领地"内才被该实例的 Dice/CE 惩罚；区域划分完全由真值决定（每步随真值重算）。
- 观察指标必须配 **instance recall / CC-Dice 型逐实例指标**；只盯全局 Dice 会完全看不到本损失的收益（全局 Dice 几乎不变正是设计目标）。

## 调参经验

- 论文用法：本损失与全局 DiceCE **1:1 加权相加**（`LossMixer`，权重各 1）；官方同时提供纯实例损失训练器（`nnUNetTrainerCCDiceCE`）。
- 组合里的全局 Dice 用**非平滑版（ε=0）**：默认平滑 Dice 在 CMB 上训练不稳（论文注明）。
- 实例数多、单实例极小的数据最受益；实例少而大时逐实例平均退化为接近全局 Dice，收益减小。
- 训练日程：WMH 上默认 1000 epoch 过拟合，150 epoch 反而 CC-Dice 更高（0.535 vs 0.496）、Dice 0.792 vs 0.799——小数据/病灶定义模糊时先缩日程。
- 2×RTX 4090 显存预算：本损失的 Voronoi/连通域计算在真值侧（scipy CPU 或官方 CuPy GPU），不占训练图显存；AMP 下概率统计保持 fp32（官方强制 probs→float32，本实现保留输入 dtype）。

## 坑

- **precision 代价与标注伪影放大**：BraTS precision 0.7708→0.0517——标注中的小伪影/碎屑连通域被逐实例项当成"必须检出"的目标，网络被迫大量误报。数据集标注含伪影时，先按体积阈值清洗连通域再训练。
- 非普适提升：WMH 全局 recall 微降（0.6564→0.6432）；病灶定义模糊/边界不清的数据要同时盯全局指标，防"实例 recall 涨、整体变差"。
- 各向异性 spacing：官方与参考实现的 EDT 均不带 sampling（各向同性体素假设），Voronoi 划分在各向异性体数据上会偏向某轴；必要时先重采样或给 `distance_transform_edt` 传 sampling（官方未做，改动需自担）。
- 实例 = **26 连通域**，不是解剖学病灶数：贴邻的两个病灶并为一个实例，距离变换的分片也随连通域数变化。
- 仅支持二类（恰 2 通道 softmax）；多类需逐类二值化后改造。输入 logits/概率语义见 .py 文件头（默认 logits，同官方）。
- 批内混入真值全空样本时该样本走 fallback（Dice 项常数 1 + 全图 CE 均值），会稀释其余样本的实例级梯度；nnU-Net 采样器基本不产生全空批，自研采样需注意。

## 实验（EXP-*）反链

暂无（experiments/ 目前为空；后续实验启用本损失时在此处互加反链）。

## 参考实现

`assets/objectives/cc-dicece-loss.py`（同目录）：公式逐条对应论文第 II 节与官方 `VectorizedCCLoss`（TIO-IKIM/Learning-to-Look-Closer，nnunetv2/training/loss/instance_losses.py + nnunetv2/utilities/connected_components.py）。**数值等价对拍**（2026-10-09，官方 GPU+CuPy 全量路径 vs 本实现，B=4×32³ 多实例+空真值混合批）：三种 mode 损失最大绝对差 ≤5e-07（相对差同量级，即 float32 舍入水平；dice 与 dice_ce 模式逐位一致），dice_ce 梯度逐元素最大绝对差 ≤3e-11（梯度量级 1.8e-05 的 10⁻⁶），Voronoi 划分逐体素 0 不一致——残余差异源于官方 `scatter_add`（GPU float32 累加顺序）与本实现 `index_add` 的浮点误差，可解释。与官方/论文的不一致处（scipy CPU 替代 CuPy CUDA、target dtype 放宽、输入 dtype 保留、仅 3D 二类、剥离 nnU-Net 训练器组合）见 .py 文件头编号清单。
