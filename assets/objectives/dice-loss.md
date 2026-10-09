# dice-loss（Dice 损失）`[loss]`

## 论文支撑（统一四问）

- **哪篇论文用了它**：Milletari、Navab、Ahmadi，《V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation》（V-Net），2016，3DV。链接：https://arxiv.org/abs/1606.04797 。本次阅读版本：arXiv 摘要页 + ar5iv HTML + arXiv PDF 原文（三处公式一致；PDF 中第 3 节两个公式**无编号**）。
- **目的**：3D 医疗影像里目标解剖结构常只占体积极小部分，交叉熵（多项逻辑损失）训练易陷入偏向背景的局部极小、前景漏检；论文提出直接以 Dice 系数为优化目标，从而**无需对前景/背景样本加权**就能平衡两类体素。
- **效果**：PROMISE 2012 前列腺 MRI（训练 50 卷、测试 30 卷）：V-Net + Dice loss 平均 Dice **0.869±0.033**、Hausdorff 距离 **5.71±1.20 mm**、挑战赛得分 **82.39**；同一网络换成带样本重加权的多项逻辑损失只有 0.739±0.088、10.55±5.38 mm、63.30——Dice loss 带来约 **0.13 Dice** 的提升。分割一卷约 1 秒。
- **结论**：直接优化 Dice 的可导损失在前景/背景强烈不平衡的**二值**分割上明显优于加权交叉熵；论文场景是二分类（前景 vs 背景），多类"逐类算 Dice 取平均"是后续（如 nnU-Net 系）的扩展做法，非原文。

## 形式

论文第 3 节 "Dice loss layer"（arXiv PDF 原文无公式编号，按出现顺序转写）：

- 可导 Dice 系数（D∈[0,1]，训练中最大化；等价于最小化损失 ℒ = 1 − D）：

  D = 2·Σᵢ pᵢgᵢ / (Σᵢ pᵢ² + Σᵢ gᵢ²)，i 遍历全部 N 个体素，pᵢ 为预测概率（softmax 后前景通道），gᵢ 为真值二值体素。

- 对第 j 个体素的梯度（原文给出，供手工实现/校验）：

  ∂D/∂pⱼ = 2·[ gⱼ(Σpᵢ² + Σgᵢ²) − 2pⱼ·Σpᵢgᵢ ] / (Σpᵢ² + Σgᵢ²)²

工程落地（nnU-Net 系，软标签、非平方形式）：

  ℒ_Dice = 1 − mean_c (2·Σ p·g + smooth) / clamp(Σ p + Σ g + smooth, min=1e-8)

即分母用 Σp 而非 Σp²；p∈{0,1}（硬预测）时两者相等，软预测下不同。参考实现两种都提供（`squared_pred` 开关）。

## 适用

- 分割（尤其 3D 卷积分割）；前景占比很小的类别不平衡场景（论文动机即"anatomy of interest occupies only a very small region of the scan"）。
- 二值分割最贴合原文；多类/多器官分割用逐前景类 Dice 取平均的扩展形式。
- 当前主流落地：TotalSegmentator 所用 nnUNetTrainer 默认损失即 CE+Dice 组合（weight_ce=1, weight_dice=1），Dice 侧为 `MemoryEfficientSoftDiceLoss`（省显存版软 Dice，本机路径 `/ssd/lx/totalsegmentator/nnunetv2/training/loss/dice.py:58`，组合见 `/ssd/lx/totalsegmentator/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:399-401`）。

## 行为特征

- **直接对齐 Dice 指标**：损失降 ≈ 训练 Dice 升，训练曲线上的 loss 与 Dice 指标（如 `/ssd/lx/metrics/metrics.py` 的 `dice_test`）应同向变化；两者背离（loss 降、hard Dice 不升）是过拟合或软硬预测差异的信号。
- **收敛**：相比交叉熵，早期对前景梯度的相对权重更大，前景小目标不易被背景淹没；但绝对损失值初值偏高（接近 1），下降曲线形态与 CE 类损失不可直接比大小。
- **对小目标**：逐类平均时每个前景类权重相同，小器官的 Dice 改善能直接反映到总损失上。
- **配对观察**：Dice loss（软）↔ hard Dice 指标（二值化后）；再配 Hausdorff/HD95 看边界质量（论文同时报了 Dice 与 Hausdorff，见上 0.869 / 5.71 mm）。

## 调参经验

- **与 CE 组合**：nnU-Net/TotalSegmentator 默认 `DC_and_CE_loss(weight_ce=1, weight_dice=1)`，两者无需和为 1；单用 Dice（论文做法）在二值任务可行，多类任务 CE+Dice 更稳。
- **smooth**：nnU-Net 默认 `1e-5`（`nnUNetTrainer.py:400`），配合分母 clamp 1e-8（`dice.py:116`）；不要用 1.0 这类大 smooth（会稀释小前景的梯度，`/ssd/lx/metrics/metrics.py:67` 的 smooth=1 是**评价端**写法，损失端勿照抄）。
- **do_bg**：多类时排除背景通道再平均（nnU-Net 默认 `do_bg=False`）；二值（C=1）无此问题。
- **batch_dice**：逐样本算 Dice 取平均（`batch_dice=False`，nnU-Net 默认）vs 跨 batch 池化再算（True）；batch 内样本前景悬殊时 True 更稳，nnU-Net 按各向异性等配置自动选择。
- **优化器**（论文设置，可作二值 3D 起点参考）：mini-batch 2、momentum 0.99、lr 1e-4 每 25K 迭代除以 10。

## 坑

- **分母为零**：预测与真值都空（或全 0 概率）时 D 无定义；必须加 smooth 并对分母 clamp（nnU-Net 两处 clamp 1e-8：`dice.py:46`、`dice.py:116`），否则 NaN/Inf。
- **平方与非平方形式混用**：论文分母是 Σp²，nnU-Net 主流实现是 Σp；软预测下数值不同、梯度不同，对比实验时必须写清用的哪种（p∈{0,1} 时两者才相等）。
- **软损失与硬指标背离**：Dice loss 优化的是 softmax 概率体积重叠，评测是阈值二值化后的重叠；概率高但过 0.5 阈值的体素少时会出现 loss 降、指标不升。
- **只优化重叠、不管边界**：Dice 对预测整体偏移不敏感，边界质量要靠 Hausdorff/HD95 类指标监控。
- **梯度闭式式子易写错**：论文梯度式中分母是平方项（(Σp²+Σg²)²），手写实现务必用 autograd 对拍（参考实现自测含此校验）。
- **多类扩展非原文**：原文只处理二值前景/背景；引用"论文式子"时不要把逐类平均说成论文做法。

## 参考实现

- 本仓库：`assets/objectives/dice-loss.py`（论文形式 + nnU-Net 软形式 + 硬 Dice 指标，CPU 自测含论文梯度闭式对拍）。
- 同源模型条目：`assets/models/vnet.md` / `assets/models/vnet.py`（Dice 损失出处论文 V-Net 的网络本体参考实现）。
- 外部（本地）：`/ssd/lx/totalsegmentator/nnunetv2/training/loss/dice.py:58`（MemoryEfficientSoftDiceLoss）、`/ssd/lx/totalsegmentator/nnunetv2/training/loss/compound_losses.py`（DC_and_CE_loss）、`/ssd/lx/metrics/metrics.py:47`（hard dice 评价端）。

## 实验反链

暂无（experiments/ 目前为空）。
