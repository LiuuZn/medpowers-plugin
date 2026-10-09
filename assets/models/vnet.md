# vnet（V-Net）`[model]`

## 论文支撑（统一四问）

- **哪篇论文用了它**：Milletari、Navab、Ahmadi，《V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation》（V-Net），2016，3DV。链接：https://arxiv.org/abs/1606.04797 。本次阅读版本：arXiv v1 PDF 原文（该文唯一 arXiv 版本，2016-06-15，本地 pdftotext 抽取全文核对公式与结构）+ ar5iv HTML 对照。
- **目的**：前列腺 MRI 体数据的端到端 3D 分割；并首次提出基于 Dice 系数的目标函数，应对前景/背景体素强不平衡（"anatomy of interest occupies only a very small region of the scan"）。
- **效果**：PROMISE 2012（训练 50 卷、测试 30 卷）：V-Net + Dice loss 平均 Dice **0.869±0.033**、Hausdorff 距离 **5.71±1.20 mm**、挑战赛得分 **82.39**；同一网络换成带样本重加权的多项逻辑损失只有 0.739±0.088、10.55±5.38 mm、63.30。推理约 1 秒/卷。
- **结论**：3D 全卷积 + 逐 stage 残差 + Dice 目标端到端可行；Dice 目标"无需对不同类别样本加权即可平衡前景/背景"，显著优于加权交叉熵。适用边界：前景占比小的**二值** 3D 分割（多类/逐类平均 Dice 是后续扩展做法，非原文）。

## 核心机制

对称 V 形全卷积网络：压缩/解压两路径各若干 stage，**每个 stage 学习残差**（`out = PReLU(stage_in + convs(stage_in))`，启发自 ResNet）；下采样用 2×2×2、stride 2 的**卷积**取代 max-pooling（通道随 stage 加倍），上采样用 2×2×2、stride 2 反卷积，同分辨率左右长连接 Concat 保留细节；配合 Dice 目标直接优化前景-真值重叠度，避免交叉熵在前景占比极小时把训练带向背景偏置的局部极小。

## 结构要点（论文第 2 节 + 官方 prototxt 逐层核对）

- 主卷积一律 **5×5×5**（padding 2，"The convolutions performed in each stage use volumetric kernels having size 5×5×5 voxels"），逐 stage 通道 16→32→64→128→256；stage 内卷积个数：左路径 [1,1,2,3,3]、右路径 [3,2,1,1]（官方 prototxt 实测；论文正文仅说"每 stage 一到三层"）。1×1×1 卷积只出现在输出头（5×5×5→2 通道 + PReLU，再 1×1×1→2 通道），softmax 后前景概率 >0.5 判为前景（§3.2）。
- 输入 (B, 1, D, H, W)，D/H/W 需被 16 整除；论文固定 128×128×64（1×1×1.5 mm，N4 偏置场校正后重采样）。输出 2 通道 logits（0=背景、1=前景）。
- 训练设置（论文）：batch 2、momentum 0.99、lr 1e-4 每 25K 迭代 ÷10、约 30K 迭代（~48h）；在线增强 = 2×2×2 控制点 B-spline 随机形变（控制点位移 σ=15 体素）+ 直方图匹配；PReLU 全程。权重 msra 初始化出自官方 prototxt（`weight_filler type: msra`），论文全文未规定初始化。

## 量级

- **参数量**：论文未报告；本仓库参考实现（5×5×5 主卷积）**实测 61,982,682 ≈ 62.0 M**。改用 3×3×3 卷积的变体参数量会明显低于此数（单核权重 27/125）。
- **显存**：论文在 8 GB 显存（GTX 1080）上以 batch 2、128×128×64 完成训练；本仓库 2×4090（23 GB）跑论文配置 batch 2~4 估算富余（激活主要在浅层大分辨率 stage，建议 AMP），未在本机实测。
- 参考：本机 CPU 全网 forward+backward（输入 32×32×16）约 0.8 s（自测实测），全网 128×128×64 约为其 64 倍计算量。

## 适用

- 3D **二值**分割、小到中等数据量（论文仅 50 个训练卷 + 在线增强即可端到端训练）；目标为中等尺度结构（论文为前列腺）；需要快速推理（~1 s/卷）的端到端基线。
- "V 形残差全卷积 + Dice 目标"的组合是本条目核心价值；损失侧细节与工程落地见同源条目 `../objectives/dice-loss.md`（本论文第 3 节即 dice loss 出处）。

## 短板与不适用场景

- 输入必须固定网格（被 16 整除），需配重采样/裁剪等预处理；对极小结构（细血管、小结节点）经 16× 下采样后不友好。
- 论文形式只有 2 输出通道（二值）；多类、多器官任务更适合 nnU-Net 系（其默认损失 CE+Dice 见 `../objectives/dice-loss.md`）。
- 逐 stage 残差 + 无归一化层：对初始化/学习率敏感（官方 prototxt 靠 msra 初始化 + PReLU + 小 lr 起步；论文未规定初始化）。
- 5×5×5 卷积计算/参数开销大；预算紧张或数据极少时，先考虑更窄的变体（调低 `base_channels`）而非直接砍深度。

## 经验备注

- 本仓库参考实现：`vnet.py`（同目录，PyTorch 转写官方 Caffe 逐层配置；自测含论文 Dice 梯度闭式对拍 max|Δ|=6.94e-18，CPU 数秒通过，实测参数量即上表数值）。
- 官方实现：github.com/faustomilletari/VNet（Caffe；`Prototxt/train_noPooling_ResNet_cinque.prototxt`——split/concat 实现逐 stage 残差、msra filler、Python 层 DiceLoss）。引用其他第三方复现时须逐层核对与论文/官方配置的差异，以论文为准。
- 注意：论文 arXiv v1 原文**没有**逐体素加权 d^i 形式的 Dice 目标（原文明确 "we do not need to assign weights to samples of different classes"）；带权重的推广是后续文献做法，引用时勿算到本文头上。参考实现的 `voxel_weight` 参数为标注过的可选扩展。
- 论文的固定 128×128×64 输入与本仓库预处理流水（norm → crop3D → resample）衔接：重采样到目标分辨率后 pad/裁剪到 16 的倍数即可。

## 实验反链

暂无（experiments/ 目前为空）。
