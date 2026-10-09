# mednext-v2 — MedNeXt-v2（ConvNeXt 式 3D U-Net + 3D GRN，大规模 CT 预训练骨干）

> 论文所读版本：v2 = arXiv:2512.17774（2025-12-19 提交；2026-10-09 核对 arXiv 摘要页，**效果数字为摘要口径，未逐表核对全文**）；架构基础 v1 = arXiv:2303.09975（MICCAI 2023，基准数字经 nnU-Net Revisited 交叉核对）；架构代码逐行核对官方仓库 [MIC-DKFZ/MedNeXt](https://github.com/MIC-DKFZ/MedNeXt) main 分支 `nnunet_mednext/network_architecture/mednextv1/{blocks,MedNextV1,create_mednext_v1}.py` 与 README（2026-10-09 抓取）。

## 论文支撑（统一四问）

- **哪篇论文用了它**：S. Roy、Y. Kirchhoff、C. Ulrich、M. Rokuss、T. Wald、F. Isensee、K. Maier-Hein（DKFZ），《MedNeXt-v2: Scaling 3D ConvNeXts for Large-Scale Supervised Representation Learning in Medical Image Segmentation》，arXiv 2025-12。https://arxiv.org/abs/2512.17774
- **目的**：回答"大规模监督预训练中，骨干本身是否是有效的表征学习器"——在 MedNeXt 上加 3D Global Response Normalization，按深度/宽度/上下文复合缩放，在约 1.8 万卷 CT 上预训练后跨基准微调。
- **效果**：v2（摘要口径）：六个 CT/MR 基准（144 结构）微调后达到 SOTA，胜过七个公开预训练模型；表征缩放对病理分割收益尤其大；预训练模态特异性在全量微调后收益可忽略（逐基准数字待通读全文后补录）。架构本身（v1）：nnU-Net Revisited 基准上唯一在全部 6 个数据集同时 ≥ 原版 nnU-Net 的方法（MedNeXt L k3：BTCV 84.70 / ACDC 92.65 / LiTS 82.14 / BraTS 91.35 / KiTS 88.25 / AMOS 89.62）；BraTS 2023/2024 多个任务冠军方案的核心组件。
- **结论**：v2 论文发现"从零训练更强 ⇒ 预训练后下游更强"可可靠预测，骨干缩放本身值得投入；代码与预训练权重经官方 nnUNet 仓库发布。v1 的边界：等训练预算下只追平 nnU-Net——架构不免费，增益要靠 k5（UpKern）、深监督、残差与更长训练换。

## 核心机制

MedNeXt 块 = 逐通道深度卷积 k³ → GroupNorm（每通道一组，等价 ConvNeXt 的通道序 LayerNorm）→ 1×1×1 扩张 ×r → GELU → **3D GRN** → 1×1×1 压缩，块内残差相加；下/上采样用步长 2 深度（转置）卷积 + 可选残差路；skip 是**相加**不是拼接；瓶颈与 3 个解码级深监督（共 5 个输出头）。v2 的块级增量是 3D GRN（ConvNeXt-V2 的 GRN 推广到体数据：逐通道空间 L2 范数 gx → nx=gx/mean(gx)+1e-6 → gamma·x·nx + beta + x，gamma/beta 零初始化即恒等映射），外加复合缩放与 18k CT 预训练。变体只动扩张比 r 与每级块数（基宽均 32）：S r=2、每级 2 块；B r=[2,3,4,4,4,4,4,3,2]、2 块；M 同 B 的 r、块数 [3,4,4,4,4,4,4,4,3]；L r=[3,4,8,8,8,8,8,4,3]、块数 [3,4,8,8,8,8,8,4,3]。

## 量级

- 参数量（本机参考实现实测，k3、grn=True）：S **5,568,450 ≈ 5.57M**（官方 README 表 5.6M，对拍一致）、L **61,969,474 ≈ 61.97M**。
- 显存/时长（nnU-Net Revisited 口径，A100）：L k3 17.3GB / 约 68h；k5 18GB / 约 233h（UpKern）——2×4090 上 k5 不现实。
- 2×4090（标称 24GB/卡）：S/B 单卡宽裕；M/L 单卡可训（AMP + 必要时梯度检查点，官方 checkpoint_style='outside_block'）；主要预算项是训练时长（约为 nnU-Net 的 3–5 倍）。

## 对比基准与选型建议

- **对比基准**：同框架（nnU-Net 配方）训练的原版 nnU-Net 与 ResEnc 预设（见 `nnunet-resenc.md`）。Revisited 复核结论：等训练预算下 MedNeXt 只追平；k5 的小幅超越部分归因于 target-spacing 选择而非架构。
- **更好的条件**：大数据集 + 肯花训练时长；挑战赛式多模型集成（BraTS 2023/2024 冠军组件）；MedNeXt-v2 预训练权重可直接微调的 CT 任务。
- **值得尝试**：① 官方 nnUNet 仓库发布 v2 预训练权重后，CT 任务"权重微调 vs ResEnc 从零训练"对比——当前最值得跟进的新东西；② 标注 >1000 卷、冲精度上限且接受 3–5× 训练时长的任务。
- **不值得**：小数据集从零训练（追平 nnU-Net 而已，训练时长反而更长）；把 MedNeXt 当"免费精度"——它赢在配方与预算，不在架构魔术。

## 短板与不适用场景

- 训练时长是主要成本（L k3 约 68h A100 口径，4090 更久）；k5/UpKern 与 2×4090 预算不兼容。
- 官方训练框架基于 nnU-Net v1（包名 `nnunet_mednext`），接 v2 训练流水有摩擦；权重经官方 nnUNet 仓库发布后此摩擦才消除。
- 官方 README 提示 v1 以 **1mm isotropic spacing** 训练（对齐 UNETR/SwinUNETR 一类）；median spacing（nnU-Net 原生偏好）未测、可能影响性能——预处理口径要对齐。
- skip 相加不拼接：高分辨率特征以加法汇入，极细小结构任务上未必优于带拼接的 U-Net 谱系（机制推断，无基准证据——以任务实测为准）。

## 经验备注

- 变体配置表见参考实现 `MEDNEXT_V2_VARIANTS`（官方 `create_mednext_v1.py` 原样）；UpKern = 用 k3 权重插值初始化 k5 网络（官方仓库提供加载器）。
- 官方 `MedNextV1.py` 的 down_0 漏传 grn 参数（其余重采样块均传）——疑为笔误；参考实现统一传，与官方对拍时注意这处不一致。
- 论文训练配置：深监督 + 块内残差 + 重采样残差全开；GRN 零初始化使 v2 初始等价 v1，迁移与对拍友好。
- 大变体训练开梯度检查点（官方 `checkpoint_style='outside_block'`）换显存。
- 参考实现：`mednext-v2.py`（自包含，CPU 自测 5 项通过：GRN 恒等性/退化输入、5 尺度深监督形状、无残差路径、梯度有限、变体参数量对拍官方表）。

## 实验反链

- 暂无（建议首验：MedNeXt-v2 预训练权重微调 vs `nnunet-resenc.md` ResEnc-M 从零训练，登记后回链 `EXP-*`）。
