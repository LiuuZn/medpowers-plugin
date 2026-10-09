# nnunet-resenc — nnU-Net ResEnc 预设（残差编码器 U-Net：M/L 计算分级，官方推荐新默认）

> 论文所读版本：arXiv:2404.09556（MICCAI 2024，《nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image Segmentation》，2026-10-09 核对 arXiv HTML 全文基准数字）；预设使用口径另核对官方文档 [resenc_presets.md](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md)（main 分支，2026-10-09 抓取）与本机 nnunetv2 2.7.0、dynamic_network_architectures 源码（`/ssd/lx/totalsegmentator/.venv`）。以下 Dice 均为 5 折交叉验证 DSC。

## 论文支撑（统一四问）

- **哪篇论文用了它**：F. Isensee、T. Wald、C. Ulrich、M. Rokuss 等（德国癌症研究中心 DKFZ），《nnU-Net Revisited》，MICCAI 2024。https://arxiv.org/abs/2404.09556
- **目的**：论文一面系统指出 2022–2024 年"超越 nnU-Net"文献中的验证缺陷（弱基线、不公平训练配方、小数据集、忽略算力开销），一面把 nnU-Net 自身推向现代 GPU 的算力上限——编码器换成残差块，按显存预算给出 ResEnc-M/L/XL 三个预设。
- **效果**：对同时训练的原版 nnU-Net：KiTS 86.04→86.79(M)/88.17(L)/88.67(XL)；AMOS 88.64→88.77/89.41/89.68；LiTS 80.09→80.75/81.60/81.19——大数据集增益明显；小数据集（BTCV/ACDC/BraTS）增益≈0。代价（A100 40GB 单卡）：M 9.1GB/约 12h、L 22.7GB/约 35h、XL 36.6GB/约 66h。
- **结论**：验证严谨的前提下，配置得当的 CNN U-Net 仍优于被复核的多数新架构；官方文档明确 "We recommend nnU-Net ResEnc L as the new default nnU-Net configuration"，并要求新方法与同显存/算力预算的 ResEnc 变体对比。

## 核心机制

两层改动：①**网络**——编码器每级换成 2 个 ResNet-D 式残差块（conv(步长)→IN→LeakyReLU ×2，短路 = AvgPool→1×1×1 卷积→IN，stem 为 1 个普通卷积），解码器保持普通卷积级联；②**配方**——预设不是固定拓扑，而是把"显存目标"（M=8/L=24/XL=40 GB 规划值）交给 nnU-Net planner，按数据集 spacing 自动算 patch、各向异性步长、通道与批大小，并取消"单批 ≤ 数据集 5%"的旧上限。预处理、增广、训练配方、滑窗推理与原版 nnU-Net 完全一致——增益只来自网络容量与显存利用率的放大。

## 量级

- 参数量：随数据集规划变化；参考实现按 planner 示例的典型 3D fullres 配置（6 级、通道 (32,64,128,256,320,320)、每级 2 残差块、解码每级 1 卷积）实测 **41,992,490 ≈ 42.0M**。
- 显存/时长（官方 A100 40GB 口径）：M 9.1GB / 约 12h；L 22.7GB / 约 35h；XL 36.6GB / 约 66h。
- 2×4090（标称 24GB/卡）：M 轻松（官方推荐 9–11GB 卡即可）；L 官方要求 24GB 卡，4090 可跑（紧张时框架自动缩 patch），是本仓库预算下的事实上限配置；XL 单卡装不下。

## 对比基准与选型建议

- **对比基准**：同一框架、同一预处理与增广下同时训练的原版 nnU-Net（PlainConv U-Net）。官方文档还给出对比纪律：任何新方法应与"匹配自己显存/算力预算的 ResEnc 变体"比——本卡即 2×4090 实验室的对照片。
- **更好的条件**：大数据集（KiTS/AMOS/LiTS 级、数百卷以上）+ 大显存预算，增益 +0.8～2.6 DSC。
- **无增益条件**：小数据集（BTCV/ACDC/BraTS 级、数十卷）——额外容量消化不了，残差编码器无钱可赚。
- **值得尝试**：一切 3D 分割任务的第一基线——先默认配置跑通，再换 ResEnc-M/L 获取"零创新成本"的增益；拿到 40GB 级卡才考虑 XL。
- **不值得**：数据 <100 卷时为 L 的训练时长买单（约 35h 对默认 <10h）；该预算投给增广/伪标签收益更高。

## 短板与不适用场景

- 预设本质是"配方"不是"一个网络"：脱离 nnU-Net 框架（自动规划 + 训练配方 + 滑窗推理）拿单网络复刻不构成 ResEnc 基线——参考实现只服务架构理解与外挂管线。
- 训练时长约为默认配置的 3 倍以上（L），推理显存同量级上涨。
- 小数据集无增益；XL 与 2×4090 预算不兼容。
- 各向异性数据上 patch/步长由 planner 决定；用 `overwrite_target_spacing` 覆盖 spacing 会连带改变拓扑——调参前先读 plans JSON。

## 经验备注

- 调用：`nnUNetv2_plan_and_preprocess -d DATASET -pl nnUNetPlannerResEncM/L/XL`（已有预处理则用 `nnUNetv2_plan_experiment`）；训练/推理加 `-p nnUNetResEncUNetM/L/XLPlans`。
- 自定义显存目标：`-gpu_memory_target <GB>` 且必须配 `-overwrite_plans_name`（官方指引）；结果进独立目录，不覆盖标准 plans。
- 多卡：官方建议按单卡显存规划（如 L=24GB），再在 plans 里手动放大 batch（batch 2 × 8 卡 = 16），不做模型并行。
- TotalSegmentator v2 的官方权重即 ResEnc-L（104 结构整体 Dice 0.943，训练约 7 天缩至 1.25 天）——CT 解剖分割的现役参照点。
- 参考实现：`nnunet-resenc.py`（自包含 ResidualEncoderUNet3D：ResNet-D 残差编码器 + 普通解码器 + 深监督 + He 初始化/残差支路末级 norm 置零，CPU 自测 6 项通过）。

## 实验反链

- 暂无（作为基线的对照实验登记后回链 `EXP-*`）。
