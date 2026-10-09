# 强度归一化（模态分治：CT 全局百分位截断 + z-score；MRI/其他逐图 z-score）

类别：`[preprocessing]` ｜ 参考实现：`assets/preprocessing/intensity-normalization.py`

## 论文支撑（统一四问）

- **哪篇论文用了它**：Isensee 等（德国癌症研究中心 DKFZ），**nnU-Net**，2021，Nature Methods：<https://www.nature.com/articles/s41592-020-01008-z>。
  > **所读版本**：出版社官方免费下载的**官方补充材料全文**（Supplementary Information，`41592_2020_1008_MOESM1_ESM.pdf`，44 页，<https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-020-01008-z/MediaObjects/41592_2020_1008_MOESM1_ESM.pdf>）为第一手来源：2.2 节写明 "Z-score per image ... is a good default. We deviate from this default only for CT images, where a global normalization scheme is determined based on the intensities found in foreground voxels across all training cases."；3.2 节 LiTS walk-through 的 "Intensity normalization" 段写明前景体素强度跨训练例汇总后取 mean/std 与 0.5/99.5 百分位；SN6.7（LiTS）只含实例值：Clip to [−17, 201]、减 99.40、除 39.39（并注明 "131 training CT images"）。交叉核对：同团队方法论文全文 **arXiv:1904.08128**（ar5iv HTML：<https://ar5iv.labs.arxiv.org/html/1904.08128>）与出版社官方页摘要（正文付费墙，仅摘要可读），并与官方开源实现 MIC-DKFZ/nnUNet 及本地旧流水逐条核对。
- **目的**：消除不同设备/采集协议造成的**强度尺度不一致**，作为 nnU-Net 自动预处理（裁剪→归一化→重采样）的默认一环，让同一套网络配置能跨数据集工作。
- **效果**：作为 nnU-Net 默认预处理步骤，nnU-Net 无需人工干预即在 **23 个公共数据集**（国际医学图像分割挑战赛）上超过大多数现有方法（含高度专用的参赛方案）——摘要原文称 "most existing approaches, including highly specialized solutions"。
- **结论**：归一化规则按模态分治是论文明确设计：**CT** 用训练集**全局前景统计**——0.5/99.5 百分位截断 + 全局前景 mean/std 做 z-score，所有图像用同一组数；**其余模态（MRI 等）默认逐图 z-score**（每张图减自己的均值、除自己的标准差）；裁剪后尺寸缩小 ≥25% 时创建非零体素 mask，统计只在 mask 内计算，且**归一化只作用于 mask 内体素**（论文原句 "the normalization is applied within that mask only"）。

## 动机

医学影像没有统一的强度标度：MRI 的灰度值依赖场强、线圈与序列参数，同一种组织在不同机器上分布完全不同；CT 虽有 HU 定量标度，但扫描视野外的空气、床板、金属伪影等离群值会主导逐图统计。逐图对 CT 归一会破坏 HU 的物理一致性（同组织在两台机器上归一化后数值不同），而 MRI 则恰恰相反——必须逐图归一，因为不存在可跨机共享的全局标度。nnU-Net 的模态分治正是对这两种情况的分别处理。

## 适用

- **模态**：CT（全局截断 + z-score，需要训练集统计与标注/前景 mask）；MRI 及其他模态（逐图 z-score，无需标注）。
- **2D/3D**：皆可。统计在体数据体素集合上做，与维度无关；numpy 布局 2D=(B,C,H,W)、3D=(B,C,D,H,W) 对应张量输入，预处理脚本通常按 (D,H,W) 逐例处理。
- **数据量级**：MRI 的逐图 z-score 不依赖数据量，小数据（几十例）即可用；CT 的全局统计需要跨全部训练例汇总，数据太少时统计本身不稳，建议至少几十例。
- **不适用**：强度本身就是目标信号的任务（如定量 MRI、PET 摄取量化）要慎用 z-score——归一化后物理单位丢失。

## 参数与经验值

- **百分位 0.5 / 99.5**（CT 截断界，全局前景分布）：nnU-Net 论文取值。截断界过宽 → 离群值（空气、金属）拉大方差，组织对比被压扁；过窄 → 组织间对比度被削掉，小病灶可能被截没。
- **eps = 1e-8**（std 除法防零除）：本地旧流水同值；医学影像归一化后的 std 通常 O(1)，1e-8 远小于它，只防退化情形（全零/常数图）不改变正常结果。
- **mask 触发条件与作用范围**：论文写"裁剪使平均尺寸缩小 ≥25% 时，创建非零体素 mask，归一化只作用于 mask 内"（"the normalization is applied within that mask only"）。不满足则 MRI 直接全图统计。mask 外体素论文未规定：官方 v2 保持原值（`default_normalization_schemes.py` 的 `ZScoreNormalization` 只赋值 `image[mask]`），官方 v1 置 0（`preprocessing.py` 的 `data[c][mask == 0] = 0`）——复现谁就用谁，别混。
- **统计范围（CT）**：论文原文写 foreground voxels（属任意前景类的体素）跨全部训练例汇总后取百分位/mean/std；官方 v1/v2 实现同为前景口径（本地 nnunetv2 `fingerprint_extractor.py:58` `foreground_mask = segmentation[0] > 0`）。前景口径需要标注；无标注时的整图统计不是论文/官方口径，只能作退化扩展。
- **论文实例（补充材料 SN6.7，LiTS）**：Clip to [−17, 201]，减 99.40，除 39.39——截断界、均值、标准差都来自 131 例训练数据的前景（肝+肝肿瘤）体素汇总，可作自实现的对数参考。

## 实现参考

- 旧流水：`/ssd/lx/body_analysis/preprocessed/prepro_0_norm.py:17-42`（CT：前景体素汇总 → [0.5, 99.5] 百分位截断 → 前景 mean/std z-score，std 加 1e-8；其他模态：全图 z-score 加 1e-8；统计量存 prepro json 的 `forground value` 字段供推理复用）。对应流水位置 `prepro_0_norm → prepro_1_crop3D → prepro_2_resample3_2D` 的第一步。
- 本仓库参考实现：`assets/preprocessing/intensity-normalization.py`（按论文：CT 全局前景百分位截断 + 全局 z-score；非 CT 逐图 z-score；mask 变体只改 mask 内，`zero_outside_mask` 参数切换 v1 置 0 / v2 保持原值两种 mask 外口径）。
- 官方开源实现：MIC-DKFZ/nnUNet。v2（本地 `/ssd/lx/totalsegmentator/nnunetv2/`）：CT 前景统计见 `experiment_planning/dataset_fingerprint/fingerprint_extractor.py:58,62`（`foreground_mask = segmentation[0] > 0`）；归一化见 `preprocessing/normalization/default_normalization_schemes.py`（`CTNormalization` 截断+全局 z-score、`ZScoreNormalization` 只改 `image[mask]`）。v1（GitHub nnunetv1 分支 `nnunet/preprocessing/preprocessing.py`）：mask 内 z-score + mask 外置 0。

## 坑

- **训练/推理不一致**：CT 的截断界与 mean/std 是训练集统计量，必须持久化（如 JSON）并在推理时复用同一组数；用推理图自己的统计会破坏跨例一致性。本地旧流水就为此把 `forground value` 写进 json。
- **CT 前景统计依赖标注**：论文与官方实现（v1/v2）都是前景（标注 >0）口径，没有标注就算不了；退化用整图统计（如本仓库实现的 `masks=None` 扩展）不是论文口径，分布会偏向空气等背景体素，须在卡片/实验记录里写明并保证训练/推理一致。
- **std=0 退化**：纯空气/常数切片逐图 z-score 会除零 → NaN/Inf，除法必须加 eps。
- **物理单位丢失**：z-score 后 HU 值不再有意义，窗宽窗位、HU 阈值类的后处理/增强（如肺窗）必须在归一化**前**做。
- **流水顺序要固定**：标准顺序是裁剪到非零区域 → 归一化 → 重采样；归一化统计受裁剪影响（论文：先 crop to nonzero），重采样插值也会改分布，中途插入或调序都会让统计失效。
- **标签不受归一化影响**（逐体素单调变换），但若先重采样后归一化，注意插值 overshoot 不能超出截断界之外被误截——本流水先归一化后重采样无此问题。

## 实验反链

暂无（`experiments/` 目前为空）。
