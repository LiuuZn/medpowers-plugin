# 前景裁剪（非零区域包围盒裁剪 foreground crop）

类别：`[preprocessing]` ｜ 参考实现：`assets/preprocessing/foreground-crop.py`

## 论文支撑（统一四问）

- **哪篇论文用了它**：Isensee, F., Jaeger, P.F., Kohl, S.A.A. 等（德国癌症研究中心 DKFZ），**nnU-Net**，2021，Nature Methods 18:203–211：<https://www.nature.com/articles/s41592-020-01008-z>。
  > **所读版本**：出版社官方页正文付费墙，本次读到摘要与官方补充材料 PDF（Supplementary Notes 1–9，44 页：<https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-020-01008-z/MediaObjects/41592_2020_1008_MOESM1_ESM.pdf>）；裁剪的正文细节读自同团队 nnU-Net 方法论文全文 **arXiv:1904.08128**（ar5iv HTML：<https://ar5iv.labs.arxiv.org/html/1904.08128>），并与官方开源实现 MIC-DKFZ/nnUNet、本地旧流水交叉核对。
- **目的**：把每例图像裁到**前景（非零体素）的最小包围盒**，去掉大片值为 0 的空气背景。显存开销正比于体素数，裁剪直接降低显存占用、在同等显存预算下等效提升有效分辨率/可用 patch 尺寸；它还是 nnU-Net 预处理**第一步**，后续数据指纹（spacing、强度统计）与 patch size 决策都建立在裁剪后的数据上。
- **效果**：论文未单列裁剪消融。原文："As a first processing step, nnU-Net crops the provided training cases to their nonzero region. While this had no effect on most datasets in our experiments, it reduced the image size of brain datasets such as D1 (Brain Tumor) and D15 (MSLes) substantially and thus improved computational efficiency."（对脑部数据集图像尺寸缩小"substantially"）。量化联动规则："If cropping resulted in an average size decrease of 25% or more, a mask for central non-zero voxels is created and the normalization is applied within that mask only."（平均缩小 ≥25% 时归一化只在非零 mask 内统计）。含此步骤的整条流水在 **19 项**国际竞赛超过大多数专用流水线、**49 项**任务多数刷新 SOTA（Results：29/49 目标结构）。
- **结论**：裁剪是 nnU-Net 固定的流水线首步，**无超参数**、全模态默认启用；前景判定规则就是"非零体素"（官方实现逐通道 OR 后 `binary_fill_holes`）。论文同时说明它对多数数据集无影响——价值集中在背景占比大的数据（脑部 MRI、大 FOV CT）。

## 动机

CT/MRI 原始影像里目标只占视野一小部分，四周大片空气/零背景不携带信息却占据体素：卷积与插值的显存、时间开销都按全图计。裁到前景包围盒后：① 同一张卡的显存预算下能放下更大 patch 或更大 batch；② 后续重采样的目标形状、进而 patch size（本地 `net_arch/arch_Unet3_2D.py:110` 从预处理 json 读 `target patient shape final` 当 patch_size 用）都在更紧凑的域上计算；③ 数据指纹的强度统计不被空气拖偏（与 25% 规则联动，见 [intensity-normalization.md](intensity-normalization.md)）。

## 适用

- **模态**：CT/MRI 均默认启用；MRI 个别序列背景有非零噪声，见"坑"第 1 条。
- **2D/3D**：皆可。本仓库 2.5D 按 3D 存（data-mgmt 约定），裁剪在 3D 体数据上做，再重采样到 2D 切片。
- **数据量级**：逐例进行、无需跨例统计，任何量级可用；数据集越小、显存越紧，裁剪的等效收益越大。
- **不适用/慎用**：背景非零且承载信号（伪影、体表标记、文字水印都会把包围盒撑大）；前景判定失效的场景见"坑"。

## 参数与经验值

- **前景阈值 0**（`data != 0`）：论文的 nonzero region + 官方 nnUNetv2 实现（`create_nonzero_mask` 逐通道 OR）。注意这只认"值为 0"的背景——CT 空气是 −1000 HU 非零（见"坑"第 1 条）。
- **`binary_fill_holes`**：官方实现细节。对包围盒本身**无影响**（填洞不改变极值），保留它是因为同一 mask 被复用于归一化统计域（对应论文 25% 规则）与训练采样。
- **包围盒半开区间 `[start, end)`**：官方 acvl_utils/nnUNetv2 约定，`end = 非零最大索引 + 1`，python 切片 `data[s:e]` 直接可用。
- **25%**：论文联动阈值——裁剪平均缩减 ≥25% 时，归一化只在非零 mask 内统计；`crop_volume_reduction()` 返回该比例。
- **nonzero_label=-1**（可选，官方代码默认行为）：裁剪域内、非零掩码之外值为 0 的 seg 体素标为 −1（官方只改写 (seg==0) 且在掩码外的体素，掩码外原有非零 seg 值保留），配合训练采样；论文未提，本仓库参考实现默认关闭。

## 实现参考

- 旧流水：`/ssd/lx/body_analysis/preprocessed/prepro_1_crop3D.py:87-141`（`_create_nonzero_mask`：`data != 0` + `binary_fill_holes`；`_get_bbox_from_mask`：逐轴 min/max+1；`_crop_to_bbox`：切片）。对应 `prepro_0_norm → prepro_1_crop3D → prepro_2_resample3_2D` 流水中的第二步；`crop_image_test` 返回 bbox、`restore_mask_test` 用 bbox 把预测掩码贴回原坐标系（:149-176, :273-296）。
- 官方开源实现：MIC-DKFZ/nnUNet（nnunetv2/preprocessing/cropping/cropping.py 的 `create_nonzero_mask`/`crop_to_nonzero`；bbox 约定来自 acvl_utils bounding_boxes 的半开区间）。nnUNetv2 预处理实际顺序是 **crop → normalize → resample**（default_preprocessor.py:65-91，注释明确 normalization 必须先于 resample）。
- 本仓库参考实现：`assets/preprocessing/foreground-crop.py`（与论文对齐的默认口径 + nonzero_label 可选开关；`create_nonzero_mask` 供归一化条目复用）。

## 坑

- **前景判定只认"值为 0"的背景**：① MRI 个别序列背景有非零噪声，一个噪点就能把包围盒撑到全图（参考实现自测用例 7 复现）；② CT 空气 −1000 HU 非零，此启发式只能裁掉 FOV 外写 0 的区域（DICOM→NIfTI 导出常见），体内空气裁不掉；若整图无 0 体素则裁剪退化为恒等。论文只提 nonzero 启发式、未讨论失效模式——对 MRI 新数据先做背景去噪/阈值化，或改用标注联合判定前景。
- **bbox 必须持久化**：裁剪改变坐标系，推理预测要贴回原图就得记录 `shape_before_cropping` 与 bbox（nnUNetv2 写进 properties，旧流水写进返回值/推理链路）。训练用了裁剪、推理不还原是最常见的事故。
- **流水顺序定死一种**：论文裁剪是第一步；官方代码 crop → normalize → resample（normalize 统计域依赖裁剪结果，且必须先于 resample，否则非零 mask 与重采样图错位）；本地旧流水是 norm → crop → resample。同一条流水内训练与推理必须一致，混序会让强度统计与 mask 失效。
- **裁剪不代替重采样**：裁剪不改变 spacing，只去冗余体素；它影响的是重采样目标形状与 patch size 决策，两者是串联的两步。
- **2D 多通道布局歧义**：ndim=3 的数组既可能是 3D 体数据 (Z,Y,X) 也可能是多通道 2D (C,H,W)，参考实现用 `spatial_dims` 参数显式消歧（默认按 3D 解释）。

## 实验反链

暂无（`experiments/` 目前为空）。
