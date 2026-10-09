# resample-isotropic — 体数据重采样到目标（近各向同性）spacing

类别：`preprocessing`　|　参考实现：同目录 [resample-isotropic.py](./resample-isotropic.py)

## 论文支撑（统一四问）

- **哪篇论文用了它**：Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation*. Nature Methods 18, 203–211 (2021)，简称 nnU-Net。链接：https://www.nature.com/articles/s41592-020-01008-z
  所读版本说明：Nature 正文付费墙（官方页仅可读摘要）；本文引用的规则全部出自作者提供的官方补充材料 **Supplementary Notes 1–9**（Springer 静态资源公开 PDF，含 MOESM1，"In the format provided by the authors and unedited"），主要出处为 Supplementary Note 2.2 "nnU-Net design principles — Rule-based parameters: Target spacing and resampling"（p.29）与 Supplementary Note 3.1 ACDC 管线分析（p.31）。
- **目的**：训练集内各病例体素 spacing（体素物理尺寸 mm）不一致，且常为各向异性（如 CT 面内 0.7 mm、层间 5 mm）。把每个病例重采样到由数据集推断的统一目标 spacing，使 3D 卷积核在三个轴上感受野物理尺度均匀，patch 长宽比可以跟随中位形状、网络深度与池化按统一规则自动配置。
- **效果**：作为 nnU-Net 自动配置预处理（crop → resample → 归一化）的核心一环，nnU-Net 无人工干预即在 23 个国际分割挑战公开数据集上超过绝大多数已有方法（含高度专用方案）。ACDC 数例：cine MRI 中位 spacing 10×1.56×1.56 mm，out-of-plane 轴目标取 10th 百分位 = 5 mm，重采样后中位形状 18×237×208（论文 Fig. SN3.1 / Suppl. Note 3.1）。
- **结论**：各向同性数据目标 spacing 取训练集**逐轴中位数**；各向异性数据在 out-of-plane 轴取该轴 spacing 的 **10th 百分位**（比中位更细，宁可选"下采样训练、上采样导出"，避免把稀疏层上采样产生插值伪影），且该轴重采样用最近邻。适用于体数据的 3D 与 2.5D（3D 存 2D）预处理流水。

## 动机

- 各向异性 spacing 下，固定的 3×3×3 卷积核沿粗轴跨过的物理距离是细轴的数倍：跨层聚合把相距甚远的层当作相邻信息，不聚合又浪费 3D 结构；重采样到近各向同性 spacing 后卷积/池化行为可预测。
- 不统一 spacing 则 patch 的物理覆盖随病例漂移，同一超参在不同病例上等效感受野不一致，训练与推理行为不稳定。
- 论文的配套设计（Suppl. Note 2.2）：重采样后中位形状决定 patch 长宽比；各向异性数据初期用 (1×3×3) 等效 2D 核、只做面内池化，直到轴间 spacing 比值进入 2 倍以内才恢复各向同性池化——这些都以重采样后的 spacing 为前提。

## 适用

- 模态：CT / MRI 等带体素 spacing 元数据的体数据（3D 采集）。
- 建模维度：3D 全分辨率、3D cascade 低分辨率分支、2.5D（3D 存储、逐层 2D 输入——切片轴 spacing 保持不动，只重采样面内两轴）。
- 数据量级：重采样参数由训练集统计推断，与数据量无关；注意目标 spacing 越细体素数越多，须按显存预算（本机 2×4090，约 23 GB/卡）核算 patch 可行性。
- 2D 自然图像/已各向同性的数据不需要本方法（spacing 相等时重采样退化为恒等）。

## 参数与经验值

| 参数 | 经验值/规则 | 出处 |
|---|---|---|
| 目标 spacing（各向同性） | 训练集逐轴 50th 百分位（中位数） | Suppl. Note 2.2 (p.29)；nnunetv2 planner `np.percentile(spacings, 50, 0)` |
| 各向异性判定 | `max(spacing)/min(spacing) > 3`（ANISO_THRESHOLD=3），且该轴中位体素数 ×3 < 其他轴最小体素数（官方实现的双条件，论文只给规则不给判别式） | nnunetv2 `configuration.py`、`default_experiment_planner.py` |
| 各向异性轴目标 spacing | 该轴 spacing 的 **10th 百分位**，并钳制到不低于其他轴的最大 spacing（防止比面内更细、体素数爆炸；官方实现 +1e-5 mm 防止两轴完全相等） | Suppl. Note 2.2 (p.29)、3.1 (p.31)；nnunetv2 planner |
| 重采样后形状 | `new_shape = round(old_shape × old_spacing / new_spacing)`（逐轴） | nnunetv2 `compute_new_shape` |
| 图像插值 | 3 阶样条（third order spline） | Suppl. Note 2.2 (p.29) |
| 标签插值（各向同性路径） | 逐类 one-hot 后线性插值、≥0.5 归类（论文写 "linear interpolation (one hot encoded segmentation maps)"） | Suppl. Note 2.2 (p.29)；nnunetv2 `order=1` + batchgenerators `resize_segmentation` |
| 各向异性（separate-z）路径 | out-of-plane 轴对图像与标签**都用最近邻**；面内轴图像仍 3 阶样条、标签仍 one-hot 线性 | Suppl. Note 2.2 (p.29) |
| 2.5D 变体 | 切片轴 spacing 不动（旧流水用 -1 占位），只重采样面内两轴；保存时把切片轴转置到第 0 轴 | 旧流水 `prepro_2_resample3_2D.py` |

**越界后果**：目标 spacing 取过细 → 体素数与显存爆炸（patch 被迫缩小）；取过粗 → 小目标结构直接消失。标签直接用样条/线性插值而不做 one-hot+阈值或最近邻 → 凭空出现标签集中不存在的值（如 0/1 之间的 0.4），训练与指标全错。

## 实现参考

- 旧三级流水：`prepro_0_norm → prepro_1_crop3D → prepro_2_resample3_2D`（本条对应第三级），本地落点 `/ssd/lx/body_analysis/preprocessed/prepro_2_resample3_2D.py`（`RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD = 3`，SimpleITK `sitkBSpline` / `sitkNearestNeighbor`，spacing 与 numpy 轴序反转）。
- nnU-Net 官方实现（nnunetv2，vendored 于 `/ssd/lx/totalsegmentator/nnunetv2/`）：
  - `preprocessing/resampling/default_resampling.py`：`compute_new_shape`、`determine_do_sep_z_and_axis`（原/新 spacing 都查、两轴并列最粗则不 separate-z）、`resample_data_or_seg`。
  - `experiment_planning/experiment_planners/default_experiment_planner.py`：`determine_fullres_target_spacing`（中位 + 10th 百分位 + 双条件触发 + 钳制）、data `order=3` / seg `order=1`、`order_z=0`。
  - `configuration.py`：`ANISO_THRESHOLD = 3`。
  - 标签 one-hot+阈值语义来自 batchgenerators `augmentations/utils.py: resize_segmentation`。
- 本仓库参考实现：`assets/preprocessing/resample-isotropic.py`（numpy+scipy 自包含，CPU 可跑）。

## 坑

1. **图像与标签的插值必须区分**（本条最大坑）：图像用 3 阶样条；标签要么逐类 one-hot 后线性插值再 ≥0.5 归类，要么直接最近邻。对标签直接做全局线性/样条插值会在类别边界产生 0.4 这类"半张嘴"值，或凭空制造新类别；0.5 阈值可能把小结构 thin-out 到消失，故 nnU-Net 对各向异性数据的层间方向干脆用最近邻。
2. **各向异性数据别对层间轴用中位 spacing**：中位会上采样稀疏层（ACDC 10 mm 中位层距上采样到 5 mm 以下即是插值造层），论文因此改用 10th 百分位——宁可训练时下采样、导出时上采样。
3. **spacing 与数组轴序**：SimpleITK `GetSpacing()` 顺序与 numpy 数组 shape 相反（旧流水注释 `spacing[::-1]`）；NIfTI/ITK 里 (x,y,z)，numpy 是 (z,y,x)。轴对错一个，重采样就是在错误的物理方向上缩放。
4. **separate-z 判定要同时看原 spacing 与新 spacing**（nnunetv2 `determine_do_sep_z_and_axis`）；若有两轴并列最粗（如 0.24×1.25×1.25），官方实现不做 separate-z（层间方向不唯一），本实现沿用。
5. **round 的细节**：`round(shape × old/new)`，官方用 banker's rounding（np.rint 同）；新形状钳制 ≥1 体素，避免零除/空轴。
6. **2.5D 时切片轴必须保持原 spacing**（论文 2D 配置明确 "the out of plane resolution ... is not altered"），否则层间插值伪影直接进训练数据。
7. **先裁剪后重采样**：先 crop 到前景包围盒再 resample（nnunetv2 顺序 transpose→crop→resample），否则背景稀释 spacing 推断与体素预算。

## 实验反链

暂无（`experiments/` 目前为空）。
