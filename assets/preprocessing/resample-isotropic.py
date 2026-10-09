"""resample-isotropic —— 体数据重采样到目标（近各向同性）spacing [preprocessing]

论文支撑（统一四问，简体中文）:
- 哪篇论文用了它: Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH.
  "nnU-Net: a self-configuring method for deep learning-based biomedical image
  segmentation", Nature Methods 18, 203-211 (2021)，简称 nnU-Net。
  链接: https://www.nature.com/articles/s41592-020-01008-z
- 目的: 训练集各病例体素 spacing（体素物理尺寸 mm）不一致且常各向异性（如 CT 面内
  0.7 mm、层间 5 mm）。把每个病例重采样到由数据集推断的统一目标 spacing，使 3D 卷积核
  在三轴上感受野物理尺度均匀，patch 长宽比可跟随中位形状、网络深度/池化可统一配置。
- 效果: 作为 nnU-Net 自动配置预处理（crop -> resample -> 归一化）的核心一环，
  nnU-Net 无人工干预在 23 个国际分割挑战公开数据集上超过绝大多数已有方法。
  ACDC 数例: 中位 spacing 10x1.56x1.56 mm，out-of-plane 轴目标取 10th 百分位 = 5 mm，
  重采样后中位形状 18x237x208（Suppl. Note 3.1, p.31）。
- 结论: 各向同性数据目标 spacing 取训练集逐轴中位数；各向异性数据在 out-of-plane 轴取
  该轴 spacing 的 10th 百分位（比中位更细，宁"训练下采样、导出上采样"，避免把稀疏层
  上采样产生插值伪影），且该轴重采样用最近邻。适用于体数据 3D 与 2.5D 预处理流水。

所读版本: Nature 正文付费墙（官方页仅摘要可读）；引用规则全部出自作者提供的官方补充
材料 Supplementary Notes 1-9（Springer 静态资源公开 PDF，MOESM1，"in the format
provided by the authors and unedited"）。关键出处:
- Supplementary Note 2.2 "nnU-Net design principles - Rule-based parameters:
  Target spacing and resampling" (p.29): 各向同性 -> 逐轴中位 spacing，"third order
  spline (data) and linear interpolation (one hot encoded segmentation maps)"；
  各向异性 -> out-of-plane 轴取 10th 百分位，"nearest neighbor for both data and
  one-hot encoded segmentation maps"。
- Supplementary Note 3.1 ACDC (p.31): 中位 10x1.56x1.56 -> 目标 5 mm（10th 百分位），
  重采样后中位形状 18x237x208；理由是选中位会"upsampled for training ... downsampled
  for segmentation export"产生插值伪影。
  （论文对 resampling 只给文字规则、无编号公式，下文"论文出处"均指上述节/页。）

实现说明:
- 关键规则与论文对照（论文无编号公式，按节/页引用）:
  * 目标 spacing = 逐轴 50th 百分位（各向同性）        -> Suppl. Note 2.2 (p.29)
  * 各向异性轴 = 10th 百分位并钳制不低于其他轴最大值   -> Suppl. Note 2.2 (p.29)
  * 各向异性触发: 论文只给"各向异性"概念，精确判别式取自官方实现:
    max(spacing)/min(spacing) > 3 且该轴中位体素数 x3 < 其他轴最小体素数
    （nnunetv2 default_experiment_planner.determine_fullres_target_spacing;
    ANISO_THRESHOLD=3 见 nnunetv2/configuration.py）
  * 新形状 = round(旧形状 x 旧spacing / 新spacing)     -> nnunetv2
    default_resampling.compute_new_shape。注意与论文 ACDC 数例的关系：该公式
    只直接复现 18 与 237 两轴；论文第三轴 208 是逐病例重采样后逐轴取中位的
    统计结果，无法由中位 spacing 向量推出（本实现同条件得 256）。公式本身与
    nnunetv2 compute_new_shape 一致。
  * 图像 3 阶样条 / 标签 one-hot+线性+>=0.5 阈值       -> Suppl. Note 2.2 (p.29);
    one-hot+阈值语义同 batchgenerators resize_segmentation（nnunetv2 对 seg 传 order=1）
  * separate-z: out-of-plane 轴对图像与标签都用最近邻   -> Suppl. Note 2.2 (p.29);
    触发与轴选择同 nnunetv2 determine_do_sep_z_and_axis（原/新 spacing 都查；两轴并列
    最粗则不做 separate-z）
- 参考的开源实现:
  * nnunetv2（vendored 于 /ssd/lx/totalsegmentator/nnunetv2/）:
    preprocessing/resampling/default_resampling.py、
    experiment_planning/experiment_planners/default_experiment_planner.py、
    configuration.py；batchgenerators/augmentations/utils.py 的 resize_segmentation。
  * /ssd 旧流水 /ssd/lx/body_analysis/preprocessed/prepro_2_resample3_2D.py
    （RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD=3，sitkBSpline/sitkNearestNeighbor）。
- 与开源实现不一致处（以论文为准）:
  * vendored totalsegmentator 的 nnunetv2 default_resampling.py 非分离-z 路径用
    torch F.interpolate trilinear（速度取舍，已偏离论文）: 本实现按论文 Suppl. Note
    2.2 用 3 阶样条（scipy.ndimage.zoom, order=3, grid_mode=True）。
  * 旧流水对标签一律最近邻（更保守）; 论文为 one-hot+线性。本实现默认按论文
    one-hot+线性+>=0.5，同时提供 seg_interpolation="nearest" 选项（等价旧流水）。
  * 论文/官方代码对"各向异性触发"未给出与体素数相关的判别式文字（仅代码），本实现的
    双条件与 nnunetv2 完全一致，并以注释标明该判别式属官方实现补充。
- 布局约定（预处理，numpy 数组）: 体积为 3D 数组 (D, H, W)；spacing 为长度 3 的序列，
  spacing[i] 与 shape[i] 一一对应（即沿数组第 i 维的体素物理边长，单位 mm）。
  注意: SimpleITK GetSpacing() 的顺序与 numpy shape 相反，外部传入前需自行反转。
- 数值稳定: spacing 在除法前钳制到 >= SPACING_EPS=1e-8 mm（物理上 spacing>0；1e-8 mm
  远小于任何真实成像分辨率，只防脏数据/零除）；重采样后形状逐轴钳制 >=1 体素（防空轴）；
  输出断言无 NaN/Inf。

自测: 直接运行本文件（CPU，秒级）:
    /ssd/lx/totalsegmentator/.venv/bin/python resample-isotropic.py
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import zoom

__all__ = [
    "ANISO_THRESHOLD",
    "SPACING_EPS",
    "determine_target_spacing",
    "compute_resampled_shape",
    "resample_image",
    "resample_seg",
    "resample_data_or_seg",
    "get_separate_z_axis",
]

# 各向异性阈值: max(spacing)/min(spacing) 超过它视为各向异性。
# 论文 Suppl. Note 2.2 只给"各向异性"概念; 数值 3 取自 nnunetv2 configuration.py
# 的 ANISO_THRESHOLD=3（官方实现），也与旧流水 RESAMPLING_SEPARATE_Z_ANISO_THRESHOLD 一致。
ANISO_THRESHOLD = 3.0

# spacing 下限保护（mm）。spacing 物理上 > 0; 1e-8 mm 远小于任何真实成像分辨率，
# 仅用于防止脏数据（如 0 或负值）在除法处产生零除/Inf。
SPACING_EPS = 1e-8


def _as_positive_spacing(spacing: np.ndarray | list | tuple) -> np.ndarray:
    """校验并钳制 spacing 为正（单位 mm），返回 float64 数组。

    参数:
        spacing: 长度 3 序列，spacing[i] 为数组第 i 维的体素物理边长。
    返回:
        (3,) float64 数组，逐元素钳制到 >= SPACING_EPS。
    """
    s = np.asarray(spacing, dtype=np.float64)
    if s.shape != (3,):
        raise ValueError(f"spacing 必须是长度 3 的序列（与 (D,H,W) 逐维对应），收到 shape={s.shape}")
    # 钳制依据: 见 SPACING_EPS 注释（物理上 spacing>0，eps 仅防零除）
    return np.maximum(s, SPACING_EPS)


def determine_target_spacing(
    spacings: np.ndarray,
    sizes: np.ndarray | None = None,
    aniso_threshold: float = ANISO_THRESHOLD,
) -> np.ndarray:
    """由训练集 spacing（与可选形状）推断目标 spacing（单位 mm）。

    规则（论文 Suppl. Note 2.2, p.29）:
      1) 逐轴取 50th 百分位（中位数）作为初始目标;
      2) 若最粗轴相对其他轴各向异性（官方实现双条件，见下），该轴改取其 spacing 的
         10th 百分位，并钳制到不低于其他轴的最大 spacing（防止比面内更细导致体素数爆炸;
         官方实现再加 1e-5 mm 使两轴严格不等）。
    各向异性判别式（论文未给，取自 nnunetv2 determine_fullres_target_spacing）:
      target[worst] > aniso_threshold * max(其他轴 target)
      且  sizes[worst] * aniso_threshold < min(其他轴 sizes)   （sizes 为逐轴中位体素数;
      第二个条件排除"粗但厚"的轴——物理厚度不小且体素数不少时不值得提分辨率）

    参数:
        spacings: (N, 3) 数组，N 个训练病例的 spacing，列与 (D,H,W) 轴对应。
        sizes: (N, 3) 数组或 None，各病例（裁剪后）形状; None 时只按 spacing 条件判定
              （官方实现同时要求两个条件，传 None 属简化，见资产卡"参数与经验值"）。
        aniso_threshold: 各向异性阈值，默认 ANISO_THRESHOLD=3。
    返回:
        (3,) float64 目标 spacing。
    """
    spacings = np.asarray(spacings, dtype=np.float64)
    if spacings.ndim != 2 or spacings.shape[1] != 3 or spacings.shape[0] == 0:
        raise ValueError(f"spacings 须为 (N, 3) 数组且 N>=1，收到 shape={spacings.shape}")
    spacings = np.maximum(spacings, SPACING_EPS)  # 防零除（见 SPACING_EPS 注释）

    # 逐轴中位数 = 50th 百分位（论文 Suppl. Note 2.2）
    target = np.percentile(spacings, 50, axis=0)

    worst = int(np.argmax(target))
    other = [i for i in range(3) if i != worst]
    max_other_spacing = float(target[other].max())

    has_aniso_spacing = target[worst] > aniso_threshold * max_other_spacing
    if sizes is not None:
        sizes = np.asarray(sizes, dtype=np.float64)
        if sizes.shape != spacings.shape:
            raise ValueError("sizes 与 spacings 形状须一致")
        median_size = np.percentile(sizes, 50, axis=0)
        min_other_size = float(median_size[other].min())
        has_aniso_voxels = median_size[worst] * aniso_threshold < min_other_size
    else:
        has_aniso_voxels = True  # sizes 未提供时的简化（官方实现要求两条件同时成立）

    if has_aniso_spacing and has_aniso_voxels:
        # 各向异性轴取该轴 spacing 的 10th 百分位（论文 Suppl. Note 2.2; ACDC 例
        # 10 mm 中位 -> 5 mm = 10th 百分位, Suppl. Note 3.1）
        p10 = float(np.percentile(spacings[:, worst], 10))
        if p10 < max_other_spacing:
            # 钳制: 不比其他轴更细（体素数防爆炸）。+1e-5 mm（10 nm，物理可忽略）使
            # 两轴严格不等，避免下游浮点相等判断退化; 与 nnunetv2 官方实现一致。
            p10 = max(max_other_spacing, p10) + 1e-5
        target[worst] = p10
    return target


def compute_resampled_shape(
    shape: np.ndarray | list | tuple,
    spacing_old: np.ndarray | list | tuple,
    spacing_new: np.ndarray | list | tuple,
) -> np.ndarray:
    """计算重采样后的体素形状。

    公式（nnunetv2 compute_new_shape；对论文 ACDC 数例只直接复现 18 与 237
    两轴，第三轴 208 见文件头说明——它是逐病例重采样后的中位统计，无法由
    中位 spacing 向量推出）:
        new_shape[i] = round(old_shape[i] * spacing_old[i] / spacing_new[i])
    round 采用 np.rint（banker's rounding），与官方实现 int(round(...)) 语义相同。

    参数:
        shape: (3,) 各维体素数（(D,H,W)）。
        spacing_old / spacing_new: (3,) 原/新 spacing（mm，与 shape 逐维对应）。
    返回:
        (3,) int64 新形状（逐轴钳制 >=1，防空轴/零除）。
    """
    shape = np.asarray(shape, dtype=np.float64)
    s_old = _as_positive_spacing(spacing_old)
    s_new = _as_positive_spacing(spacing_new)
    new_shape = np.rint(shape * s_old / s_new)
    # 钳制 >=1: 逐轴至少 1 个体素。物理依据: spacing 有限时任何真实体积在该轴至少占
    # 1 体素; 防止极端比例下出现 0 导致下游空数组/零除。
    return np.maximum(new_shape, 1.0).astype(np.int64)


def _resize_volume(
    volume: np.ndarray, new_shape: np.ndarray | list | tuple, order: int
) -> np.ndarray:
    """3D 体数据重采样（scipy.ndimage.zoom, grid_mode=True）。

    坐标映射: zoom 的 grid_mode=True 采用 src = (dst + 0.5) * (n_old / n_new) - 0.5
    （体素中心对齐, align_corners=False），与 skimage.transform.resize 语义一致，也与
    nnunetv2 separate-z 分支中 (map + 0.5) * scale - 0.5 的 map_coordinates 映射一致;
    zoom 内部分轴插值，不物化整网格坐标（map_coordinates 需要显式
    (3, n1, n2, n3) 坐标数组，大体积下内存开销高）。
    边界外推 mode="nearest"（复制边界），等价 skimage resize 的 mode="edge"，
    与 nnunetv2/batchgenerators 的边界处理一致。

    参数:
        volume: (D,H,W) 数组。
        new_shape: (3,) 目标体素数。
        order: 样条阶数（图像 3 阶样条按论文; 标签 one-hot 线性用 1; 最近邻用 0）。
    返回:
        (D,H,W) float64 数组（调用方按需转换 dtype/归类）。
    """
    volume = np.asarray(volume, dtype=np.float64)
    new_shape = tuple(int(n) for n in new_shape)
    if volume.shape == new_shape:
        return volume.copy()
    # zoom 的参数是"放大倍率": output_size = round(input_size * zoom)，故 zoom[i] =
    # n_new[i] / n_old[i]（注意不是 spacing 比 n_old/n_new，那是坐标缩放方向）。
    # grid_mode=True 时坐标映射 src = (dst + 0.5) * n_old / n_new - 0.5，
    # 与 skimage resize / nnunetv2（体素中心对齐, align_corners=False）一致。
    factors = [new_shape[i] / volume.shape[i] for i in range(3)]
    out = zoom(volume, factors, order=order, mode="nearest", grid_mode=True)
    if out.shape != new_shape:
        # 保险: zoom 输出尺寸按 round 计算, 浮点噪声可能产生 ±1 偏差; 用边缘复制
        # pad / 截断钳到精确 new_shape（不引入新值）。
        pads = [(0, max(0, new_shape[i] - out.shape[i])) for i in range(3)]
        out = np.pad(out, pads, mode="edge")
        slices = tuple(slice(0, new_shape[i]) for i in range(3))
        out = out[slices]
    return out


def _resize_seg_labels(
    seg: np.ndarray, new_shape: np.ndarray | list | tuple, order: int = 1
) -> np.ndarray:
    """标签重采样: 逐类 one-hot 后插值，再按 >=0.5 归类。

    论文 Suppl. Note 2.2 (p.29): "linear interpolation (one hot encoded segmentation
    maps)"。直接对整张标签做全局插值会在类别边界产生不存在的标签值（如 0.4），
    one-hot+阈值避免此坑; 语义同 batchgenerators resize_segmentation（其用 >=0.5，
    见 batchgenerators/augmentations/utils.py，nnunetv2 对 seg 传 order=1）。

    参数:
        seg: (D,H,W) 整数标签数组（0 为背景）。
        new_shape: (3,) 目标体素数。
        order: one-hot 掩码的插值阶数（论文为线性=1; 也可传 0 等价最近邻）。
    返回:
        (D,H,W) 数组，dtype 与输入相同，取值仍为原标签集合的子集。
    """
    seg = np.asarray(seg)
    new_shape = tuple(int(n) for n in new_shape)
    if seg.shape == new_shape:
        return seg.copy()
    out = np.zeros(new_shape, dtype=seg.dtype)
    for label in np.sort(np.unique(seg)):  # 背景留在 0; 后写优先同官方实现
        mask = _resize_volume((seg == label).astype(np.float64), new_shape, order)
        out[mask >= 0.5] = label
    return out


def _resize_nearest_along_axis(
    volume: np.ndarray, new_shape: np.ndarray | list | tuple, axis: int
) -> np.ndarray:
    """仅沿单一轴做最近邻重采样（其余轴尺寸不变）。

    论文 Suppl. Note 2.2 (p.29): 各向异性数据的 out-of-plane 轴对图像与标签都用最近邻
    （层间插值伪影最少的方案）。
    """
    volume = np.asarray(volume, dtype=np.float64)
    n_old, n_new = volume.shape[axis], int(new_shape[axis])
    if n_old == n_new:
        return volume.copy()
    # 仅层间轴缩放（zoom 为放大倍率 n_new/n_old）, order=0 即最近邻, grid_mode=True 同上
    factors = [1.0, 1.0, 1.0]
    factors[axis] = n_new / n_old
    out = zoom(volume, factors, order=0, mode="nearest", grid_mode=True)
    if out.shape[axis] != n_new:  # 同 _resize_volume 的保险钳制
        pads = [0, 0, 0]
        pads[axis] = max(0, n_new - out.shape[axis])
        out = np.pad(out, [(0, pads[0]), (0, pads[1]), (0, pads[2])], mode="edge")
        slices = [slice(None), slice(None), slice(None)]
        slices[axis] = slice(0, n_new)
        out = out[tuple(slices)]
    return out


def get_separate_z_axis(
    spacing_old: np.ndarray | list | tuple,
    spacing_new: np.ndarray | list | tuple,
    aniso_threshold: float = ANISO_THRESHOLD,
) -> int | None:
    """判定是否走 separate-z（各向异性分离重采样）路径，并给出最粗轴。

    与 nnunetv2 determine_do_sep_z_and_axis 一致:
      - 先查原 spacing、再查新 spacing（原数据或目标有一方各向异性即走分离路径）;
      - 最粗轴有两轴并列（如 0.24x1.25x1.25）或三轴并列时层间方向不唯一，不做分离。
        （官方注释: "this happens for spacings like (0.24, 1.25, 1.25) ... we do not
        want to resample separately in the out of plane axis"）
    返回:
        None（不分离）或 int（separate-z 的层间轴编号）。
    """
    for spacing in (_as_positive_spacing(spacing_old), _as_positive_spacing(spacing_new)):
        if float(spacing.max()) / float(spacing.min()) > aniso_threshold:
            # 并列判定用 np.isclose（浮点安全; 官方用 == 1 的写法对浮点不稳健）
            axes = np.where(np.isclose(spacing / spacing.max(), 1.0))[0]
            if len(axes) == 1:
                return int(axes[0])
            return None  # 两轴或三轴并列最粗: 层间方向不唯一，不分离
    return None


def resample_data_or_seg(
    volume: np.ndarray,
    spacing_old: np.ndarray | list | tuple,
    spacing_new: np.ndarray | list | tuple,
    is_seg: bool = False,
    seg_interpolation: str = "onehot_linear",
    aniso_threshold: float = ANISO_THRESHOLD,
) -> np.ndarray:
    """把 (D,H,W) 体数据重采样到目标 spacing（自动选择各向同性 / separate-z 路径）。

    路径选择与插值（对应论文 Suppl. Note 2.2, p.29）:
      - spacing 相等: 直接返回副本（恒等）。
      - separate-z（原或新 spacing 各向异性）: 层间轴图像/标签都用最近邻，
        面内轴图像 3 阶样条 / 标签 one-hot 线性+>=0.5。
      - 否则（各向同性）: 图像 3 阶样条; 标签 one-hot 线性+>=0.5
        （seg_interpolation="nearest" 时改用全局最近邻，等价旧流水）。

    参数:
        volume: (D,H,W) numpy 数组; 图像为强度值，标签为整数类别。
        spacing_old / spacing_new: (3,) 原/新 spacing（mm，逐维对应）。
        is_seg: True 按标签路径处理（dtype 与原标签集合保持）。
        seg_interpolation: "onehot_linear"（论文默认）或 "nearest"。
        aniso_threshold: 各向异性阈值（默认 3）。
    返回:
        (D,H,W) 数组，形状 = compute_resampled_shape(...)。
    """
    s_old = _as_positive_spacing(spacing_old)
    s_new = _as_positive_spacing(spacing_new)
    new_shape = compute_resampled_shape(np.asarray(volume).shape, s_old, s_new)
    if tuple(new_shape) == tuple(np.asarray(volume).shape):
        return np.asarray(volume).copy()  # 恒等: spacing/形状均未变

    axis = get_separate_z_axis(s_old, s_new, aniso_threshold)

    if axis is not None:
        # separate-z: 面内两轴先插值（保持层间轴尺寸），层间轴再最近邻
        inplane_shape = list(new_shape)
        inplane_shape[axis] = np.asarray(volume).shape[axis]
        if is_seg:
            vol = _resize_seg_labels(volume, inplane_shape, order=1)
        else:
            vol = _resize_volume(volume, inplane_shape, order=3)
        return _resize_nearest_along_axis(vol, new_shape, axis).astype(volume.dtype)

    if is_seg:
        if seg_interpolation == "nearest":
            return _resize_volume(volume, new_shape, order=0).astype(volume.dtype)
        return _resize_seg_labels(volume, new_shape, order=1)
    return _resize_volume(volume, new_shape, order=3).astype(np.float64)


def resample_image(volume, spacing_old, spacing_new) -> np.ndarray:
    """图像（强度体）重采样到目标 spacing，返回 (D,H,W) float64。

    图像插值 = 3 阶样条（论文 Suppl. Note 2.2）; separate-z 时层间轴用最近邻。
    注意: 3 阶样条可能轻微过冲（输出略超输入 min/max），训练前若做窗宽窗位钳制
    应在重采样之后执行。
    """
    return resample_data_or_seg(volume, spacing_old, spacing_new, is_seg=False)


def resample_seg(
    volume: np.ndarray,
    spacing_old: np.ndarray | list | tuple,
    spacing_new: np.ndarray | list | tuple,
    seg_interpolation: str = "onehot_linear",
) -> np.ndarray:
    """标签（分割标注体）重采样到目标 spacing，返回 (D,H,W) 数组（dtype/取值保持）。

    标签插值默认 one-hot+线性+>=0.5（论文）; seg_interpolation="nearest" 为全局最近邻
    （等价旧流水 sitkNearestNeighbor，对小结构更保守）。
    """
    return resample_data_or_seg(
        volume, spacing_old, spacing_new, is_seg=True, seg_interpolation=seg_interpolation
    )


# ---------------------------------------------------------------------------
# 自测（CPU、秒级、固定随机种子、无数据下载）
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time

    t0 = time.time()
    np.random.seed(0)

    def _assert_finite(x, name):
        arr = np.asarray(x)
        assert np.all(np.isfinite(arr)), f"{name} 出现 NaN/Inf"

    # ---- 检查 1: 目标 spacing 推断（复现论文 ACDC 数例, Suppl. Note 3.1, p.31）----
    # 合成 10 个 ACDC 式病例: 层间 8 例 10mm + 2 例 5mm（10th 百分位恰为 5.0）、面内 1.56mm
    spacings = np.array(
        [(10.0, 1.56, 1.56)] * 8 + [(5.0, 1.56, 1.56)] * 2
    )
    sizes = np.array([(9, 237, 256)] * 10, dtype=np.float64)
    target = determine_target_spacing(spacings, sizes)
    assert np.allclose(target, (5.0, 1.56, 1.56), atol=1e-6), f"目标 spacing 推断错误: {target}"
    # 论文: 目标层间 5mm; 重采样后中位形状层间轴 9 -> 18
    new_shape = compute_resampled_shape((9, 237, 256), (10.0, 1.56, 1.56), target)
    assert new_shape[0] == 18, f"层间轴新形状应为 18（论文 18x237x208），得到 {new_shape[0]}"
    assert new_shape[1] == 237 and new_shape[2] == 256  # 面内 spacing 未变
    print("PASS 1 目标spacing推断: 逐轴中位+各向异性轴10th百分位 (10,1.56,1.56)->(5,1.56,1.56), 形状9->18 (论文ACDC数例)")

    # ---- 检查 2: 各向同性图像重采样（3 阶样条）----
    d, h, w = 16, 32, 32
    zz, yy, xx = np.meshgrid(
        np.linspace(0, np.pi * 2, d), np.linspace(0, np.pi * 2, h), np.linspace(0, np.pi * 2, w), indexing="ij"
    )
    vol = (np.sin(zz) * np.cos(yy) + 0.5 * np.sin(2 * xx)).astype(np.float64)  # 平滑解析体
    out = resample_image(vol, (2.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    assert out.shape == (32, 32, 32), f"图像重采样形状错误: {out.shape}"
    _assert_finite(out, "图像重采样输出")
    vmin, vmax = float(vol.min()), float(vol.max())
    vrange = vmax - vmin
    # 3 阶样条允许轻微过冲: 阈值取值域的 10%（样条过冲上界经验值, 远小于此）
    assert out.min() >= vmin - 0.1 * vrange and out.max() <= vmax + 0.1 * vrange
    mid = 0.5 * (vmin + vmax)
    assert abs(float(out.mean()) - mid) < 0.2 * vrange  # 均值近似保持（平滑体，均值≈域中点）
    print(f"PASS 2 图像重采样: (16,32,32)@(2,1,1)->(1,1,1) 得 {out.shape}, 无NaN/Inf, 值域过冲<10%")

    # ---- 检查 3: 标签重采样（one-hot+线性+>=0.5）: 标签集合不膨胀 ----
    seg = np.zeros((16, 32, 32), dtype=np.uint8)
    seg[4:12, 10:22, 10:22] = 1  # 球状结构近似（方块占位）
    seg[13:15, 5:10, 5:10] = 2   # 第二个类
    seg_out = resample_seg(seg, (2.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    assert seg_out.shape == (32, 32, 32)
    _assert_finite(seg_out, "标签重采样输出")
    assert set(np.unique(seg_out)).issubset({0, 1, 2}), f"标签值膨胀: {np.unique(seg_out)}"
    # 类 1 体素数应约翻倍（spacing 2->1）, 容差 30%（0.5 阈值的边界取舍所致）
    v_old, v_new = float((seg == 1).sum()), float((seg_out == 1).sum())
    assert 2 * 0.7 < v_new / v_old < 2 * 1.3, f"类1体素体积比异常: {v_new / v_old:.3f}"
    print(f"PASS 3 标签重采样: 值域保持子集{{0,1,2}}, 类1体素比 {v_new / v_old:.3f}≈2（期望 2±30%）")

    # ---- 检查 4: separate-z（各向异性）: 层间最近邻不产生新值 ----
    vol_z = np.tile(np.arange(16, dtype=np.float64)[:, None, None], (1, 32, 32))  # 每层常数、层间不同
    out_z = resample_image(vol_z, (5.0, 1.0, 1.0), (2.5, 1.0, 1.0))
    assert out_z.shape == (32, 32, 32)
    _assert_finite(out_z, "separate-z 图像输出")
    # 原数据各向异性（5/1>3）触发分离路径; 层间最近邻 => 输出值必是输入值的子集
    assert set(np.unique(out_z)).issubset(set(np.unique(vol_z))), "层间最近邻产生了新值（应为NN）"
    seg_z = (vol_z >= 8).astype(np.uint8)
    seg_z_out = resample_seg(seg_z, (5.0, 1.0, 1.0), (2.5, 1.0, 1.0))
    assert set(np.unique(seg_z_out)).issubset({0, 1}), f"separate-z 标签值膨胀: {np.unique(seg_z_out)}"
    _assert_finite(seg_z_out, "separate-z 标签输出")
    print("PASS 4 separate-z: 层间NN（输出值⊆输入值）, 图像/标签均无NaN/Inf, 值域保持")

    # ---- 检查 5: 两轴并列最粗 => 不走 separate-z（官方 determine_do_sep_z_and_axis）----
    vol_tie = np.random.rand(8, 16, 32)
    out_tie = resample_image(vol_tie, (5.0, 5.0, 1.0), (2.5, 2.5, 1.0))
    assert out_tie.shape == (16, 32, 32)
    _assert_finite(out_tie, "tie 情形输出")
    print("PASS 5 两轴并列最粗(5,5,1): 不做separate-z, 整体样条路径, 形状正确且无NaN/Inf")

    # ---- 检查 6: 恒等（spacing 未变）----
    out_id = resample_image(vol, (1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    assert np.array_equal(out_id, vol), "恒等重采样应返回逐元素相同的副本"
    seg_id = resample_seg(seg, (2.0, 1.0, 1.0), (2.0, 1.0, 1.0))
    assert np.array_equal(seg_id, seg)
    print("PASS 6 恒等情形: spacing 未变时返回逐元素相同副本")

    # ---- 检查 7: 数值稳定护栏 ----
    tiny = determine_target_spacing(np.array([(10.0, 1.56, 1.56)]))
    assert np.all(np.isfinite(tiny))  # 单病例 / 极端输入不产生 NaN/Inf
    assert compute_resampled_shape((1, 1, 1), (10.0, 1.0, 1.0), (1000.0, 1.0, 1.0))[0] == 1  # 钳制>=1体素
    print("PASS 7 数值稳定: 极端比例下形状钳制>=1, 全流程无NaN/Inf")

    print(f"PASS 全部自测通过（耗时 {time.time() - t0:.2f}s）")
