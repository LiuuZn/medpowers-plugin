"""intensity-normalization — 强度归一化（模态分治：CT 全局百分位截断 + 全局 z-score；MRI/其他逐图 z-score）[preprocessing]

资产卡：assets/preprocessing/intensity-normalization.md

论文支撑（统一四问）：
- 哪篇论文用了它：Isensee 等（德国癌症研究中心 DKFZ），nnU-Net，2021，
  Nature Methods，https://www.nature.com/articles/s41592-020-01008-z 。
- 目的：消除设备/采集协议造成的强度尺度不一致，作为 nnU-Net 自动预处理
  （裁剪 → 归一化 → 重采样）的默认一环。
- 效果：作为默认预处理步骤，nnU-Net 无需人工干预在 23 个公共数据集
  （国际分割挑战赛）上超过大多数现有方法（含高度专用方案）。
- 结论：归一化按模态分治——CT 用训练集全局前景统计（0.5/99.5 百分位截断 +
  全局前景 mean/std 的 z-score，所有图像共用一组数）；其余模态（MRI 等）
  默认逐图 z-score；裁剪使尺寸平均缩小 ≥25% 时统计只在非零 mask 内计算，
  且归一化只作用于 mask 内体素。
- 所读版本（一手优先）：Nature Methods 官方补充材料全文已读——
  Supplementary Information（41592_2020_1008_MOESM1_ESM.pdf，44 页，
  出版社官方免费下载：
  https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-020-01008-z/
  MediaObjects/41592_2020_1008_MOESM1_ESM.pdf ），其中 2.2 节写明
  "Z-score per image ... is a good default. We deviate from this default
  only for CT images, where a global normalization scheme is determined
  based on the intensities found in foreground voxels across all training
  cases."；3.2 节 LiTS walk-through 的 "Intensity normalization" 段写明
  "the intensities of the voxels belonging to
  any of the foreground classes ... are collected across all training
  cases ... normalized by clipping them to the 0.5 and 99.5 percentiles,
  followed by subtraction of the global mean and division by the global
  standard deviation"；SN6.7（LiTS）只含实例值：Clip to [−17, 201]、
  减 99.40、除 39.39（出处节还注明 "131 training CT images"）。
  交叉核对：arXiv:1904.08128（同团队方法论文全文，
  https://ar5iv.labs.arxiv.org/html/1904.08128 ）与出版社官方页摘要
  （正文付费墙，仅摘要可读）。

实现说明——关键公式与论文的对应（论文无编号公式，按原文句子）：
1) CT（补充材料 2.2 节与 SN6.7；arXiv:1904.08128 同义原句 "nnU-Net uses
   the 0.5 and 99.5 percentiles of the foreground voxels for clipping as
   well as the global foreground mean and standard deviation for
   normalization on all images."）：
       z = (clip(x, P0.5, P99.5) - mean_fg) / (std_fg + eps)
   其中 P0.5/P99.5/mean_fg/std_fg 在训练集前景体素（标注 >0）集合上跨
   全部训练例一次性统计，训练与推理共用（见 compute_ct_global_stats /
   normalize_ct）。
2) 非 CT 默认（补充材料 2.2 节 "Z-score per image (mean substraction and
   division by standard deviation) is a good default."；arXiv 原句
   "each image is normalized independently by subtracting its mean,
   followed by division with its standard deviation."）：
       z = (x - mean(x)) / (std(x) + eps)
   统计与归一都限定在本张图像上（见 normalize_zscore，mask=None）。
3) mask 变体（arXiv:1904.08128 原句 "If cropping resulted in an average
   size decrease of 25% or more, a mask for central non-zero voxels is
   created and the normalization is applied within that mask only,
   ignoring the surrounding zero voxels."）：统计量只在 mask>0 内计算，
   且归一化只作用于 mask>0 的体素；mask 外体素论文未规定处理方式
   （见下"与开源实现不一致处"）。
- 参考的开源实现（均已逐一核对源码）：
  * 官方 nnU-Net v2（本地 /ssd/lx/totalsegmentator/nnunetv2/）：
    - experiment_planning/dataset_fingerprint/fingerprint_extractor.py:58,62
      —— CT 统计为前景口径：foreground_mask = segmentation[0] > 0，
      只取前景体素跨例汇总，与论文一致；
    - preprocessing/normalization/default_normalization_schemes.py
      —— CTNormalization：clip(percentile_00_5, percentile_99_5) 后减
      mean 除 std（除法用 max(std, 1e-8)）；ZScoreNormalization::run：
      image[mask] = (image[mask] - mean) / max(std, 1e-8)，只改 mask 内，
      mask 外保持原值。
  * 官方 nnU-Net v1（GitHub MIC-DKFZ/nnUNet nnunetv1 分支
    nnunet/preprocessing/preprocessing.py，经网页核对）：mask 变体为
    data[c][mask] = (data[c][mask] - mean) / (std + 1e-8)，随后
    data[c][mask == 0] = 0——mask 外置 0。
  * 本地旧流水 /ssd/lx/body_analysis/preprocessed/prepro_0_norm.py:17-42：
    CT 前景 [0.5, 99.5] 截断 + 前景 mean/std z-score，其他模态全图
    z-score，std 均加 1e-8。
- 与开源实现不一致处（以论文为准）：
  * mask 外体素：论文只写 "the normalization is applied within that mask
    only"，未规定 mask 外像素的去留。官方 v2 保持原值、官方 v1 置 0。
    本实现默认保持原值（最贴近论文句面，与 v2 一致）；提供
    zero_outside_mask=True 复现 v1 的置 0 口径。
  * 防零除形式：v2 为 max(std, 1e-8)，v1 与本地旧流水为 std + 1e-8。
    本实现取 std + eps（eps=1e-8，与 v1/旧流水一致；std=0 时两者退化
    行为相同，正常情形 eps 远小于典型 std≈O(1)，结果无差别）。
  * compute_ct_global_stats 的 masks=None（整图统计）为本实现的扩展
    口径，供无标注数据使用；论文与官方实现（v1/v2）均为前景口径，
    该扩展不属于论文方案，使用时须自行保证训练/推理一致。

依赖：仅 numpy。布局：预处理为逐例 numpy 数组，3D 影像布局 (D, H, W)
（与 SimpleITK GetArrayFromImage 一致），2D 切片为 (H, W)；统计对全体
体素做，与维度无关，任意同形数组均可。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DEFAULT_EPS",
    "DEFAULT_PERCENTILE_LOW",
    "DEFAULT_PERCENTILE_HIGH",
    "compute_ct_global_stats",
    "normalize_ct",
    "normalize_zscore",
]

# 与本地旧流水 prepro_0_norm.py 相同的防零除常数（依据见文件头 docstring）
DEFAULT_EPS = 1e-8

# 论文规定的 CT 截断百分位（0.5 / 99.5，全局前景分布）
DEFAULT_PERCENTILE_LOW = 0.5
DEFAULT_PERCENTILE_HIGH = 99.5


def compute_ct_global_stats(
    images: list[np.ndarray],
    masks: list[np.ndarray] | None = None,
    percentile_low: float = DEFAULT_PERCENTILE_LOW,
    percentile_high: float = DEFAULT_PERCENTILE_HIGH,
) -> dict[str, float]:
    """在训练集上汇总 CT 的全局前景统计量（论文 CT 方案的第一步）。

    论文口径：把所有训练例的前景体素强度（mask>0，即标注属任意前景类
    的体素）汇总成一个分布，一次性计算 0.5/99.5 百分位截断界与全局
    mean/std。官方 nnU-Net v1/v2 亦为前景口径。返回值必须持久化（如
    JSON），训练与推理共用同一组数。

    Args:
        images: 训练图像列表，每个元素为 numpy 数组，3D 布局 (D, H, W)
            （2D 为 (H, W)），float 类型建议 float32。
        masks: 与 ``images`` 等长等形的前景 mask 列表，前景为 mask>0。
            论文与官方实现均为前景口径，正式使用应提供标注。None 仅为
            无标注数据时的扩展退化口径（整图体素统计，非论文方案）。
        percentile_low: 下截断百分位，论文取 0.5。
        percentile_high: 上截断百分位，论文取 99.5。

    Returns:
        dict，含 ``clip_low`` / ``clip_high``（截断界）、``mean`` / ``std``
        （全局 z-score 参数）、``n_voxels``（参与统计的体素数）。

    Raises:
        ValueError: images 为空，或 masks 长度/形状与 images 不一致，
            或汇总后没有任何体素。
    """
    if len(images) == 0:
        raise ValueError("images 不能为空")
    pooled: list[np.ndarray] = []
    if masks is None:
        for img in images:
            pooled.append(np.asarray(img, dtype=np.float64).reshape(-1))
    else:
        if len(masks) != len(images):
            raise ValueError(
                f"masks 长度 {len(masks)} 与 images 长度 {len(images)} 不一致"
            )
        for img, mask in zip(images, masks):
            img = np.asarray(img, dtype=np.float64)
            mask = np.asarray(mask)
            if mask.shape != img.shape:
                raise ValueError(f"mask 形状 {mask.shape} 与影像形状 {img.shape} 不一致")
            pooled.append(img[mask > 0])
    values = np.concatenate(pooled) if pooled else np.empty(0, dtype=np.float64)
    if values.size == 0:
        raise ValueError("汇总后没有任何体素：请检查 mask 是否全为 0")

    mean = float(values.mean())
    std = float(values.std())
    clip_low = float(np.percentile(values, percentile_low))
    clip_high = float(np.percentile(values, percentile_high))
    return {
        "clip_low": clip_low,
        "clip_high": clip_high,
        "mean": mean,
        "std": std,
        "n_voxels": int(values.size),
    }


def normalize_ct(
    image: np.ndarray,
    stats: dict[str, float],
    eps: float = DEFAULT_EPS,
) -> np.ndarray:
    """按论文 CT 方案归一化单张影像：先全局百分位截断，再做全局 z-score。

    对应论文公式（补充材料 2.2/SN6.7 节；arXiv:1904.08128 Preprocessing
    一节）：
        z = (clip(x, clip_low, clip_high) - mean) / (std + eps)
    其中所有统计量来自 ``compute_ct_global_stats`` 在训练集前景上的汇总，
    训练与推理必须传入同一组数。论文实例（补充材料 SN6.7，LiTS）：
    clip [−17, 201]，减 99.40，除 39.39。

    Args:
        image: 单张 CT 影像，numpy 数组，3D 布局 (D, H, W)（2D 为 (H, W)）。
        stats: ``compute_ct_global_stats`` 的返回值（clip_low/clip_high/
            mean/std），须与训练时持久化的统计一致。
        eps: 防 std=0 的常数，默认 1e-8（依据见文件头 docstring）。

    Returns:
        与输入同形的 float32 数组；截断后 |z| 大致落在 0~4 之间，远端
        组织对比被削平属论文方案的预期行为。

    Raises:
        ValueError: stats 缺少必需字段。
    """
    for key in ("clip_low", "clip_high", "mean", "std"):
        if key not in stats:
            raise ValueError(f"stats 缺少必需字段：{key}")
    x = np.asarray(image, dtype=np.float32)
    x = np.clip(x, stats["clip_low"], stats["clip_high"])
    # eps 防零除（全前景常数强度时 std=0），依据见文件头 docstring
    return (x - np.float32(stats["mean"])) / (np.float32(stats["std"]) + np.float32(eps))


def normalize_zscore(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    eps: float = DEFAULT_EPS,
    zero_outside_mask: bool = False,
) -> np.ndarray:
    """按论文非 CT 模态（MRI 等）默认方案逐图 z-score 归一化。

    对应论文公式（补充材料 2.2 节；arXiv:1904.08128 Preprocessing 一节）：
        z = (x - mean(x)) / (std(x) + eps)
    统计量在本张图像内计算（逐图独立）。

    mask 变体（论文触发条件：裁剪后平均尺寸缩小 ≥25%）：统计量只在
    mask>0 内计算，且归一化只作用于 mask>0 的体素（论文原句 "the
    normalization is applied within that mask only"）；mask 外体素默认
    保持原值（与官方 v2 一致），``zero_outside_mask=True`` 时置 0
    （复现官方 v1 口径）。

    Args:
        image: 单张影像，numpy 数组，3D 布局 (D, H, W)（2D 为 (H, W)）。
        mask: 与 image 同形的布尔/0-1 数组，统计与归一范围限定 mask>0；
            None 表示全图统计并归一全部体素（论文默认）。
        eps: 防 std=0 的常数，默认 1e-8（依据见文件头 docstring）。
        zero_outside_mask: mask 变体下是否把 mask 外体素置 0。默认 False
            （保持原值，对齐官方 v2 与论文句面）；True 复现官方 v1。

    Returns:
        与输入同形的 float32 数组；全零/常数图因 eps 不产生 NaN/Inf。

    Raises:
        ValueError: mask 形状与 image 不一致，或 mask 内没有体素。
    """
    x = np.asarray(image, dtype=np.float32)
    if mask is None:
        mean = x.mean()
        std = x.std()
        # eps 防零除（全零图/常数图 std=0），依据见文件头 docstring
        return (x - mean) / (std + np.float32(eps))

    m = np.asarray(mask)
    if m.shape != x.shape:
        raise ValueError(f"mask 形状 {m.shape} 与影像形状 {x.shape} 不一致")
    region = x[m > 0]
    if region.size == 0:
        raise ValueError("mask 内没有任何体素（mask>0 为空）")
    mean = region.mean()
    std = region.std()
    # eps 防零除（mask 内全零/常数时 std=0），依据见文件头 docstring
    out = x.copy()
    # 论文："the normalization is applied within that mask only"——只改 mask 内
    out[m > 0] = (region - mean) / (std + np.float32(eps))
    if zero_outside_mask:
        out[m == 0] = 0
    return out


if __name__ == "__main__":
    # 自测：CPU、固定随机种子、数秒内完成；逐项打印 PASS <检查点>
    rng = np.random.default_rng(0)

    # ---- 用例 1：CT 全局方案（论文 CT 公式；官方 v1/v2 与论文一致的前景口径）----
    # 造 3 例 HU 风格合成影像：前景软组织 ~40±30 HU + 离群负值（空气）与
    # 离群正值（金属），验证百分位截断把离群值挡在 [P0.5, P99.5] 内
    ct_images, ct_masks = [], []
    for _ in range(3):
        vol = rng.normal(-900.0, 50.0, size=(16, 32, 32)).astype(np.float32)
        body = np.zeros_like(vol, dtype=bool)
        body[4:12, 8:24, 8:24] = True
        vol[body] = rng.normal(40.0, 30.0, size=int(body.sum()))
        vol[3, 16, 16] = 3000.0  # 金属类离群正值
        ct_images.append(vol)
        ct_masks.append(body)

    stats = compute_ct_global_stats(ct_images, ct_masks)
    # 统计量来自前景汇总：截断界必须夹住绝大多数前景体素（0.5/99.5 定义）
    fg_all = np.concatenate(
        [img[m > 0] for img, m in zip(ct_images, ct_masks)]
    )
    assert stats["clip_low"] <= np.percentile(fg_all, 0.5) + 1e-3
    assert stats["clip_high"] >= np.percentile(fg_all, 99.5) - 1e-3
    assert stats["n_voxels"] == int((np.stack(ct_masks) > 0).sum())

    out_ct = normalize_ct(ct_images[0], stats)
    assert out_ct.shape == ct_images[0].shape
    assert out_ct.dtype == np.float32
    assert np.isfinite(out_ct).all(), "CT 归一化输出含 NaN/Inf"
    # 截断生效：任何体素不得越过截断界映射出的数值区间
    z_low = (stats["clip_low"] - stats["mean"]) / (stats["std"] + DEFAULT_EPS)
    z_high = (stats["clip_high"] - stats["mean"]) / (stats["std"] + DEFAULT_EPS)
    assert out_ct.min() >= z_low - 1e-5 and out_ct.max() <= z_high + 1e-5
    # 前景均值≈0（截断使均值略偏，但应远小于 1 个 std）
    fg_out = out_ct[ct_masks[0] > 0]
    assert abs(fg_out.mean()) < 0.5
    print(f"PASS CT 全局方案：形状 {out_ct.shape}，截断界 [{z_low:.2f}, {z_high:.2f}]，"
          f"前景均值 {fg_out.mean():.4f}")

    # masks=None 扩展口径（无标注时可用；非论文/官方口径）也能跑通
    stats_all = compute_ct_global_stats(ct_images, masks=None)
    out_ct_all = normalize_ct(ct_images[0], stats_all)
    assert np.isfinite(out_ct_all).all() and out_ct_all.shape == ct_images[0].shape
    print("PASS CT 整图统计扩展口径（masks=None，非论文口径）：无 NaN/Inf，形状不变")

    # ---- 用例 2：MRI 逐图 z-score（论文非 CT 默认，全图口径）----
    mri = rng.normal(700.0, 200.0, size=(16, 32, 32)).astype(np.float32)
    out_mri = normalize_zscore(mri)
    assert out_mri.shape == mri.shape
    assert np.isfinite(out_mri).all(), "MRI 全图 z-score 输出含 NaN/Inf"
    # 全图口径：均值≈0，std≈1（eps 仅在分母加一次，std/(std+eps)≈1）
    assert abs(out_mri.mean()) < 1e-4
    assert abs(out_mri.std() - 1.0) < 1e-3
    print(f"PASS MRI 全图 z-score：均值 {out_mri.mean():.2e}，std {out_mri.std():.6f}")

    # ---- 用例 3：MRI mask 变体（论文 mask 变体公式：只改 mask 内）----
    m = mri > 600.0
    out_masked = normalize_zscore(mri, mask=m)
    assert out_masked.shape == mri.shape
    assert np.isfinite(out_masked).all(), "MRI mask z-score 输出含 NaN/Inf"
    region = out_masked[m]
    assert abs(region.mean()) < 1e-4, "mask 内均值应为 0"
    assert abs(region.std() - 1.0) < 1e-3, "mask 内 std 应为 1"
    # 论文句面："applied within that mask only"——mask 外保持原值
    assert np.array_equal(out_masked[~m], mri[~m]), "默认 mask 外体素应保持原值"
    # zero_outside_mask=True 复现官方 v1 的 mask 外置 0 口径
    out_zero = normalize_zscore(mri, mask=m, zero_outside_mask=True)
    assert (out_zero[~m] == 0).all(), "zero_outside_mask=True 时 mask 外应为 0"
    assert np.allclose(out_zero[m], out_masked[m]), "两种口径 mask 内结果应一致"
    print(f"PASS MRI mask 变体：mask 内均值 {region.mean():.2e}，std {region.std():.6f}；"
          f"mask 外默认保持原值 / 置 0 口径均验证")

    # ---- 用例 4：退化情形（全零图、常数图）不产生 NaN/Inf ----
    out_zero_img = normalize_zscore(np.zeros((4, 8, 8), dtype=np.float32))
    out_const = normalize_zscore(np.full((4, 8, 8), 5.0, dtype=np.float32))
    assert np.isfinite(out_zero_img).all() and np.isfinite(out_const).all()
    assert (out_zero_img == 0.0).all() and (out_const == 0.0).all()
    print("PASS 退化情形（全零图/常数图）：加 eps 后输出全 0，无 NaN/Inf")

    print("PASS 全部自测通过")
