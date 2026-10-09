# -*- coding: utf-8 -*-
"""foreground-crop — 前景（非零区域）包围盒裁剪 [preprocessing]

论文支撑（统一四问）
====================
- 哪篇论文用了它：Isensee, F., Jaeger, P.F., Kohl, S.A.A. 等（德国癌症研究中
  心 DKFZ），《nnU-Net: a self-configuring method for deep learning-based
  biomedical image segmentation》，Nature Methods 18:203-211，2021。
  https://www.nature.com/articles/s41592-020-01008-z
  所读版本：Nature 官方页正文付费墙（仅读到摘要）；方法细节读自官方补充材料
  PDF（Supplementary Notes 1-9，44 页，
  https://static-content.springer.com/esm/art%3A10.1038%2Fs41592-020-01008-z/MediaObjects/41592_2020_1008_MOESM1_ESM.pdf）
  与同团队 nnU-Net 方法论文全文 arXiv:1904.08128（ar5iv HTML，
  https://ar5iv.labs.arxiv.org/html/1904.08128 ），并与官方开源实现
  MIC-DKFZ/nnUNet、本地旧流水交叉核对。
- 目的：把每例图像裁到前景（非零体素）的最小包围盒，去掉大片值为 0 的空气
  背景。显存开销正比于体素数，裁剪直接降低显存占用；同等显存预算下等效提升
  有效分辨率/可用 patch 尺寸。裁剪也是 nnU-Net 预处理第一步，后续数据指纹
  （spacing、强度统计）与 patch size 决策都建立在裁剪后的数据上。
- 效果：论文未单列裁剪消融。原文："As a first processing step, nnU-Net crops
  the provided training cases to their nonzero region. While this had no effect
  on most datasets in our experiments, it reduced the image size of brain
  datasets such as D1 (Brain Tumor) and D15 (MSLes) substantially and thus
  improved computational efficiency."（裁剪把脑部数据集图像尺寸缩小
  "substantially"）。量化联动规则："If cropping resulted in an average size
  decrease of 25% or more, a mask for central non-zero voxels is created and
  the normalization is applied within that mask only."（缩小 >=25% 时归一化
  只在非零 mask 内做）。含此步骤的整条流水在 19 项国际竞赛超过大多数专用
  流水线，49 项任务中多数刷新 SOTA（Results：29/49 目标结构）。
- 结论：裁剪是 nnU-Net 流水线固定的第一步，无超参数、默认全模态启用；前景
  判定规则就是"非零体素"（官方实现逐通道 OR + binary_fill_holes）。注意论文
  同时说明该步骤对多数数据集无影响——它的价值集中在背景占比大的数据（脑部
  MRI、大 FOV CT）。

实现说明
========
本条目是数据流步骤，论文无公式；与论文的对应是三条文字规则：
1. "crops the provided training cases to their nonzero region"
   -> crop_to_nonzero()：前景 = 非零体素集合，取其逐轴最小包围盒后切片。
2. 25% 联动规则
   -> crop_volume_reduction()：返回裁剪缩减比例 1 - N_crop/N_orig，
      供归一化条目（intensity-normalization）判定是否只在非零 mask 内统计。
3. "The patch size is initialized as the median image shape after resampling."
   （patch size 最大值受 GPU 显存限制）-> 裁剪决定重采样的输入域，进而决定
   resample 后的中位形状与 patch size；对应本地 net_arch/arch_Unet3_2D.py:110
   从预处理 json 读 "target patient shape final" 作为 patch_size。

布局约定（预处理用 numpy 数组，spatial_dims 参数消歧）：
- 3D：spatial_dims=3（默认）。单通道 (Z, Y, X)（ndim=3）；多通道
  (C, Z, Y, X)（ndim=4，逐通道 OR 取并集）。
- 2D：spatial_dims=2。单通道 (H, W)（ndim=2）；多通道 (C, H, W)
  （ndim=3，必须显式传 spatial_dims=2，因为 ndim=3 默认按 3D 体数据解释）。
- 空间轴之外的轴（如通道 C）在裁剪中原样保留。
- seg/标签数组与图像空间形状一致，整数类别。

参考的开源实现：
- nnUNetv2 nnunetv2/preprocessing/cropping/cropping.py（create_nonzero_mask：
  data[c] != 0 逐通道 OR 后 binary_fill_holes；crop_to_nonzero：图像与 seg 同
  bbox 裁剪，并在 properties 记下 shape_before_cropping / bbox）。
- acvl_utils cropping_and_padding/bounding_boxes.py get_bbox_from_mask：包围盒
  为半开区间 [start, end)，end = 非零最大索引 + 1，可直接用于 python 切片。
- 本地旧流水 /ssd/lx/body_analysis/preprocessed/prepro_1_crop3D.py:87-141
  （_create_nonzero_mask/_get_bbox_from_mask/_crop_to_bbox 同构；crop_image_test
  返回 bbox、restore_mask_test 用 bbox 把预测掩码贴回原坐标系）。

与开源/论文不一致处及以论文为准的理由：
1. nnUNetv2 的 crop_to_nonzero 会把裁剪域内、非零掩码之外的 seg 体素置为
   nonzero_label=-1——这是配合训练采样的代码工程细节，论文未提。本实现默认
   nonzero_label=None（只裁剪不改 seg 值，论文口径）；显式传整数才启用官方约定。
2. acvl_utils 的 get_bbox_from_mask 用逐轴扫描省内存（大数组优化）；本实现用
   np.argwhere 一次取极值，语义等价（同为半开区间 [start, end)），小数据
   可读性优先。全零图时官方初始化 min=0/max=shape、退化为全图包围盒，本实现
   显式保持同一退化行为。
3. 官方 create_nonzero_mask 强制通道优先 (C,X,Y,Z)/(C,X,Y)；本实现默认把
   ndim=3 解释为单通道 3D (Z,Y,X)（与本地旧流水、SimpleITK 数组布局一致，
   也是医学预处理最常见的布局），多通道 3D 用 (C,Z,Y,X)、多通道 2D 显式
   spatial_dims=2——比官方更宽松，单通道语义与官方 (C=1) 完全一致。
4. binary_fill_holes 对包围盒本身无影响（填洞不改变极值），官方保留它是因为
   同一 mask 被复用于归一化统计域（对应论文 25% 规则）；本实现保留该步以与
   官方 mask 语义一致，供归一化条目复用。
5. 启用 nonzero_label 时与官方的写入口径仍有差异（官方
   nnunetv2/preprocessing/cropping/cropping.py 的 crop_to_nonzero）：官方只把
   (seg==0) & (~nonzero_mask) 的体素标为 nonzero_label（掩码外原有非零 seg
   值保留），且 seg=None 时会返回一个用 0/nonzero_label 构造的掩码 seg；
   本实现用 np.where 覆盖掩码外全部体素（含原有非零 seg 值），seg=None 时
   返回 None。nonzero_label 官方默认取 -1，与本实现文档/自测启用该功能时用
   的 -1 一致；此机制论文未提。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_fill_holes

__all__ = [
    "create_nonzero_mask",
    "get_bbox_from_mask",
    "crop_to_bbox",
    "crop_to_nonzero",
    "crop_volume_reduction",
]


def create_nonzero_mask(data: np.ndarray, spatial_dims: int | None = None) -> np.ndarray:
    """生成前景（非零）掩码：非零体素逐通道取并集后填补内部孔洞。

    参数
    ----
    data : np.ndarray
        输入图像数组。布局见模块 docstring：3D 单通道 (Z,Y,X)、
        3D 多通道 (C,Z,Y,X)、2D 单通道 (H,W)、2D 多通道 (C,H,W)。
    spatial_dims : int 或 None
        空间维数 2 或 3。None 时自动推断：ndim=2 -> 2；ndim=3 -> 3
        （即 (Z,Y,X) 体数据，多通道 2D (C,H,W) 必须显式传 2）；ndim=4 -> 3。

    返回
    ----
    mask : np.ndarray (bool)
        空间形状的布尔掩码（无通道轴），True 表示前景；已用
        binary_fill_holes 填补前景内部的零洞（官方 nnUNetv2 同款语义；
        对包围盒无影响，影响的是复用该 mask 的归一化统计域）。
    """
    data = np.asarray(data)
    if spatial_dims is None:
        if data.ndim == 2:
            spatial_dims = 2
        elif data.ndim in (3, 4):
            spatial_dims = 3
        else:
            raise ValueError(
                f"create_nonzero_mask: 期望 ndim 2/3/4，得到 ndim={data.ndim}；"
                f"2D 多通道 (C,H,W) 请显式传 spatial_dims=2"
            )
    if spatial_dims == 3:
        if data.ndim == 4:  # (C, Z, Y, X)：逐通道 OR 取并集（nnUNetv2 同款）
            nonzero = data[0] != 0
            for c in range(1, data.shape[0]):
                nonzero |= data[c] != 0
        elif data.ndim == 3:  # (Z, Y, X) 单通道
            nonzero = data != 0
        else:
            raise ValueError(
                f"spatial_dims=3 期望 (Z,Y,X) 或 (C,Z,Y,X)，得到 shape={data.shape}"
            )
    elif spatial_dims == 2:
        if data.ndim == 3:  # (C, H, W) 多通道（需显式 spatial_dims=2）
            nonzero = data[0] != 0
            for c in range(1, data.shape[0]):
                nonzero |= data[c] != 0
        elif data.ndim == 2:  # (H, W) 单通道
            nonzero = data != 0
        else:
            raise ValueError(
                f"spatial_dims=2 期望 (H,W) 或 (C,H,W)，得到 shape={data.shape}"
            )
    else:
        raise ValueError(f"spatial_dims 只支持 2 或 3，得到 {spatial_dims}")
    # 填补前景内部零洞（如囊肿、管腔）：官方实现细节；对 bbox 极值无影响
    return binary_fill_holes(nonzero)


def get_bbox_from_mask(mask: np.ndarray) -> list[list[int]]:
    """从布尔掩码求各轴最小包围盒（半开区间 [start, end)，可直接切片）。

    参数
    ----
    mask : np.ndarray (bool)
        空间形状布尔掩码（2D (H,W) 或 3D (Z,Y,X)）。

    返回
    ----
    bbox : list[list[int]]
        每轴 [start, end)，end = 非零最大索引 + 1（与官方 acvl_utils/
        nnUNetv2 的半开区间约定一致，python 切片 data[s:e] 直接可用）。
        全零掩码退化为全图包围盒 [0, shape)（与官方初始化行为一致）。
    """
    mask = np.asarray(mask, dtype=bool)
    coords = np.argwhere(mask)
    bbox = []
    for axis in range(mask.ndim):
        if coords.shape[0] == 0:
            # 全零掩码：退化为全图范围（官方 acvl_utils 初始化 min=0, max=shape）
            bbox.append([0, int(mask.shape[axis])])
        else:
            bbox.append([int(coords[:, axis].min()), int(coords[:, axis].max()) + 1])
    return bbox


def crop_to_bbox(array: np.ndarray, bbox: list[list[int]]) -> np.ndarray:
    """按包围盒裁剪数组（对最后 len(bbox) 个空间轴切片，通道轴原样保留）。

    参数
    ----
    array : np.ndarray
        (H, W) / (C, H, W) / (Z, Y, X) / (C, Z, Y, X) 布局数组。
    bbox : list[list[int]]
        每轴 [start, end) 半开区间，长度等于空间维数。

    返回
    ----
    cropped : np.ndarray
        同布局的裁剪结果。
    """
    array = np.asarray(array)
    n_spatial = len(bbox)
    if array.ndim not in (n_spatial, n_spatial + 1):
        raise ValueError(
            f"crop_to_bbox: bbox 有 {n_spatial} 个轴，与数组 ndim={array.ndim}"
            f"（空间轴 {array.ndim} 或 {array.ndim - 1}）不匹配"
        )
    # bbox 作用于末尾 n_spatial 个空间轴；前导轴（如通道 C）原样保留
    leading = (slice(None),) * (array.ndim - n_spatial)
    slicer = leading + tuple(slice(int(s), int(e)) for s, e in bbox)
    return array[slicer]


def crop_to_nonzero(
    data: np.ndarray,
    seg: np.ndarray | None = None,
    nonzero_label: int | None = None,
    spatial_dims: int | None = None,
) -> tuple[np.ndarray, np.ndarray | None, list[list[int]]]:
    """把图像（及可选标注）裁剪到非零前景包围盒——论文裁剪步骤的参考实现。

    对应论文原文："As a first processing step, nnU-Net crops the provided
    training cases to their nonzero region."

    参数
    ----
    data : np.ndarray
        图像，布局见模块 docstring（2D/3D、单/多通道）。
    seg : np.ndarray, 可选
        标注，空间形状与 data 一致（可含通道轴），整数类别；与 data 同 bbox
        裁剪，保证后续训练/重采样对齐。
    nonzero_label : int, 可选
        None（默认，论文口径）：seg 只裁剪不改值；
        整数（nnUNetv2 官方约定，论文未提）：裁剪域内非零掩码外的 seg 体素
        置为该标签，显式区分"图像有值但无标注"的区域。
    spatial_dims : int 或 None
        传给 create_nonzero_mask 的空间维数消歧参数（见其 docstring）。

    返回
    ----
    cropped_data : np.ndarray
        裁剪后图像（同布局）。
    cropped_seg : np.ndarray 或 None
        裁剪后标注（输入 seg 为 None 时返回 None）。
    bbox : list[list[int]]
        每轴 [start, end)，供推理时把结果贴回原坐标系持久化使用。
    """
    mask = create_nonzero_mask(data, spatial_dims=spatial_dims)
    bbox = get_bbox_from_mask(mask)
    cropped_data = crop_to_bbox(data, bbox)
    cropped_seg = None
    if seg is not None:
        cropped_seg = crop_to_bbox(np.asarray(seg), bbox)
        if nonzero_label is not None:
            # 官方 nnUNetv2 约定（工程细节，论文未提，默认关闭）：
            # 裁剪域内、掩码之外的体素显式标记，区分"有图像无标注"
            seg_mask = crop_to_bbox(mask, bbox)
            cropped_seg = np.where(
                seg_mask, cropped_seg, np.asarray(nonzero_label, dtype=cropped_seg.dtype)
            )
    return cropped_data, cropped_seg, bbox


def crop_volume_reduction(
    data: np.ndarray, spatial_dims: int | None = None, eps: float = 1e-12
) -> float:
    """计算裁剪到非零区域后的体素/像素缩减比例，用于论文 25% 联动规则。

    论文规则原文："If cropping resulted in an average size decrease of 25% or
    more, a mask for central non-zero voxels is created and the normalization
    is applied within that mask only." 缩减比例 >= 0.25 时，归一化条目
    （intensity-normalization）应只在非零 mask 内统计。

    口径提示：论文 25% 规则的 "average size decrease" 是跨训练集的数据集级
    平均缩减；本函数返回的是单例（单张图像）的缩减比例，数据集级判定需自行
    对多例结果求平均。

    参数
    ----
    data : np.ndarray
        裁剪前图像（见 crop_to_nonzero 的布局约定）。
    spatial_dims : int 或 None
        空间维数消歧参数（见 create_nonzero_mask）。
    eps : float
        分母保护项。原图体素数 N_orig 恒为正整数（构造时已 assert），
        理论上不会为 0；eps=1e-12 仅防御极端 dtype/溢出情形，不影响正常
        数值（相对 N_orig>=1，1e-12 完全可忽略）。

    返回
    ----
    reduction : float
        1 - N_crop / N_orig，钳制到 [0, 1)；全零（或全前景）图 ≈0
        （1e-14 量级，eps 影响所致，非精确 0）。
    """
    data = np.asarray(data)
    if spatial_dims is None:
        spatial_dims = 2 if data.ndim == 2 else 3
    mask = create_nonzero_mask(data, spatial_dims=spatial_dims)
    bbox = get_bbox_from_mask(mask)
    cropped = crop_to_bbox(data, bbox)
    shape_orig = data.shape[-spatial_dims:]
    shape_crop = cropped.shape[-spatial_dims:]
    numel_orig = int(np.prod(shape_orig))
    numel_crop = int(np.prod(shape_crop))
    assert numel_orig > 0, "crop_volume_reduction: 原图空间体素数必须为正"
    reduction = 1.0 - numel_crop / (numel_orig + eps)
    return float(min(max(reduction, 0.0), 1.0))


if __name__ == "__main__":
    # 自测：CPU、固定种子、秒级；预处理条目跑典型用例并断言形状/数值范围
    rng = np.random.default_rng(0)

    # ---- 用例 1：3D 单通道 (Z, Y, X)，已知包围盒 ----
    vol = np.zeros((64, 128, 128), dtype=np.float32)
    vol[10:30, 40:90, 20:70] = rng.random((20, 50, 50)).astype(np.float32) + 0.1
    cropped, _, bbox = crop_to_nonzero(vol)
    assert bbox == [[10, 30], [40, 90], [20, 70]], f"3D bbox 错误: {bbox}"
    assert cropped.shape == (20, 50, 50), f"3D 裁剪形状错误: {cropped.shape}"
    assert np.array_equal(cropped, vol[10:30, 40:90, 20:70]), "3D 裁剪值被改动"
    assert np.isfinite(cropped).all(), "3D 裁剪结果出现 NaN/Inf"
    print("PASS 用例1 3D 单通道包围盒与切片保真")

    # ---- 用例 2：3D 多模态 (C, Z, Y, X)，两通道前景取并集包围盒 ----
    multi = np.zeros((2, 32, 64, 64), dtype=np.float32)
    multi[0, 4:12, 8:24, 8:24] = 1.0
    multi[1, 16:28, 30:50, 40:60] = 2.0  # 第二通道前景更靠后，并集应覆盖两者
    cropped_m, _, bbox_m = crop_to_nonzero(multi)
    assert bbox_m == [[4, 28], [8, 50], [8, 60]], f"多模态并集 bbox 错误: {bbox_m}"
    assert cropped_m.shape == (2, 24, 42, 52)
    assert np.isfinite(cropped_m).all()
    print("PASS 用例2 3D 多模态逐通道 OR 并集包围盒")

    # ---- 用例 3：前景内部零洞（fill_holes 后包围盒不被洞分裂）----
    holed = np.zeros((20, 20, 20), dtype=np.float32)
    holed[5:15, 5:15, 5:15] = 1.0
    holed[8:12, 8:12, 8:12] = 0.0  # 内部零洞
    _, _, bbox_h = crop_to_nonzero(holed)
    assert bbox_h == [[5, 15], [5, 15], [5, 15]], f"含洞 bbox 错误: {bbox_h}"
    print("PASS 用例3 前景内部零洞不影响包围盒连续性")

    # ---- 用例 4：seg 对齐裁剪 + nonzero_label 官方约定开关 ----
    # 双目标 + 目标间空气：包围盒横跨两者，间隙图像为零且不在 fill_holes 掩码内
    seg = np.zeros((64, 128, 128), dtype=np.int8)
    seg[10:12, 40:60, 20:40] = 1  # 目标 A 的标注
    img4 = np.zeros((64, 128, 128), dtype=np.float32)
    img4[10:20, 40:60, 20:40] = 0.5  # 目标 A
    img4[22:30, 70:90, 50:70] = 0.5  # 目标 B（无标注，如未标注的相邻器官）
    gap = (img4 == 0)
    c_img, c_seg, bbox4 = crop_to_nonzero(img4, seg)  # 论文口径：只裁剪不改 seg 值
    assert c_seg.shape == c_img.shape == (20, 50, 50), "裁剪形状不符"
    assert bbox4 == [[10, 30], [40, 90], [20, 70]]
    assert set(np.unique(c_seg)) == {0, 1} and c_seg[0:2, 0:20, 0:20].min() == 1
    c_img2, c_seg2, _ = crop_to_nonzero(img4, seg, nonzero_label=-1)  # 官方口径
    gap_crop = crop_to_bbox(gap, bbox4)
    assert gap_crop.sum() > 0 and (c_seg2[gap_crop] == -1).all(), "掩码外间隙应标为 -1"
    assert (c_seg == 0)[gap_crop].all(), "论文口径下间隙 seg 不应被改动"
    print("PASS 用例4 seg 同 bbox 对齐裁剪与 nonzero_label 开关")

    # ---- 用例 5：2D 单通道 (H, W) 与多通道 (C, H, W) ----
    img2d = np.zeros((48, 64), dtype=np.float32)
    img2d[8:16, 20:40] = 1.0
    c2d, _, bbox2d = crop_to_nonzero(img2d)
    assert bbox2d == [[8, 16], [20, 40]] and c2d.shape == (8, 20)
    multi2d = np.zeros((3, 48, 64), dtype=np.float32)
    multi2d[1, 8:16, 20:40] = 1.0
    multi2d[2, 30:44, 50:60] = 1.0
    cm2d, _, bboxm2d = crop_to_nonzero(multi2d, spatial_dims=2)  # 多通道 2D 显式消歧
    assert bboxm2d == [[8, 44], [20, 60]] and cm2d.shape == (3, 36, 40)
    assert np.isfinite(cm2d).all()
    print("PASS 用例5 2D 单/多通道布局")

    # ---- 用例 6：全零图退化为全图（不崩溃、形状不变）----
    zero_vol = np.zeros((8, 9, 10), dtype=np.float32)
    cz, _, bbox_z = crop_to_nonzero(zero_vol)
    assert cz.shape == (8, 9, 10) and bbox_z == [[0, 8], [0, 9], [0, 10]]
    print("PASS 用例6 全零图退化为全图包围盒")

    # ---- 用例 7：坑复现——MRI 背景非零噪声点会把包围盒撑到全图 ----
    noisy = np.zeros((16, 64, 64), dtype=np.float32)
    noisy[4:12, 16:48, 16:48] = 1.0
    _, _, bbox_clean = crop_to_nonzero(noisy.copy())
    assert bbox_clean == [[4, 12], [16, 48], [16, 48]], f"无噪 bbox: {bbox_clean}"
    noisy[0, 0, 63] = 0.3  # 单个背景噪声点（对角远端）
    cn, _, bbox_n = crop_to_nonzero(noisy)
    assert bbox_n == [[0, 12], [0, 48], [16, 64]], f"噪声点 bbox: {bbox_n}"
    # 单个噪声点使 z 起点 4->0、y 起点 16->0、x 终点 48->64，包围盒整体膨胀
    # （论文只说 nonzero 启发式，未讨论该失效模式；工程上需对新数据检查）
    assert cn.shape[0] == 12 > bbox_clean[0][1] - bbox_clean[0][0] == 8
    print("PASS 用例7 背景噪声点撑大包围盒（非零启发式的已知失效模式）")

    # ---- 用例 8：裁剪缩减比例与 25% 联动规则数值 ----
    big = np.zeros((100, 200, 200), dtype=np.float32)
    big[40:60, 80:120, 80:120] = 1.0
    r_big = crop_volume_reduction(big)
    r_full = crop_volume_reduction(np.ones((10, 10, 10), dtype=np.float32))
    assert 0.0 < r_big < 1.0, f"缩减比例越界: {r_big}"
    # 前景 20x40x40=32000 / 全图 100x200x200=4e6 -> 缩减约 99.2%，远超 25%
    assert r_big > 0.25, f"大背景用例应触发 25% 规则: {r_big}"
    assert abs(r_full) < 1e-6, f"全前景图缩减应为 0: {r_full}"
    print(f"PASS 用例8 缩减比例数值（大背景 {r_big:.4f} >= 0.25 触发 mask 归一化）")

    print("PASS 全部自测通过")
