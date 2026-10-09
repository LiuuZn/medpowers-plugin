# -*- coding: utf-8 -*-
"""hd95 — 95 分位 Hausdorff 距离（HD95 / HD@0.95）[metric]

论文支撑（统一四问）
====================
- 哪篇论文用了它：Abdel Aziz Taha & Allan Hanbury，《Metrics for evaluating
  3D medical image segmentation: analysis, selection, and tool》，BMC Medical
  Imaging 15:29，2015。所读版本：PMC 全文
  https://pmc.ncbi.nlm.nih.gov/articles/PMC4533825/ （DOI: 10.1186/s12880-015-0068-x）。
- 目的：该文系统评估 20 个分割评价指标。对 Hausdorff 距离（HD，式 48-49），
  论文原文指出 "The HD is generally sensitive to outliers. Because noise and
  outliers are common in medical segmentations, it is not recommended to use
  the HD directly"，并收录 Huttenlocher 等（论文参考文献 [41]）的分位数法：
  "the HD is defined to be the qth quantile of distances instead of the
  maximum, so that possible outliers are excluded"；其工具语法为
  `-use HD@0.9`（按 0.9 分位数计算 HD）。
- 效果：论文属指标评估研究，无分割精度提升数字；给出的数字是性质标注与
  效率——Table 4 标注 HD 为"离群敏感 + 位置敏感 + 距离型"，Table 5 选型
  指南建议"存在离群值时避免原始 HD（可用分位数法补救）、复杂边界/小目标/
  低密度/边界精度重要时推荐 HD"；其 C++ 工具对 240 例 BRATS2012 平均
  1.3 s 算完 HD，比 ITK 4.4.1 快 2.4 倍。把分位数固定为 95% 并成为标准
  报告量是其后 BraTS 挑战赛与 nnU-Net 的实践，非该论文数字。
- 结论：HD95 是抗离群值的边界贴合度不可导指标（越小越贴合），与 Dice 等
  重叠指标互补——重叠为零时 Dice 恒为 0，而距离型指标仍能反映空间远近；
  适合小目标、复杂边界、低密度分割与边界精度要求高的场景。

实现说明
========
关键公式与论文式子对应：
- 式 (48)：HD(A,B) = max(h(A,B), h(B,A))          -> hausdorff_distance()
- 式 (49)：h(A,B) = max_{a∈A} min_{b∈B} ||a−b||   -> directed_surface_distances()
  （||·|| 取欧氏范数，即论文所说 "some norm, e.g. Euclidean distance"）
- 分位数法（论文 "spatial distance based" 一节，引 Huttenlocher [41]）：
  以距离集的第 q 分位数取代式 (48)(49) 中的最大值 -> hausdorff_distance_95()
  （q=95 即 HD95；EvaluateSegmentation 语法 HD@0.95）
- 论文默认 "All distances calculated in this section are in voxel, which
  means the voxel size is not taken into account"：故 spacing=None 时按
  体素单位计算（与论文一致）；spacing 为可选参数（各轴 mm），是 BraTS/
  临床通行扩展，属于对论文的有意扩展而非替代。

参考的开源实现：
- medpy.metric.binary（__surface_distances / hd95）：nnU-Net v1 与 BraTS
  评估链路的事实标准。本文件的表面体素提取（binary_erosion 取反异或）、
  距离变换求有向最近邻、"两方向距离拼接后取 np.percentile(95)" 的聚合
  约定均与之对齐（约定层面参考，未逐行抄写；本地环境无 medpy 包）。
- 论文自带工具 EvaluateSegmentation（github.com/codalab/EvaluateSegmentation）：
  其在稠密体素云上"只需对表面/边界体素求最近邻"的优化论证（论文 Fig.1(5)(6)
  及 AVD 一节文字）是本实现取表面体素集合作为式 (49) 点集 A、B 的依据。
- 本地 /ssd/lx/metrics/metrics.py 与 totalsegmentator .venv 均无 HD 类实现
  （已核实），未参考本地代码。

与开源/论文不一致处及以论文为准的理由：
1. 论文式 (48)(49) 定义在一般点集上，未强制取表面；主流实现 medpy 把 A、B
   取为表面体素集（EvaluateSegmentation 经源码核实其 HD 在全体素上计算——
   查询集为 FP/FN 不一致体素、参照集为对方分割全部体素——并不取表面体素
   集，其"表面优化"只是论文层面的论证思路）。理由见上——
   查询点到对方分割的最近点必落在其表面，数学上对 min 侧等价；已知近似是
   查询侧取表面会漏掉"位于本分割内部但不在对方分割内"的体素（如对方有
   孔洞），medpy/nnU-Net 报告值均含此近似，为与其可比本实现保持一致。
2. 论文只说"用第 q 分位数代替最大值"，未指明两个方向如何合成；medpy hd95
   将两个方向的有向距离数组拼接后取分位数（reduction="concat"，默认，
   保证与 BraTS/nnU-Net 报告值可比）；reduction="max" 先各方向取分位数再
   取 max，更贴近论文式 (48) 的 max 结构。两种聚合本实现都提供并写明，
   对比他人数字时先确认聚合方式（论文"Multiple definition"一节的主旨）。
3. 论文用其近线性专用算法（参考文献 [39]）算 HD；本实现用精确欧氏距离
   变换（scipy，Maurer 线性时间精确 EDT）实现 min_{b∈B}——实现手段差异，
   数学对象一致（表面点集上的精确最近邻距离）。
4. 数值稳定：本指标无除法/log/exp/幂运算；数值风险点是空分割（距离数组
   为空时分位数无定义）——双空约定返回 0.0（两分割一致地无前景，视为
   完全一致），单空 hausdorff_distance_95 抛 ValueError（hd95_batch 批量
   场景改填 NaN 以不中断评估），全部显式处理而非静默。

布局约定：
- numpy 路径：2D 掩码 (H,W)、3D 掩码 (D,H,W)，bool / 非零为前景；
  spacing 与数组轴同序（2D=(H 间距, W 间距)，3D=(D,H,W 间距)）。
- torch 路径（hd95_batch）：2D=(B,C,H,W)、3D=(B,C,D,H,W)，逐 (样本,类别)
  计算，返回 float32 张量 (B,C)。本指标不可导，封装内部 detach 且在
  no_grad 下计算，只用于验证/报告，不进 loss。
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure

__all__ = [
    "surface_voxels",
    "directed_surface_distances",
    "hausdorff_distance",
    "hausdorff_distance_95",
    "hd95",
    "hd95_batch",
]


def surface_voxels(mask: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """提取二值分割的表面（边界）体素集合。

    参数:
        mask: 二值掩码 numpy 数组。布局：2D=(H,W) 或 3D=(D,H,W)；
              非 0 视为前景。
        connectivity: 形态学结构元连通性（2D 取 1~2，3D 取 1~3；1=最松）。
              medpy/nnU-Net 默认 1，只影响"表面"的判定口径。
    返回:
        与输入同形状的 bool 数组，True 处为表面体素。空分割返回全 False。
    """
    m = np.asarray(mask) != 0
    if m.ndim not in (2, 3):
        raise ValueError(f"仅支持 2D (H,W) 或 3D (D,H,W)，收到 ndim={m.ndim}")
    if not (1 <= connectivity <= m.ndim):
        raise ValueError(f"connectivity 须在 1~{m.ndim}，收到 {connectivity}")
    footprint = generate_binary_structure(m.ndim, connectivity)
    # binary_erosion 默认 border_value=0：贴图像边缘的体素腐蚀后消失，
    # 会被判定为表面——边缘即图像域边界，符合直觉（与 medpy 一致）。
    return m & ~binary_erosion(m, structure=footprint)


def directed_surface_distances(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: tuple | list | None = None,
    connectivity: int = 1,
) -> np.ndarray:
    """pred 表面每个体素到 gt 表面的最近欧氏距离集合（论文式 49 内层的 min_{b∈B}||a−b||）。

    参数:
        pred: 预测二值掩码，布局 2D=(H,W) 或 3D=(D,H,W)，非 0 为前景。
        gt: 真值二值掩码，与 pred 同形状。
        spacing: 各轴体素间距（与数组轴同序），None 时按体素单位计
              （论文默认："in voxel, voxel size is not taken into account"）。
        connectivity: 表面判定连通性，同 surface_voxels。
    返回:
        一维 float64 数组，长度 = pred 表面体素数，单位为体素或 spacing 单位。
    抛出:
        ValueError: pred 或 gt 为空分割（此时最近邻距离无定义）。
    """
    s_pred = surface_voxels(pred, connectivity)
    s_gt = surface_voxels(gt, connectivity)
    if not s_pred.any() or not s_gt.any():
        raise ValueError(
            "空分割上 Hausdorff 距离无定义（pred 表面体素数=%d，gt=%d）；"
            "批量场景请用 hd95_batch（双空=0.0，单空=NaN）" % (int(s_pred.sum()), int(s_gt.sum()))
        )
    # 对 ~s_gt 做精确欧氏距离变换（Maurer 线性时间算法，scipy 实现），
    # 每个体素的值即其到 gt 表面最近体素的欧氏距离——式 49 的 min 侧。
    dist_map = distance_transform_edt(~s_gt, sampling=spacing)
    return dist_map[s_pred]


def hausdorff_distance(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: tuple | list | None = None,
    connectivity: int = 1,
) -> float:
    """原始 Hausdorff 距离（论文式 48：HD = max(h(A,B), h(B,A))，在表面体素集上）。

    对离群值敏感（论文 Table 4），仅作对照；实际报告请用 hausdorff_distance_95。

    参数:
        pred/gt: 同 directed_surface_distances。
        spacing: 各轴体素间距，None 按体素单位。
        connectivity: 表面判定连通性。
    返回:
        float，HD 值（越小越贴合）。
    """
    d_ab = directed_surface_distances(pred, gt, spacing, connectivity)
    d_ba = directed_surface_distances(gt, pred, spacing, connectivity)
    return float(max(d_ab.max(), d_ba.max()))  # 式 (48) 的 max（两个有向 h 的较大者）


def hausdorff_distance_95(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: tuple | list | None = None,
    percentile: float = 95.0,
    connectivity: int = 1,
    reduction: str = "concat",
) -> float:
    """分位数 Hausdorff 距离（论文分位数法；percentile=95 即 HD95）。

    论文原文："the HD is defined to be the qth quantile of distances instead
    of the maximum, so that possible outliers are excluded"。q 以百分数传入
    （95.0 对应 EvaluateSegmentation 的 HD@0.95）。

    参数:
        pred/gt: 二值掩码 numpy 数组，布局 2D=(H,W) 或 3D=(D,H,W)，非 0 为前景。
        spacing: 各轴体素间距（与数组轴同序），None 按体素单位（论文默认）。
        percentile: 分位数百分数，0~100，默认 95.0（BraTS/nnU-Net 事实标准）。
        connectivity: 表面判定连通性，默认 1。
        reduction: 两方向距离的聚合方式：
              "concat" — 两个方向的有向距离数组拼接后取分位数
                         （medpy hd95 / nnU-Net v1 约定，默认，榜单可比）；
              "max"    — 两方向各自取分位数再取 max（贴近论文式 48 的 max 结构）。
    返回:
        float，HD@percentile 值（越小越贴合）。
    抛出:
        ValueError: 恰有一方为空分割（HD 无定义）；双方皆空返回 0.0。
    """
    if not (0.0 <= percentile <= 100.0):
        raise ValueError(f"percentile 须在 [0,100]，收到 {percentile}")
    if reduction not in ("concat", "max"):
        raise ValueError(f"reduction 须为 'concat' 或 'max'，收到 {reduction!r}")
    p_empty = not (np.asarray(pred) != 0).any()
    g_empty = not (np.asarray(gt) != 0).any()
    if p_empty and g_empty:
        return 0.0  # 双方一致地无前景：视为完全一致（本仓库自有约定；nnU-Net v1
                    # 默认对空分割返回 NaN，仅显式传 nan_for_nonexisting=False 才返回 0）
    d_ab = directed_surface_distances(pred, gt, spacing, connectivity)
    d_ba = directed_surface_distances(gt, pred, spacing, connectivity)
    if reduction == "concat":
        # medpy/nnU-Net v1 约定：拼接后取分位数
        return float(np.percentile(np.concatenate([d_ab, d_ba]), percentile))
    # 每个方向先取分位数，再按式 (48) 的 max 结构合成
    return float(max(np.percentile(d_ab, percentile), np.percentile(d_ba, percentile)))


def hd95(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: tuple | list | None = None,
    connectivity: int = 1,
) -> float:
    """HD95 的直呼别名：hausdorff_distance_95(..., percentile=95.0, reduction="concat")。

    参数与语义同 hausdorff_distance_95；与 BraTS/nnU-Net 报告口径一致。
    """
    return hausdorff_distance_95(pred, gt, spacing, 95.0, connectivity, "concat")


def hd95_batch(
    pred: torch.Tensor,
    gt: torch.Tensor,
    spacing: tuple | list | None = None,
    percentile: float = 95.0,
    connectivity: int = 1,
    reduction: str = "concat",
) -> torch.Tensor:
    """torch 批量封装：逐 (样本, 类别) 计算 HD@percentile。

    参数:
        pred/gt: torch 张量，布局 2D=(B,C,H,W) 或 3D=(B,C,D,H,W)；
              二值掩码（bool 或 0/1 浮点，非 0 为前景）。内部 detach+CPU，
              no_grad 计算——本指标不可导，仅用于验证/报告。
        spacing: 各轴体素间距（对全批相同），None 按体素单位。
        percentile/reduction: 同 hausdorff_distance_95。
    返回:
        torch.float32 张量，shape=(B,C)。
        空分割语义（批量场景为不中断评估，与单例函数刻意不同）：
        双空=0.0，单空=NaN。
    """
    import torch  # 延迟导入：numpy 路径不必装 torch

    with torch.no_grad():
        p = pred.detach().cpu()
        g = gt.detach().cpu()
    if p.ndim not in (4, 5) or p.ndim != g.ndim:
        raise ValueError(f"张量须为 (B,C,H,W) 或 (B,C,D,H,W) 且两者同维，收到 {tuple(p.shape)} 与 {tuple(g.shape)}")
    b, c = p.shape[0], p.shape[1]
    out = np.empty((b, c), dtype=np.float32)
    for i in range(b):
        for j in range(c):
            pm = (p[i, j].numpy() != 0)
            gm = (g[i, j].numpy() != 0)
            if not pm.any() and not gm.any():
                out[i, j] = 0.0
                continue
            if pm.any() != gm.any():
                out[i, j] = np.nan  # 单侧为空：HD 无定义，批量下以 NaN 标记
                continue
            out[i, j] = hausdorff_distance_95(pm, gm, spacing, percentile, connectivity, reduction)
    return torch.from_numpy(out)


# ---------------------------------------------------------------- 自测（CPU，秒级）

def _blob_2d(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """随机生成非空 2D 团块（H,W），用于自测。"""
    from scipy.ndimage import gaussian_filter

    noise = gaussian_filter(rng.random((h, w)), sigma=1.5)
    m = noise > np.percentile(noise, 60)
    return m if m.any() else m | (np.arange(h * w).reshape(h, w) == 0)


def _blob_3d(d: int, h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """随机生成非空 3D 团块（D,H,W），用于自测。"""
    from scipy.ndimage import gaussian_filter

    noise = gaussian_filter(rng.random((d, h, w)), sigma=1.0)
    m = noise > np.percentile(noise, 55)
    return m if m.any() else m | (np.arange(d * h * w).reshape(d, h, w) == 0)


def _brute_force(pred: np.ndarray, gt: np.ndarray, spacing, percentile: float, reduction: str) -> float:
    """暴力 O(|A||B|) 对照实现：直接按论文式 (48)(49) 在表面坐标集上双循环求最近邻。"""
    pa = np.argwhere(surface_voxels(pred)).astype(np.float64)
    pb = np.argwhere(surface_voxels(gt)).astype(np.float64)
    if spacing is not None:
        scale = np.asarray(spacing, dtype=np.float64)
        pa, pb = pa * scale, pb * scale
    diff = pa[:, None, :] - pb[None, :, :]
    dist = np.sqrt((diff ** 2).sum(-1))  # 幂运算仅发生在 float64 常规域，量级由输入尺寸界定
    d_ab, d_ba = dist.min(1), dist.min(0)  # 式 (49) 的 min 侧，两个方向
    if reduction == "concat":
        return float(np.percentile(np.concatenate([d_ab, d_ba]), percentile))
    return float(max(np.percentile(d_ab, percentile), np.percentile(d_ba, percentile)))


if __name__ == "__main__":
    rng = np.random.default_rng(20261005)  # 固定随机种子，保证自测可复现
    collected: list[float] = []  # 汇总全部数值输出，末尾统一断言无 NaN/Inf

    # 1) 解析用例：2D 两单体素相距 3（式 48/49 直算：距离集 {3},{3}）
    gt1 = np.zeros((8, 8)); gt1[0, 0] = 1
    pr1 = np.zeros((8, 8)); pr1[0, 3] = 1
    v_hd = hausdorff_distance(pr1, gt1)
    v_95 = hausdorff_distance_95(pr1, gt1)
    assert abs(v_hd - 3.0) < 1e-9 and abs(v_95 - 3.0) < 1e-9
    collected += [v_hd, v_95]
    print(f"PASS 1 论文式48/49解析值：两单体素相距3 -> HD={v_hd} HD95={v_95}")

    # 2) 完全一致分割 -> 0
    blob = _blob_2d(16, 16, rng)
    v0 = hausdorff_distance_95(blob, blob)
    assert v0 == 0.0
    collected.append(v0)
    print("PASS 2 完全一致分割：HD95=0")

    # 3) 与暴力 O(|A||B|) 实现在随机 2D/3D 上一致（EDT 实现的正确性对拍）
    blob2d_b = _blob_2d(16, 16, rng)
    blob3d_a = _blob_3d(8, 10, 12, rng)
    blob3d_b = _blob_3d(8, 10, 12, rng)
    for name, a, b, sp in [
        ("2D", blob, blob2d_b, None),
        ("3D", blob3d_a, blob3d_b, None),
        ("3D+spacing", blob3d_a, blob3d_b, (1.0, 0.8, 1.5)),
    ]:
        for reduction in ("concat", "max"):
            fast = hausdorff_distance_95(a, b, sp, 95.0, 1, reduction)
            slow = _brute_force(a, b, sp, 95.0, reduction)
            assert abs(fast - slow) < 1e-9, (name, reduction, fast, slow)
            collected += [fast, slow]
        # 式(48)：HD = max(h(A,B), h(B,A))，用暴力实现逐方向取 100 分位数对拍
        fast_hd = hausdorff_distance(a, b, sp)
        slow_hd = max(_brute_force(a, b, sp, 100.0, "concat"),
                      _brute_force(b, a, sp, 100.0, "concat"))
        assert abs(fast_hd - slow_hd) < 1e-9, (name, "HD", fast_hd, slow_hd)
        collected += [fast_hd, slow_hd]
    print("PASS 3 EDT 实现与暴力式48/49实现对拍一致（随机 2D/3D ± spacing，HD/HD95/concat/max）")

    # 4) 离群值敏感性复现（论文 Table 4 + 分位数法动机）：主结构完全一致 + 单个远端离群体素
    yy, xx = np.mgrid[0:28, 0:28]
    disk = (yy - 14) ** 2 + (xx - 14) ** 2 <= 5 ** 2  # 半径 5 圆盘，表面体素数 >30
    pr4 = disk.copy(); pr4[2, 2] = True  # 远端离群点
    hd_out = hausdorff_distance(pr4, disk)
    hd95_out = hausdorff_distance_95(pr4, disk)
    assert hd_out > 10.0, hd_out  # 离群点把 max 顶到 ~10+（离 (2,2) 最近盘面距离）
    assert hd95_out <= 1.0, hd95_out  # 95 分位把 <5% 的离群距离截掉
    collected += [hd_out, hd95_out]
    print(f"PASS 4 离群值复现：加单离群体素后 HD={hd_out:.3f} 但 HD95={hd95_out:.3f}（分位数法排除离群）")

    # 5) spacing 各向异性按轴缩放：体素 (0,0)->(3,0)，spacing=(2,1)（H 轴 2mm）-> 6.0
    gt5 = np.zeros((8, 8)); gt5[0, 0] = 1
    pr5 = np.zeros((8, 8)); pr5[3, 0] = 1
    v_sp = hausdorff_distance_95(pr5, gt5, spacing=(2.0, 1.0))
    assert abs(v_sp - 6.0) < 1e-9 and abs(hausdorff_distance_95(pr5, gt5) - 3.0) < 1e-9
    collected.append(v_sp)
    print(f"PASS 5 spacing 按轴缩放：无 spacing=3.0，spacing=(2,1) 时={v_sp}")

    # 6) 空分割语义：双空=0.0；单空抛 ValueError
    empty = np.zeros((8, 8))
    assert hausdorff_distance_95(empty, empty) == 0.0
    try:
        hausdorff_distance_95(pr5, empty)
        raise AssertionError("单空分割应抛 ValueError")
    except ValueError:
        pass
    print("PASS 6 空分割语义：双空=0.0，单空 ValueError")

    # 7) torch 批量封装：(B,C,H,W) 与 (B,C,D,H,W)，双空=0、单空=NaN、无梯度
    import torch

    gt7 = torch.zeros((2, 1, 16, 16)); gt7[0, 0, 4:12, 4:12] = 1  # 样本 1 双方皆空（见下）
    pr7 = gt7.clone(); pr7[0, 0, 4:12, 6:14] = 1; pr7[1, 0] = 0  # 样本 1 预测为空，真值本就为空 -> 双空
    out7 = hd95_batch(pr7, gt7)
    assert out7.shape == (2, 1) and out7.dtype == torch.float32
    assert torch.isfinite(out7).all() and float(out7[1, 0]) == 0.0
    pr7n = pr7.clone(); pr7n[1, 0, 3:9, 3:9] = 1  # 样本 1 变为单空 -> NaN
    out7n = hd95_batch(pr7n, gt7)
    assert torch.isnan(out7n[1, 0]) and torch.isfinite(out7n[0, 0])
    gt3d = torch.zeros((1, 1, 6, 12, 12)); gt3d[0, 0, 2:4, 4:8, 4:8] = 1
    out7b = hd95_batch(gt3d, gt3d)
    assert float(out7b[0, 0]) == 0.0
    x = torch.zeros((1, 1, 8, 8), requires_grad=True)  # 指标不可导：输出不应携带梯度
    out7c = hd95_batch((x > 0.5).float(), (x > 0.5).float())
    assert not out7c.requires_grad
    collected += [float(out7[0, 0]), float(out7b[0, 0]), float(out7c[0, 0])]
    print(f"PASS 7 torch 封装 (B,C,H,W)/(B,C,D,H,W)：shape={tuple(out7.shape)}，双空=0，单空=NaN，无梯度")

    # 8) 全部数值输出无 NaN/Inf（本指标无除法/log/exp，风险点只有空分割，已显式处理）
    arr = np.asarray(collected, dtype=np.float64)
    assert np.isfinite(arr).all(), arr
    print(f"PASS 8 全部 {arr.size} 个数值输出无 NaN/Inf")

    print("PASS 全部自测通过")
