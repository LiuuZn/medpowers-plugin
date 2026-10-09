"""cc-dicece-loss —— 基于连通域实例级 CC-Metrics 的 Dice+CE 复合损失 [loss]
（资产卡：assets/objectives/cc-dicece-loss.md）

论文支撑（统一四问，依据论文原文 arXiv v3 全文，链接见下）：
- 哪篇论文用了它：Luc Bouteille, Alexander Jaus, Jens Kleesiek, Rainer
  Stiefelhagen, Lukas Heine（德国 TIO/IKIM 等团队），"Learning to Look Closer:
  A New Instance-Wise Loss for Small Cerebral Lesion Segmentation"，
  IEEE ISBI 2026，https://arxiv.org/abs/2511.17146
- 目的：脑内小病灶（腔隙 LAC、微出血 CMB、脑转移 SBM、白质高信号 WMH、
  BraTS 胶质瘤）体积占比极小，全局 Dice/DiceCE 中单个病灶的贡献可忽略，
  网络系统性漏检小病灶。论文基于 CC-Metrics 框架构造实例级损失 CC-DiceCE：
  对真值每个连通域（实例）取 Voronoi 区域，逐实例计算 Dice 与 CE，使每个
  真实病灶在损失中权重相等，与病灶体积无关。
- 效果：5 个脑 MRI 数据集 5 折交叉验证（nnU-Net 框架，CC-DiceCE 对 DiceCE
  基线）：instance recall LAC 0.3211→0.3698、CMB 0.5417→0.5780、
  BraTS 0.4513→0.5286；全局 Dice 几乎不变（最差 LAC −0.0113）；与 blob loss
  相比，25 个"指标-数据集"组合中 22 个更优。
- 结论：CC-DiceCE 提升 detection recall 而分割质量基本不动；precision 代价
  随数据集而异——BraTS 上 precision 0.7708→0.0517，小标注伪影被逐实例项
  放大。论文实验把本损失与全局 DiceCE 1:1 相加使用；WMH 等数据上需缩短
  训练日程（1000→150 epoch）以抑制过拟合。

实现说明
- 关键公式与论文章节的对应（所读版本：arXiv v3 全文 HTML，2026-04-06，
  第 II 节"Instance-wise losses"）：
  * Voronoi 区域（论文式(1)附近）：对真值前景 K 的每个 26 连通分量 C∈𝒦，
    R_C = { t ∈ T : d(t,C) < d(t,C′), ∀ C′ ∈ 𝒦\\{C} }（平局任意取向），
    d 为欧氏距离 —— voronoi_partition（连通域标记 + 距离变换 return_indices
    回贴最近分量标签，等价于逐体素取最近分量）；
  * CC-Metrics（论文式(2)）：V_m(P,K) = (1/|𝒦|)·Σ_{C∈𝒦} m(P∩R_C, C)，
    对每个实例等权平均 —— cc_dicece_loss 的逐实例均值口径；
  * 逐实例 Dice 项：1 − 2·Σ_{R_C}(p·g) / (Σ_{R_C} p + Σ_{R_C} g)（非平滑
    Dice，分母下限 eps 防 0/0）；逐实例 CE 项：R_C 内逐体素真值类
    −log p_class 的均值。CC-DiceCE = 逐实例 (Dice + CE) 的实例间均值再对批
    平均。真值某实例完全漏检时其 Dice 项 = 1（最差），权重 1/|𝒦| —— 结构性
    偏置 recall 的来源。
- 参考的开源实现（官方）：
  * TIO-IKIM/Learning-to-Look-Closer（本地落点 /tmp/mtskills-ref-cc-dicece）：
    - nnunetv2/training/loss/instance_losses.py 的 VectorizedCCLoss
      （mode="dice_ce"，训练器实际使用的快路径）：probs=softmax(logits)、
      ce_map=−log(probs.clamp_min(1e-7)) 逐体素真值类负对数、按 Voronoi
      区域 scatter 求和出 2·I/(P+G) 与区域 CE 均值、对有效实例取均值、
      批内取均值；
    - nnunetv2/utilities/connected_components.py 的 compute_voronoi_cupy /
      get_cc：3×3×3 全 1 结构元 = 26 连通标记，EDT return_indices 回贴；
    - nnunetv2/training/nnUNetTrainer/variants/loss/
      nnUNetTrainerInstanceLosses.py：训练器封装（deep supervision、
      LossMixer 与全局 DiceCE 组合）。
- 与官方实现不一致处（逐条编号，理由以论文/通用性为准）：
  1. 官方的连通域/Voronoi 用 CuPy 在 CUDA 上计算且强制输入为 CUDA 张量；
     本实现用 scipy.ndimage 在 CPU 上只对真值计算（离散预处理，不参与
     梯度），算法逐条对应（26 连通标记 + EDT 索引回贴）。距离平分体素的
     取向在 scipy 与 CuPy 间可能不同，只影响个别体素的区域归属，整体等价
     已由 GPU 对拍验证（见资产卡"参考实现"）。
  2. 官方强制 y.dtype == torch.int16（nnU-Net 数据管线约定）；本实现接受
     任意整型 dtype，并校验取值 ⊆ {0,1}，越界报 ValueError（官方此时会在
     one_hot 处直接崩溃）。
  3. 官方把概率强制 .to(torch.float32) 后再统计；本实现保留输入 dtype
     （float64 输入统计精度更高；与官方对拍时用 float32 对齐）。
  4. 官方仅支持 3D 二类（输入恰 2 通道）；本实现同样仅支持 3D
     (B,2,D,H,W)——2D 可按 8 连通自然推广，但论文与官方均未覆盖，有意不做。
  5. 官方在 nnU-Net 内叠加 deep supervision 加权与 LossMixer；论文实验把
     本损失与全局 DiceCE 1:1 相加（nnUNetTrainerGlobalCCDiceCE）。组合与
     深监督属训练器策略，不在本资产实现内。
- 数值稳定（与官方逐字对齐）：
  * Dice 分母 clamp_min(eps=1e-7)：正常实例区域的分母是区域软计数（≥1，
    区域至少含该连通域自身），1e-7 远小于任何非空区域计数量级，防的是空
    槽位与退化输入的 0/0；
  * CE：log 前对概率 clamp_min(1e-7)：softmax 在 fp32 下极端 logit 可下溢
    出精确 0，−log(1e-7)≈16.1 大而有限；
  * 退化路径：真值无前景（𝒦=∅）→ Dice 项取常数 1.0（无梯度，官方
    fallback 行为）+ 全图均值 CE；真值全前景 → EDT 返回自身索引、区域即
    连通域本身，数学自洽；两类路径在自测中断言有限。

布局与输入约定：pred=(B,2,D,H,W)——3D 二类 logits（未过 softmax，通道
0=背景、1=前景；from_logits=False 时为已 softmax 的概率）；target=(B,1,D,H,W)
整型 {0,1}；返回批内逐样本均值标量。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage

__all__ = ["cc_dicece_loss", "CCDiceCELoss", "voronoi_partition"]

_MODES = ("dice", "ce", "dice_ce")

# Dice 分母下限与 CE 的 log 前概率下限：与官方 VectorizedCCLoss 一致（取值
# 依据见文件头"数值稳定"）
_DEFAULT_EPS = 1e-7

# 26 连通结构元素（3D）：官方 compute_voronoi_cupy 同款 3×3×3 全 1 结构元
_STRUCTURE_3D = np.ones((3, 3, 3), dtype=bool)


def voronoi_partition(gt: np.ndarray) -> np.ndarray:
    """计算真值前景的 Voronoi 划分（论文 R_C 区域，离散体素版）。

    对 gt 前景做 26 连通标记，再对背景部分做欧氏距离变换取最近前景体素
    索引回贴，使每个体素归属最近连通分量——即论文 R_C = {t : d(t,C) <
    d(t,C′), ∀C′≠C} 的离散实现（平分距离时取向任意，同官方）。

    参数：
        gt: numpy.ndarray，3D 二值真值 (D,H,W)，非零视为前景。只作用于
            真值，不需要可导。

    返回：
        numpy.ndarray，(D,H,W) 整数标签：0 = 真值无前景时的占位（或称
        "空分量"槽位），1..K = 各前景实例的 Voronoi 区域（前景存在时覆盖
        全部体素）。
    """
    gt = np.asarray(gt)
    if gt.ndim != 3:
        raise ValueError(f"gt 应为 3D (D,H,W)，收到 ndim={gt.ndim}")
    labels, _ = ndimage.label(gt > 0, structure=_STRUCTURE_3D)
    if labels.max() == 0:  # 无前景：直接返回全 0，跳过 EDT（结果同官方全 0 槽位）
        return labels
    # 对背景（labels==0）做 EDT 取最近前景体素的索引；前景体素距离 0、索引
    # 指向自身，标签不变。真值全前景时 labels==0 为空数组，EDT 返回自身索引，
    # 区域=连通域本身，数学自洽。return_distances=False 时返回值就是索引数组。
    indices = ndimage.distance_transform_edt(
        labels == 0, return_distances=False, return_indices=True
    )
    return labels[tuple(indices)]


def _validate_pair(pred: torch.Tensor, target: torch.Tensor) -> None:
    """校验 pred/target 形状与取值（3D 二类，布局见文件头）。"""
    if pred.ndim != 5 or pred.shape[1] != 2:
        raise ValueError(
            f"pred 应为 (B,2,D,H,W) 的 3D 二类张量，收到 shape {tuple(pred.shape)}"
        )
    if target.numel() == 0:
        raise ValueError("target 为空张量")
    expected = (pred.shape[0], 1, *pred.shape[2:])
    if tuple(target.shape) != expected:
        raise ValueError(
            f"target 应为 {expected}（单通道二值标签），收到 {tuple(target.shape)}"
        )
    t = target.long()
    if int(t.min()) < 0 or int(t.max()) > 1:
        raise ValueError("target 取值必须 ⊆ {0,1}（二类标签）")


def _segment_sums(values: torch.Tensor, idx: torch.Tensor, nbins: int) -> torch.Tensor:
    """按体素所属区域标签分段求和（等价官方 scatter_add，bin 0 为空占位）。"""
    out = torch.zeros(nbins, dtype=values.dtype, device=values.device)
    return out.index_add(0, idx, values)


def _fallback_loss(mode: str, ce_map: torch.Tensor) -> torch.Tensor:
    """真值无前景（𝒦=∅）时的官方 fallback：dice 项常数 1.0（无梯度），
    CE 项为全图逐体素 CE 的均值。"""
    if mode == "dice":
        return ce_map.new_tensor(1.0)
    if mode == "ce":
        return ce_map.mean()
    return ce_map.mean() + 1.0


def _per_sample_loss(
    fg_prob: torch.Tensor,
    fg_true: torch.Tensor,
    ce_map: torch.Tensor,
    regions: torch.Tensor,
    mode: str,
    eps: float,
) -> torch.Tensor:
    """单个样本的实例级损失（官方 _vectorized_cc_reduction 逐样本口径）。

    参数：
        fg_prob: (D,H,W) 前景概率（float，已过 softmax）。
        fg_true: (D,H,W) 前景真值指示 ∈ {0,1}。
        ce_map: (D,H,W) 逐体素真值类 CE = −log(clamp(p_true, eps))。
        regions: (D,H,W) long，Voronoi 区域标签（0=空占位，1..K=实例）。
        mode: "dice" / "ce" / "dice_ce"。
        eps: 见文件头。

    返回：
        标量张量：有效实例（区域非空）上逐实例项的均值；无前景时 fallback。
    """
    k = int(regions.max().item())
    if k == 0:
        return _fallback_loss(mode, ce_map)

    flat = regions.reshape(-1)
    nbins = k + 1
    intersection = _segment_sums((fg_prob * fg_true).reshape(-1), flat, nbins)[1:]
    pred_sum = _segment_sums(fg_prob.reshape(-1), flat, nbins)[1:]
    true_sum = _segment_sums(fg_true.reshape(-1), flat, nbins)[1:]
    ce_sum = _segment_sums(ce_map.reshape(-1), flat, nbins)[1:]
    count = _segment_sums(torch.ones_like(fg_prob).reshape(-1), flat, nbins)[1:]

    # 逐实例项（官方 _combine_vectorized_loss_terms 同式）：
    # dice: 1 − 2I/(P+G)，分母 clamp_min(eps) 防空区域 0/0（正常实例分母≥1）
    # ce:   Σce/count，count≥1 的整数计数，clamp 仅护栏
    dice_loss = 1.0 - (2.0 * intersection / (pred_sum + true_sum).clamp_min(eps))
    ce_loss = ce_sum / count.clamp_min(1.0)
    if mode == "dice":
        component_loss = dice_loss
    elif mode == "ce":
        component_loss = ce_loss
    else:
        component_loss = dice_loss + ce_loss

    # 只对非空实例平均（CC 情形下 1..K 全部非空，此掩码为通用护栏，同官方）
    valid = (count > 0).to(component_loss.dtype)
    return (component_loss * valid).sum() / valid.sum().clamp_min(1.0)


def cc_dicece_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mode: str = "dice_ce",
    from_logits: bool = True,
    eps: float = _DEFAULT_EPS,
) -> torch.Tensor:
    """CC-DiceCE 损失：基于真值连通域 Voronoi 划分的实例级 Dice+CE（CC-Metrics）。

    每个真值实例（26 连通分量）的 Voronoi 区域内独立计算 Dice 损失与 CE
    损失，对所有实例等权平均——单个实例漏检贡献 ≈ 1/|𝒦| 的最差 Dice 项
    （结构性偏置 recall），单个假阳只污染自己所在区域的统计。

    参数：
        pred: torch.Tensor，(B,2,D,H,W)。from_logits=True 时为未过 softmax
            的 logits（通道 0=背景、1=前景，官方训练器语义）；False 时为已
            softmax 的概率。
        target: torch.Tensor，(B,1,D,H,W) 整型二值标签 {0,1}（任意整型
            dtype；真值侧连通域/Voronoi 用 scipy 在 CPU 计算，不参与梯度）。
        mode: "dice_ce"（默认，CC-Dice + CC-CE）/ "dice"（仅 CC-Dice）/
            "ce"（仅 CC-CE）。
        from_logits: pred 是否为 logits，默认 True（同官方 activation=softmax）。
        eps: Dice 分母下限与 CE 的 log 前概率下限，默认 1e-7（取值依据见
            文件头"数值稳定"）。

    返回：
        torch.Tensor 标量：批内逐样本损失的均值。典型量级：完美预测 ≈0；
        mode="dice_ce" 时单实例完全漏检的单样本损失上界约 2（dice≤1 +
        ce 有界）。

    异常：
        ValueError：形状不符、target 取值越界或 mode 非法。
    """
    if mode not in _MODES:
        raise ValueError(f"mode 只支持 {_MODES}，收到 {mode!r}")
    _validate_pair(pred, target)

    prob = torch.softmax(pred, dim=1) if from_logits else pred
    target_long = target.long()
    # 逐体素真值类 CE：−log(clamp(p_true, eps))——softmax 在 fp32 下可精确
    # 下溢为 0，clamp 防 log(0)（同官方 ce_map）
    ce_map = -torch.log(prob.clamp_min(eps)).gather(1, target_long).squeeze(1)
    fg_prob = prob[:, 1]
    fg_true = (target_long[:, 0] == 1).to(fg_prob.dtype)

    per_sample: list[torch.Tensor] = []
    for b in range(pred.shape[0]):
        # 真值侧 Voronoi：scipy 在 CPU 计算（离散预处理，不参与梯度），标签
        # 回贴到 pred 设备参与分段求和
        regions_np = voronoi_partition(target[b, 0].detach().cpu().numpy())
        regions = torch.from_numpy(np.ascontiguousarray(regions_np)).to(pred.device)
        per_sample.append(
            _per_sample_loss(
                fg_prob[b], fg_true[b], ce_map[b], regions.long(), mode, eps
            )
        )
    return torch.stack(per_sample).mean()


class CCDiceCELoss(nn.Module):
    """CC-DiceCE 损失的 nn.Module 封装，参数同 cc_dicece_loss。

    参数：
        mode: "dice_ce"（默认）/ "dice" / "ce"。
        from_logits: forward 收到的 pred 是否为 logits，默认 True。
        eps: 数值稳定下限，默认 1e-7。
    """

    def __init__(
        self,
        mode: str = "dice_ce",
        from_logits: bool = True,
        eps: float = _DEFAULT_EPS,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.from_logits = from_logits
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算 CC-DiceCE 损失，参数语义同 cc_dicece_loss。"""
        return cc_dicece_loss(
            pred, target, mode=self.mode, from_logits=self.from_logits, eps=self.eps
        )


# ---------------------------------------------------------------------------
# 自测：CPU 可跑、固定随机种子、数秒内完成
# ---------------------------------------------------------------------------
def _make_blobs(
    shape: tuple[int, int, int],
    centers: list[tuple[int, int, int]],
    radii: list[int],
) -> np.ndarray:
    """在给定形状内放置若干球形团块，返回二值前景掩码。"""
    fg = np.zeros(shape, dtype=bool)
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    for (cz, cy, cx), r in zip(centers, radii):
        fg |= (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
    return fg


def _naive_cc_dicece(
    prob: np.ndarray, target: np.ndarray, mode: str
) -> float:
    """朴素双精度参考实现：严格按论文公式逐实例循环（float64，无任何稳定项，
    独立于主实现的分段求和/钳制路径）。prob (B,2,D,H,W)、target (B,1,D,H,W)，
    概率须全部 >0。"""
    total = 0.0
    for b in range(prob.shape[0]):
        regions = voronoi_partition(target[b, 0])
        ids = [int(i) for i in np.unique(regions) if i > 0]
        if not ids:
            p_true = np.where(target[b, 0] == 1, prob[b, 1], prob[b, 0])
            ce = float(-np.log(p_true).mean())
            total += ce if mode == "ce" else ce + 1.0
            continue
        sample = 0.0
        for cid in ids:
            mask = regions == cid
            pf = prob[b, 1][mask]                     # 前景概率 p0
            is_fg = target[b, 0][mask] == 1           # 真值指示 g0
            p_true = np.where(is_fg, pf, prob[b, 0][mask])
            inter = pf[is_fg].sum()
            dice = 1.0 - 2.0 * inter / (pf.sum() + is_fg.sum())
            ce = float(-np.log(p_true).mean())
            sample += dice if mode == "dice" else ce if mode == "ce" else dice + ce
        total += sample / len(ids)
    return total / prob.shape[0]


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    # 1) Voronoi 划分：3 个分离小团块 → 3 个区域、全域覆盖、团块体素各归
    #    其分量；对角相邻两体素属同一分量（26 连通）
    blobs3 = _make_blobs(
        (32, 32, 32),
        [(6, 8, 8), (16, 20, 9), (26, 7, 22)],
        [2, 3, 2],
    )
    regions = voronoi_partition(blobs3.astype(np.uint8))
    assert set(np.unique(regions)) == {1, 2, 3}, np.unique(regions)
    # 每个团块的体素必须取得非零且互不相同的区域标签（前景体素归属自身分量）
    for cid in (1, 2, 3):
        sel = regions[blobs3 & (regions == cid)]
        assert sel.size > 0 and (sel == cid).all(), f"区域 {cid} 标签错乱"
    # 26 连通性：单点对角相邻
    diag = np.zeros((4, 4, 4), dtype=np.uint8)
    diag[0, 0, 0] = diag[1, 1, 1] = 1
    assert voronoi_partition(diag).max() == 1, "对角相邻应属同一 26 连通分量"
    print("PASS Voronoi 划分：3 团块 → 3 区域、前景全覆盖、26 连通正确")

    # 2) 与朴素双精度论文公式对拍：float64 概率输入，三种模式逐标量一致
    target_np = blobs3.astype(np.int64)[None, None]
    # 构造非平凡概率场（全部远离 0，使朴素版无钳制路径成立）
    dist_in = np.linalg.norm(
        np.stack(np.meshgrid(*[np.arange(32)] * 3, indexing="ij"), axis=0)[None]
        - np.array([[[6, 8, 8]], [[16, 20, 9]], [[26, 7, 22]]]).reshape(3, 3, 1, 1, 1),
        axis=1,
    )
    fg_p = np.clip(1.2 - dist_in / 12.0, 0.05, 0.95) + 0.05 * np.sin(dist_in * 0.7)
    fg_p = np.clip(fg_p, 0.02, 0.98)[0]
    prob_np = np.stack([1.0 - fg_p, fg_p], axis=0).astype(np.float64)[None]
    prob_t = torch.from_numpy(prob_np)
    target_t = torch.from_numpy(target_np)
    for m in _MODES:
        got = cc_dicece_loss(prob_t, target_t, mode=m, from_logits=False)
        ref = _naive_cc_dicece(prob_np, target_np, m)
        assert torch.isfinite(got)
        assert abs(float(got) - ref) < 1e-10, (m, float(got), ref)
        print(f"PASS 与朴素双精度论文公式对拍 mode={m}：max|Δ|={abs(float(got)-ref):.2e}")

    # 3) 方向性：漏检小实例 → CC 损失显著大于全部检出，且增幅远大于全局
    #    DiceCE 的增幅。这正是论文动机：小病灶漏检在全局 Dice 里只占其体积
    #    比例的损失，而实例级损失按 1/|𝒦| 记满最差 Dice 项。预测取接近真实
    #    输出的近二值概率场（团块上 0.9、远处 0.005），三个团块体积悬殊
    #    （r=5,4,2），漏检其中最小者
    blobs_mixed = _make_blobs(
        (32, 32, 32), [(8, 8, 8), (22, 20, 14), (6, 24, 27)], [5, 4, 2]
    )
    tgt_m = torch.from_numpy(blobs_mixed.astype(np.int64))[None, None]
    fg_field = np.where(blobs_mixed, 0.9, 0.001)       # 近二值：真值体素 0.9，远处 0.001
    prob_bin = np.stack([1.0 - fg_field, fg_field], axis=0).astype(np.float64)[None]
    p_all = torch.from_numpy(prob_bin)                 # 三团块均检出
    p_miss = p_all.clone()
    p_miss[0, 1, 3:10, 21:28, 24:30] = 0.001           # 抹掉最小实例（r=2）的预测
    p_miss[0, 0] = 1.0 - p_miss[0, 1]
    loss_all = cc_dicece_loss(p_all, tgt_m, from_logits=False)
    loss_miss = cc_dicece_loss(p_miss, tgt_m, from_logits=False)
    assert loss_miss > loss_all + 0.1, (loss_all.item(), loss_miss.item())

    def _global_dicece(p: torch.Tensor, t: torch.Tensor) -> float:
        pf, gt = p[..., 1, :, :, :], t[:, 0] == 1
        inter = (pf * gt).sum()
        dice = 1 - 2 * inter / (pf.sum() + gt.sum())
        ce = -torch.log(p.gather(2, t.permute(1, 0, 2, 3, 4)).squeeze(0)).mean()
        return float(dice + ce)

    delta_cc = float(loss_miss - loss_all)
    delta_global = _global_dicece(p_miss, tgt_m) - _global_dicece(p_all, tgt_m)
    assert delta_cc > 3 * abs(delta_global), (delta_cc, delta_global)
    print(
        f"PASS 方向性：漏检小实例 loss {loss_all.item():.4f}→{loss_miss.item():.4f}"
        f"（Δ_CC={delta_cc:.4f} ≈ {delta_cc/abs(delta_global):.1f}× Δ_全局DiceCE={delta_global:.4f}）"
    )

    # 4) 退化路径：真值全空（三模式有限、dice 恒 1、ce/dice_ce 有梯度）；
    #    真值全前景；预测全空——输出均有限
    empty_t = torch.zeros(1, 1, 8, 8, 8, dtype=torch.int64)
    logits0 = torch.zeros(1, 2, 8, 8, 8, requires_grad=True)
    l_dice = cc_dicece_loss(logits0, empty_t, mode="dice")
    assert float(l_dice) == 1.0 and torch.isfinite(l_dice)
    l_ce = cc_dicece_loss(logits0, empty_t, mode="ce")
    assert torch.isfinite(l_ce)
    l_dce = cc_dicece_loss(logits0, empty_t, mode="dice_ce")
    assert torch.isfinite(l_dce) and abs(float(l_dce.detach()) - 1.0 - float(l_ce.detach())) < 1e-7
    l_dce.backward()
    assert logits0.grad is not None and torch.isfinite(logits0.grad).all()
    full_t = torch.ones(1, 1, 8, 8, 8, dtype=torch.int64)
    assert torch.isfinite(cc_dicece_loss(torch.zeros(1, 2, 8, 8, 8), full_t))
    zero_pred = torch.stack(
        [torch.full((32, 32, 32), 30.0), torch.full((32, 32, 32), -30.0)], dim=0
    )[None]  # (1,2,32,32,32)，softmax → 前景概率≈2e-27：真全空预测
    # （dice 项=1，CE 真值类为背景不受影响）
    assert torch.isfinite(cc_dicece_loss(zero_pred, target_t))
    print("PASS 退化路径：全空真值（fallback=1+CE）、全前景真值、全空预测均有限")

    # 5) forward + backward：logits 入口，损失与梯度有限且非零
    torch.manual_seed(1)
    logits = torch.randn(2, 2, 32, 32, 32, requires_grad=True)
    tgt = torch.zeros(2, 1, 32, 32, 32, dtype=torch.int64)
    tgt[0, 0] = torch.from_numpy(blobs3.astype(np.int64))
    blobs3b = _make_blobs((32, 32, 32), [(5, 25, 25), (25, 15, 28), (12, 5, 3)], [2, 2, 3])
    tgt[1, 0] = torch.from_numpy(blobs3b.astype(np.int64))
    loss = CCDiceCELoss()(logits, tgt)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0, "梯度全零"
    print(f"PASS forward+backward：loss={loss.item():.4f}，梯度有限且非零")

    # 6) 批内混合退化样本：1 个样本真值全空 + 1 个正常样本，输出有限
    tgt_mixed = torch.zeros(2, 1, 32, 32, 32, dtype=torch.int64)
    tgt_mixed[1, 0] = tgt[0, 0]
    logits_mix = torch.randn(2, 2, 32, 32, 32)
    loss_mixed = cc_dicece_loss(logits_mix, tgt_mixed, from_logits=True)
    assert torch.isfinite(loss_mixed)
    # 手工口径核验：批均值 = (fallback 样本 + 正常样本) / 2（同一批 logits、
    # 同形状 32³ 逐样本拆算）
    solo_empty = cc_dicece_loss(
        logits_mix[:1], torch.zeros(1, 1, 32, 32, 32, dtype=torch.int64)
    )
    solo_full = cc_dicece_loss(logits_mix[1:2], tgt_mixed[1:2])
    manual = 0.5 * (float(solo_empty.detach()) + float(solo_full.detach()))
    assert abs(float(loss_mixed.detach()) - manual) < 1e-6, (float(loss_mixed), manual)
    print(f"PASS 批内混合退化样本：loss={float(loss_mixed.detach()):.4f}，与手工批均值一致")

    # 7) 极端 logits：完全反向的 ±60 置信预测——真值类概率在 fp32 下溢出
    #    1e-7 以下，走 clamp 路径（CE≈16.1/体素）；输出与梯度必须有限
    t_f = tgt.float()                                  # (2,1,32,32,32)
    fg_logit = 60.0 * (1.0 - t_f)                      # 有前景处预测"必无"、无前景处预测"必有"
    extreme = torch.cat([-fg_logit, fg_logit], dim=1)  # (2,2,32,32,32)
    extreme.requires_grad_(True)
    loss_e = cc_dicece_loss(extreme, tgt)
    loss_e.backward()
    assert torch.isfinite(loss_e) and torch.isfinite(extreme.grad).all()
    print(f"PASS 极端 logits ±60（反向置信、触发概率下溢钳制）：loss={float(loss_e.detach()):.4f}，梯度有限")

    print("PASS 全部自测通过")
