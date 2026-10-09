"""boundary-dou-loss —— 边界差集并集比损失（Boundary DoU Loss）[loss]（资产卡：assets/objectives/boundary-dou-loss.md）

论文支撑（统一四问；所读版本：arXiv:2308.00220 v1，2023-08-01，含 ar5iv 全文）：
- 哪篇论文用了它：Fan Sun、Zhiming Luo、Shaozi Li（厦门大学），
  "Boundary Difference Over Union Loss For Medical Image Segmentation"，
  MICCAI 2023（Springer LNCS，DOI 10.1007/978-3-031-43901-8_28），
  https://arxiv.org/abs/2308.00220 ；官方代码
  https://github.com/sunfan-bvb/BoundaryDoULoss （本地落点
  /tmp/mtskills-ref-boundary-dou，损失实现位于 TransUNet/utils.py）。
  注意作者为 Sun/Luo/Li，网上常见误记"Wang"系与其它边界损失文献混淆。
- 目的：现有医学影像分割损失大多只关注整体区域，少数边界损失（如 surface
  loss）需与其它损失组合且收益不稳。论文把 Dice 改写为"差集并集比"，对
  预测与真值的差集（边界不吻合部分）加权，用单一损失即可引导边界区域
  分割；并按目标尺寸自适应调整加在边界上的注意力（alpha）。
- 效果（论文表 1-3，ar5iv 全文）：Synapse 上 Swin-UNet 相比 Dice+CE 基线
  DSC 78.30 -> 79.87、HD 24.71±2.84 -> 19.80±2.34 mm、Boundary IoU
  86.72 -> 87.78（相比纯 Dice 基线 77.98 为 +1.89）；ACDC 上 Swin-UNet
  DSC 90.17 -> 91.02。分桶统计（表 3）：六个（模型 x 数据集）组合中小
  目标桶增益均大于大目标桶（如 Synapse/UNet：大目标 +2.29、小目标
  +2.46；ACDC/UNet：大目标 +1.13、小目标 +1.93）。
- 结论：Boundary DoU Loss 仅依赖区域计算（无距离变换、无形态学膨胀），
  单损失使用、训练稳定，UNet/TransUNet/Swin-UNet x ACDC/Synapse 一致
  有效；小目标（C/S >= 0.2）收益大于大目标。适用边界：论文实验为 2D
  切片多类分割；alpha 截断 0.8 出自官方代码而非论文（见下）。

实现说明
- 关键公式与论文式子的对应（论文正文无编号公式，按 ar5iv 全文 Methodology
  一节；P=预测、G=真值）：
  * L_DoU = (G∪P − G∩P) / (G∪P − α·G∩P)。记差集面积 S_D = |G∪P| − |G∩P|、
    交集 S_I = |G∩P|，则 L_DoU = S_D / (S_D + (1−α)·S_I)（论文同节给出的
    等价改写；"部分交集"权重 w = 1−α，w=2 时该式与论文所写
    L_Dice = 1 − 2S_I/(2S_I+S_D) = S_D/(S_D+2S_I) 严格相等，w=1 时为
    Jaccard 距离/IoU loss——自测第 4 项按此断言）。
  * 软化（可导）口径沿官方实现：|P| 以 Σp² 软计数、|G| = Σg²、
    S_I = Σp·g、|G∪P| = Σp² + Σg² − Σp·g。硬 mask（p,g ∈ {0,1}）下与集
    合基数严格相等（自测第 1/2/3 项用朴素集合实现验证；软概率路径由与
    官方实现的数值对拍覆盖，见资产卡）。
  * alpha 自适应（论文原文）：α = 1 − 2×(C/S)，α ∈ [0,1)，C 为目标边界
    长度、S 为目标面积；大目标判据 C/S < 0.2。边界长度按官方实现口径：
    3×3 十字邻域卷积（零填充）数"在目标内但 4 邻域+自身计数 ≠ 5"的像素。
    官方代码写法 alpha = 1 − (C+s)/(S+s) 后 2α−1（s=1e-5），与论文式相同。
  * 截断：官方代码对 α 做 min(α, 0.8)（代码注释：推荐 0.8 截断，部分数据
    集更好、其余影响很小）；论文未写截断。本实现保留参数 alpha_upper，
    默认 0.8 对齐官方；设 alpha_upper=1.0（α=1−2C/S 数学上恒 <1）即复原
    论文无截断版本。
- 参考的开源实现：sunfan-bvb/BoundaryDoULoss，TransUNet/utils.py 的
  BoundaryDoULoss（本地 /tmp/mtskills-ref-boundary-dou/TransUNet/utils.py）。
- 与官方实现不一致处（默认路径数值恒等；扩展项均为有意扩展）：
  1. 维度：官方仅 2D（3×3 十字卷积）；本实现补 3D（3×3×3 六邻域十字卷积，
     "计数 ≠ 7"判边界），为论文机制的自然扩展。
  2. 二类分支：官方只写多类 softmax 路径；本实现 C==1 时走 sigmoid，对
     单前景通道按二类计算。
  3. 输入口：官方 forward 恒做 softmax（输入必为 logits）；本实现加
     from_logits 开关，默认 True 与官方一致。
  4. reduction：官方恒返回"逐类（整批池化）损失之和 / 类数"标量；本实现补
     mean/sum/none，默认 mean 与官方恒等，none 返回逐类 (C,) 向量。聚合
     口径为官方的"每类整批求和成单比值再类平均"（逐样本口径数值不同，
     mean(比值) ≠ 比值(总和)，对比实验时须一致）。
  5. 向量化：官方逐 batch item 循环 conv2d；本实现以 groups=B·C 分组卷积
     一次算出。边界计数是 0/1 输入的小整数和，两种写法逐位相等，仅大
     求和（Σp² 等）的浮点归约顺序可能有 fp32 级差异（对拍见资产卡）。
  6. 官方 _adaptive_size 中 padding_out 为未使用的死代码，本实现不含。
  7. alpha_upper/smooth 参数化（官方硬编码 0.8/1e-5），默认值同官方。
- 数值稳定：smooth=1e-5 加在分子、分母与 α 的 C/S 计数三处（与官方同值
  同位）。依据：体素软计数单点量级为 1，1e-5 远小于任何非空计数量级
  （fp32 下对正常输入不改变损失值），并保证空类（S_I=S_D=0）时
  loss = s/s = 1（官方口径：空类梯度为 0）而非 0/0。对固定 α 路径强制
  校验 α < 1：α ≥ 1 时分母 S_D + (1−α)S_I 可能非正，会静默产生负损失。

布局约定：2D = (B, C, H, W)，3D = (B, C, D, H, W)；pred 默认为 logits
（未过 softmax/sigmoid）；target 为单通道类别索引 (B, 1, S...) 或
(B, S...)，或与 pred 同形的 one-hot（元素 0/1，非零视作前景）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["boundary_dou_loss", "BoundaryDoULoss", "boundary_alpha_from_mask"]

# 分子/分母/α 计数三处共用的平滑系数：官方同值（1e-5），依据见文件头"数值稳定"
_DEFAULT_SMOOTH = 1e-5
# 自适应 α 的截断上界：官方代码注释推荐 0.8；论文无截断（设 1.0 关闭）
_DEFAULT_ALPHA_UPPER = 0.8
# 固定 α 模式的默认取值（论文语义 α ∈ [-1, 1)：-1 ≡ Dice、0 ≡ IoU loss）
_DEFAULT_FIXED_ALPHA = 0.8


def _neighbour_count(mask: torch.Tensor) -> torch.Tensor:
    """计算每个像素/体素的"十字邻域计数"（含自身，零填充）。

    参数：
        mask: torch.Tensor，二值掩码 (N, S...)，N 为批量（可为 B、B·C），
            2D 空间 (H, W) / 3D 空间 (D, H, W)，元素非零视作前景。
    返回：
        torch.Tensor，与 mask 同形状的计数图；2D 计数 ∈ {0..5}（4 邻域
        十字 + 自身），3D ∈ {0..7}（6 邻域十字 + 自身）。零填充使贴边
        像素计数偏低、被判为边界（与官方实现一致）。
    """
    n = mask.shape[0]
    if mask.dim() == 3:  # 2D: (N, H, W)
        kernel = mask.new_tensor([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]])
        weight = kernel.view(1, 1, 3, 3).expand(n, 1, 3, 3).contiguous()
        # N 个单通道掩码作为 (1,N,H,W) 的 N 个通道，groups=N 逐掩码独立卷积
        return F.conv2d(mask.unsqueeze(0), weight, padding=1, groups=n).squeeze(0)
    if mask.dim() == 4:  # 3D: (N, D, H, W)
        kernel = mask.new_zeros(3, 3, 3)
        kernel[0, 1, 1] = 1.0
        kernel[2, 1, 1] = 1.0
        kernel[1, 0, 1] = 1.0
        kernel[1, 2, 1] = 1.0
        kernel[1, 1, 0] = 1.0
        kernel[1, 1, 2] = 1.0
        kernel[1, 1, 1] = 1.0
        weight = kernel.view(1, 1, 3, 3, 3).expand(n, 1, 3, 3, 3).contiguous()
        return F.conv3d(mask.unsqueeze(0), weight, padding=1, groups=n).squeeze(0)
    raise ValueError(f"mask 应为 (N,H,W) 或 (N,D,H,W)，收到 {tuple(mask.shape)}")


def _adaptive_alpha(mask: torch.Tensor, smooth: float, alpha_upper: float) -> torch.Tensor:
    """按官方/论文口径从二值掩码计算自适应 α（整批池化）。

    参数：
        mask: torch.Tensor，(B, C, S...) 的二值掩码（非零视作前景），
            2D 为 (B,C,H,W)、3D 为 (B,C,D,H,W)。
        smooth: 平滑系数（官方 1e-5，见文件头）。
        alpha_upper: 截断上界（官方 0.8；论文无截断时设 1.0）。
    返回：
        torch.Tensor，(C,)——每个通道一个 α；C、S 在整个 batch 上池化
        （与官方 torch.count_nonzero 全量统计一致，非逐样本）。
    """
    b, c = mask.shape[0], mask.shape[1]
    gt = (mask != 0).to(mask.dtype)
    cnt = _neighbour_count(gt.reshape(b * c, *gt.shape[2:])).view(b, c, *gt.shape[2:])
    total = 5.0 if mask.dim() == 4 else 7.0  # 4D=(B,C,H,W) 十字计数满值 5；5D 为 7
    band = cnt * gt                     # 只留目标内像素的邻域计数
    band = band * (band < total - 0.5)  # 计数==total（内部）清零 -> 边界带（官方 Y[Y==5]=0）
    per_class = band.transpose(0, 1).reshape(c, -1)
    c_len = per_class.count_nonzero(dim=1).float()  # C：边界长度（整批池化）
    s_size = gt.transpose(0, 1).reshape(c, -1).count_nonzero(dim=1).float()  # S：面积
    alpha = 1.0 - (c_len + smooth) / (s_size + smooth)  # 官方写法，数值同论文 α=1−2C/S
    alpha = 2.0 * alpha - 1.0
    return alpha.clamp(max=alpha_upper)


def boundary_alpha_from_mask(
    mask: torch.Tensor,
    smooth: float = _DEFAULT_SMOOTH,
    alpha_upper: float = _DEFAULT_ALPHA_UPPER,
) -> torch.Tensor:
    """从二值掩码计算论文的自适应边界权重 α = 1 − 2×(C/S)（截断后）。

    用于调参观察：大目标（C/S < 0.2）α 接近截断上界（边界注意力强），
    小目标/细长目标 C/S 大、α 趋向 −1（退回 Dice 语义）。

    参数：
        mask: torch.Tensor，二值掩码，支持 (H, W)（单张 2D）、(D, H, W)
            （单个 3D）、(B, 1, H, W)（2D 批）、(B, D, H, W)（3D 批）、
            (B, 1, D, H, W)。非零视作前景；批量维内整批池化成单个 α
            （官方口径：C、S 为全量统计）。
        smooth: 平滑系数，默认 1e-5（见文件头）。
        alpha_upper: 截断上界，默认 0.8（官方推荐；设 1.0 复原论文无截断）。
    返回：
        标量 torch.Tensor，α ∈ (-inf, alpha_upper]；α=−1 对应 Dice 语义、
        α=0 对应 IoU loss 语义、α 越大边界注意力越强。

    异常：
        ValueError：维度 layout 不在支持列表。
    """
    m = mask
    if m.dim() == 2:  # (H, W)
        m = m.view(1, 1, *m.shape)
    elif m.dim() == 3:  # (D, H, W)：单个 3D mask
        m = m.view(1, 1, *m.shape)
    elif m.dim() == 4 and m.shape[1] == 1:  # (B, 1, H, W)：2D 批
        pass
    elif m.dim() == 4:  # (B, D, H, W)：3D 批
        m = m.unsqueeze(1)
    elif m.dim() == 5:  # (B, 1, D, H, W)
        if m.shape[1] != 1:
            raise ValueError(f"通道维应为 1，收到形状 {tuple(m.shape)}")
    else:
        raise ValueError(f"不支持的掩码布局 {tuple(m.shape)}")
    return _adaptive_alpha(m, smooth, alpha_upper).squeeze(0)


def _to_prob(pred: torch.Tensor, from_logits: bool) -> torch.Tensor:
    """把网络原始输出转成每通道概率（C==1 sigmoid、C>1 softmax）。"""
    if not from_logits:
        return pred
    if pred.shape[1] == 1:
        return torch.sigmoid(pred)
    return F.softmax(pred, dim=1)


def _one_hot(target: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
    """把真值对齐到概率张量形状 (B,C,S...)。

    支持三种形式：与 prob 同形的 one-hot；单通道类别索引 (B,1,S...)；
    无通道维的类别索引 (B,S...)。类别索引必须可转 long 且取值 < C。
    """
    if target.shape == prob.shape:
        return target.to(dtype=prob.dtype)
    if target.dim() == prob.dim():  # (B,1,S...) 单通道类别索引
        if target.shape[1] != 1:
            raise ValueError(
                f"target 形状 {tuple(target.shape)} 的通道维应为 1（类别索引）"
                f"或与 pred 相同（one-hot）"
            )
        idx = target[:, 0].long()
    elif target.dim() == prob.dim() - 1:  # (B,S...) 类别索引
        idx = target.long()
    else:
        raise ValueError(
            f"target 形状 {tuple(target.shape)} 与 pred {tuple(prob.shape)} 不匹配"
        )
    oh = F.one_hot(idx, prob.shape[1])  # (B,S...,C)
    order = (0, oh.dim() - 1) + tuple(range(1, oh.dim() - 1))
    return oh.permute(*order).to(dtype=prob.dtype)


def boundary_dou_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    adaptive: bool = True,
    alpha: float = _DEFAULT_FIXED_ALPHA,
    alpha_upper: float = _DEFAULT_ALPHA_UPPER,
    from_logits: bool = True,
    smooth: float = _DEFAULT_SMOOTH,
    reduction: str = "mean",
    include_background: bool = True,
) -> torch.Tensor:
    """Boundary DoU 损失：L_DoU = S_D / (S_D + (1−α)·S_I)（官方软化口径）。

    α 是论文的边界权重（非 Tversky 的 FP/FN 权重）：α 越大，交集计入分母
    的比例 (1−α) 越小，边界（差集 S_D）主导损失；α=−1 时严格等于 Dice
    距离、α=0 时等于 IoU loss（Jaccard 距离）。默认按目标尺寸自适应
    （α = 1 − 2C/S，截断 0.8），即官方训练口径。

    参数：
        pred: 2D=(B,C,H,W) 或 3D=(B,C,D,H,W)。C>1 为多类 logits（默认过
            softmax）；C==1 为前景 logits（过 sigmoid）。
        target: 单通道类别索引 (B,1,S...) 或 (B,S...)，或与 pred 同形的
            one-hot（非零视作前景）。
        adaptive: True（默认，官方口径）按目标逐类自适应 α 并截断
            alpha_upper；False 用固定 α（此时不做截断）。
        alpha: 固定 α 模式（adaptive=False）的取值，须 < 1；论文语义
            α ∈ [-1, 1)。默认 0.8。
        alpha_upper: 自适应模式的截断上界，默认 0.8（官方推荐）；设 1.0
            复原论文无截断版本。
        from_logits: pred 是否为 logits，默认 True（同官方）。
        smooth: 分子/分母/α 计数三处平滑，默认 1e-5（依据见文件头）。
        reduction: "mean"（对类平均的标量，默认，同官方）/ "sum" / "none"
            （返回 (C_kept,) 逐类损失）。聚合口径为官方的"每类整批求和
            成单比值再类平均"，非逐样本口径。
        include_background: 是否把通道 0（背景类）也按独立二类参与计算，
            默认 True（官方逐类循环含背景；置 False 时对剩余类平均）。

    返回：
        torch.Tensor，取值 ∈ [0, 1]（α ≤ 1 时分母 ≥ 分子恒成立）；空类
        （真值全空）按官方口径给 loss=1、梯度 0。reduction 决定标量或
        (C_kept,)。

    异常：
        ValueError：形状不匹配、reduction 非法或固定 α ≥ 1。
    """
    prob = _to_prob(pred, from_logits)
    if prob.dim() not in (4, 5):
        raise ValueError(f"pred 应为 4D (B,C,H,W) 或 5D (B,C,D,H,W)，收到 {tuple(prob.shape)}")
    gt = _one_hot(target, prob)
    if not include_background and prob.shape[1] > 1:
        prob = prob[:, 1:]
        gt = gt[:, 1:]

    axes = list(range(2, prob.dim()))
    # 官方聚合口径：每个类在整批上求和（torch.sum 全量）成单比值
    inter = (prob * gt).sum(axes).sum(0)     # S_I = Σp·g，按类整批池化 -> (C,)
    z_sum = (prob * prob).sum(axes).sum(0)   # Σp²（软 |P|，官方口径）
    y_sum = (gt * gt).sum(axes).sum(0)       # Σg²（软 |G|，硬 mask 下即 |G|）

    if adaptive:
        a = _adaptive_alpha(gt, smooth, alpha_upper)  # (C,) 整批池化
    else:
        if not alpha < 1.0:
            raise ValueError(f"固定 α 必须 < 1（α≥1 时分母可为负），收到 {alpha}")
        a = torch.as_tensor(alpha, dtype=prob.dtype, device=prob.device)

    numerator = z_sum + y_sum - 2.0 * inter + smooth           # S_D + s（软差集）
    denominator = z_sum + y_sum - (1.0 + a) * inter + smooth   # S_D + (1−α)S_I + s
    loss = numerator / denominator

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"reduction 只支持 mean/sum/none，收到 {reduction!r}")


class BoundaryDoULoss(nn.Module):
    """Boundary DoU 损失的 nn.Module 封装，参数同 boundary_dou_loss。

    参数：
        adaptive: 是否按目标尺寸自适应 α（默认 True，官方口径）。
        alpha: 固定 α 模式的取值（adaptive=False 时生效），默认 0.8。
        alpha_upper: 自适应 α 的截断上界，默认 0.8（官方推荐）。
        from_logits: forward 收到的 pred 是否为 logits，默认 True。
        smooth: 平滑系数，默认 1e-5。
        reduction: "mean" / "sum" / "none"，默认 "mean"。
        include_background: 是否计入背景通道，默认 True（官方口径）。
    """

    def __init__(
        self,
        adaptive: bool = True,
        alpha: float = _DEFAULT_FIXED_ALPHA,
        alpha_upper: float = _DEFAULT_ALPHA_UPPER,
        from_logits: bool = True,
        smooth: float = _DEFAULT_SMOOTH,
        reduction: str = "mean",
        include_background: bool = True,
    ) -> None:
        super().__init__()
        self.adaptive = adaptive
        self.alpha = alpha
        self.alpha_upper = alpha_upper
        self.from_logits = from_logits
        self.smooth = smooth
        self.reduction = reduction
        self.include_background = include_background

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算 Boundary DoU 损失，参数语义同 boundary_dou_loss。"""
        return boundary_dou_loss(
            pred,
            target,
            adaptive=self.adaptive,
            alpha=self.alpha,
            alpha_upper=self.alpha_upper,
            from_logits=self.from_logits,
            smooth=self.smooth,
            reduction=self.reduction,
            include_background=self.include_background,
        )


# ---------------------------------------------------------------------------
# 朴素双精度参考实现（自测专用）：严格按论文公式的集合口径，显式逐像素
# 循环构造边界带集合 + 集合运算，独立于主实现的卷积/向量化路径。
# ---------------------------------------------------------------------------
def _naive_dou(
    prob: np.ndarray,
    gt: np.ndarray,
    alpha: float | None,
    smooth: float = 0.0,
    alpha_upper: float = 1.0,
) -> np.ndarray:
    """论文公式的朴素 float64 参考（硬 mask 集合口径）。

    参数：
        prob/gt: np.ndarray，(N, C, S...) 二值（硬）预测/真值，float64。
        alpha: 固定论文 α；None 表示按目标自适应（含 smooth 与截断，
            对应官方口径）。
        smooth: 分子/分母/α 计数平滑（纯论文式为 0；官方口径 1e-5）。
        alpha_upper: 自适应 α 截断上界。
    返回：
        np.ndarray，(C,) 逐类损失——官方聚合口径：S_D、S_I 按类在整批上
        求和成单比值（自适应 α 同样整批池化）。
    """
    n, c = gt.shape[0], gt.shape[1]
    spatial = gt.shape[2:]
    ndim = len(spatial)
    # 十字邻域偏移（2D 4 邻域 / 3D 6 邻域，不含自身）
    offsets = [
        d
        for d in np.ndindex(*(3,) * ndim)
        if d != (1,) * ndim and sum(1 for v in d if v != 1) == 1
    ]
    total = float(len(offsets) + 1)  # 邻域计数的"内部"满值：2D 为 5、3D 为 7

    def _boundary_length(g: np.ndarray) -> float:
        """目标内、十字邻域+自身计数 != total 的像素数（越界视作不在目标）。"""
        c_len = 0.0
        for idx in np.argwhere(g):
            cnt = 1.0  # 自身
            for off in offsets:
                nb = tuple(int(idx[k]) + off[k] - 1 for k in range(ndim))
                if all(0 <= nb[k] < spatial[k] for k in range(ndim)) and g[nb]:
                    cnt += 1.0
            if cnt != total:
                c_len += 1.0
        return c_len

    if alpha is None:
        # 官方口径：C、S 按类在整批上池化，每个通道一个 α
        alpha_c = np.zeros(c, dtype=np.float64)
        for j in range(c):
            c_len = 0.0
            s_size = 0.0
            for i in range(n):
                g = gt[i, j] > 0.5
                s_size += float(g.sum())
                c_len += _boundary_length(g)
            a = 1.0 - (c_len + smooth) / (s_size + smooth)
            a = 2.0 * a - 1.0
            alpha_c[j] = min(a, alpha_upper)

    out = np.zeros(c, dtype=np.float64)
    for j in range(c):
        s_i = 0.0
        s_d = 0.0
        for i in range(n):
            g = gt[i, j] > 0.5
            p = prob[i, j] > 0.5
            s_i += float(np.logical_and(p, g).sum())    # |P∩G|
            s_d += float(np.logical_or(p, g).sum()) - float(np.logical_and(p, g).sum())
        a = alpha if alpha is not None else float(alpha_c[j])
        out[j] = (s_d + smooth) / (s_d + (1.0 - a) * s_i + smooth)
    return out


# ---------------------------------------------------------------------------
# 自测：CPU 可跑、固定随机种子、数秒内完成
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    def _hard_masks_2d(n: int, c: int, h: int, w: int, fill: float = 0.35) -> np.ndarray:
        """生成每个 (n,c) 通道都非空的随机硬 mask（float64 0/1）。"""
        while True:
            m = (np.random.rand(n, c, h, w) > 1.0 - fill).astype(np.float64)
            if all(m[i, j].any() for i in range(n) for j in range(c)):
                return m

    # 1) 论文公式对拍（2D，固定 α=0.3，硬 mask 直接作概率输入，smooth=0
    #    即纯论文式）：主实现的软计数代数在二值输入下必须与朴素集合运算
    #    逐元素一致
    masks2d = _hard_masks_2d(2, 3, 16, 16)
    got = boundary_dou_loss(
        torch.from_numpy(masks2d), torch.from_numpy(masks2d),
        adaptive=False, alpha=0.3, from_logits=False, smooth=0.0, reduction="none",
    ).numpy()
    ref = _naive_dou(masks2d, masks2d, alpha=0.3)
    assert np.isfinite(got).all() and np.abs(got - ref).max() < 1e-12, (got, ref)
    print(f"PASS 论文公式对拍 2D 固定 α：max|Δ|={np.abs(got - ref).max():.2e}")

    # 2) 论文公式对拍（3D，固定 α=0.5）：验证六邻域 3D 扩展的边界带与集合运算
    m3 = (np.random.rand(1, 2, 8, 12, 12) > 0.65).astype(np.float64)
    m3[0, 0, 2:6, 4:9, 4:9] = 1.0  # 保证两通道非空
    m3[0, 1, 1:4, 3:8, 2:7] = 1.0
    got3 = boundary_dou_loss(
        torch.from_numpy(m3), torch.from_numpy(m3),
        adaptive=False, alpha=0.5, from_logits=False, smooth=0.0, reduction="none",
    ).numpy()
    ref3 = _naive_dou(m3, m3, alpha=0.5)
    assert np.isfinite(got3).all() and np.abs(got3 - ref3).max() < 1e-12, (got3, ref3)
    print(f"PASS 论文公式对拍 3D 固定 α（六邻域扩展）：max|Δ|={np.abs(got3 - ref3).max():.2e}")

    # 3) 官方口径全链路对拍（2D 多类：硬 one-hot 作概率、自适应 α+平滑+0.8
    #    截断、含背景逐类整批池化）——含一个空类（通道 3 全零）钉住空类路径
    idx3 = torch.randint(0, 3, (2, 1, 24, 24))  # 只用类 0-2 -> 类 3 为空类
    onehot64 = F.one_hot(idx3[:, 0].long(), 4).permute(0, 3, 1, 2).double()
    got_adapt = boundary_dou_loss(
        onehot64, onehot64, adaptive=True, from_logits=False, reduction="none",
    ).numpy()
    ref_adapt = _naive_dou(onehot64.numpy(), onehot64.numpy(), alpha=None,
                           smooth=1e-5, alpha_upper=0.8)
    assert np.isfinite(got_adapt).all() and np.abs(got_adapt - ref_adapt).max() < 1e-12
    print(f"PASS 官方口径全链路对拍（自适应 α+截断+空类）：max|Δ|={np.abs(got_adapt - ref_adapt).max():.2e}")

    # 4) 与 Dice/IoU 的关系（论文给出 L_Dice = 1 − 2S_I/(2S_I+S_D) =
    #    S_D/(S_D+2S_I)）：α=−1 严格回到 Dice 距离、α=0 严格为 IoU loss
    #    （Jaccard 距离）；且固定预测下 loss 随 α 单调增（α 越大边界注意力越强）
    masks = _hard_masks_2d(1, 1, 16, 16, fill=0.4)
    g = masks[0, 0] > 0.5
    rng_p = masks[0, 0].copy()
    flip = np.random.rand(16, 16) < 0.1
    rng_p[flip] = 1.0 - rng_p[flip]  # 构造非完美预测（仍为 0/1）
    s_i = float(np.logical_and(rng_p > 0.5, g).sum())
    s_d = float(np.logical_or(rng_p > 0.5, g).sum()) - s_i
    pred_t = torch.from_numpy(rng_p[None, None])
    gt_t = torch.from_numpy(masks)
    loss_m1 = boundary_dou_loss(pred_t, gt_t, adaptive=False, alpha=-1.0,
                                from_logits=False, smooth=0.0).item()
    loss_0 = boundary_dou_loss(pred_t, gt_t, adaptive=False, alpha=0.0,
                               from_logits=False, smooth=0.0).item()
    dice_dist = s_d / (s_d + 2.0 * s_i)
    iou_dist = s_d / (s_d + s_i)
    assert abs(loss_m1 - dice_dist) < 1e-12, (loss_m1, dice_dist)
    assert abs(loss_0 - iou_dist) < 1e-12, (loss_0, iou_dist)
    losses_alpha = [
        boundary_dou_loss(pred_t, gt_t, adaptive=False, alpha=a,
                          from_logits=False, smooth=0.0).item()
        for a in (-1.0, -0.5, 0.0, 0.5, 0.8)
    ]
    assert all(b > a_ for a_, b in zip(losses_alpha, losses_alpha[1:])), losses_alpha
    print(
        f"PASS 家族关系：α=−1 ≡ Dice 距离（{loss_m1:.6f}）、α=0 ≡ IoU loss"
        f"（{loss_0:.6f}）、loss 随 α 单调增（边界注意力增强）"
    )

    # 5) forward+backward（2D 多类 logits，默认官方口径）：损失与梯度有限
    logits32 = torch.randn(2, 4, 32, 32, requires_grad=True)
    tgt = torch.randint(0, 4, (2, 1, 32, 32))
    loss2d = BoundaryDoULoss()(logits32, tgt)
    loss2d.backward()
    assert torch.isfinite(loss2d) and torch.isfinite(logits32.grad).all()
    print(f"PASS 2D 多类 forward+backward：loss={loss2d.item():.4f}，梯度有限")

    # 6) forward+backward（3D logits）：损失与梯度有限
    logits3d = torch.randn(1, 3, 8, 24, 24, requires_grad=True)
    tgt3d = torch.randint(0, 3, (1, 1, 8, 24, 24))
    loss3d = BoundaryDoULoss(from_logits=True)(logits3d, tgt3d)
    loss3d.backward()
    assert torch.isfinite(loss3d) and torch.isfinite(logits3d.grad).all()
    print(f"PASS 3D forward+backward：loss={loss3d.item():.4f}，梯度有限")

    # 7) 退化路径：完美预测 -> ≈0；空类（one-hot 全零）-> 恰为 1（官方口径）；
    #    极端 logits（全 +30）、全图目标（贴边计数）均有限
    idx_p = torch.randint(0, 4, (1, 1, 16, 16))
    oh_p = F.one_hot(idx_p[:, 0].long(), 4).permute(0, 3, 1, 2).float()
    perfect = boundary_dou_loss(oh_p, oh_p, from_logits=False)
    assert perfect.item() < 1e-6, perfect.item()
    empty_cls = boundary_dou_loss(torch.randn(1, 3, 8, 8), torch.zeros(1, 3, 8, 8))
    assert abs(empty_cls.item() - 1.0) < 1e-6, empty_cls.item()
    extreme = BoundaryDoULoss()(
        torch.full((1, 4, 8, 8), 30.0), torch.zeros(1, 1, 8, 8, dtype=torch.long)
    )
    full_img = BoundaryDoULoss()(
        torch.randn(1, 2, 8, 8), torch.ones(1, 1, 8, 8, dtype=torch.long)
    )
    assert torch.isfinite(extreme) and torch.isfinite(full_img)
    print(
        f"PASS 退化路径：完美 {perfect.item():.2e}≈0，空类 {empty_cls.item():.4f}=1，"
        f"极端 logits/全图目标均有限（{extreme.item():.4f}/{full_img.item():.4f}）"
    )

    # 8) 小目标方向性：预测在小目标边缘外溢时损失必须高于贴合预测
    small = np.zeros((1, 1, 32, 32), dtype=np.float32)
    small[0, 0, 14:20, 14:20] = 1.0  # 6×6 小目标
    exact = torch.from_numpy(small)
    spill_arr = small.copy()
    spill_arr[0, 0, 13, 14:20] = 1.0
    spill_arr[0, 0, 20, 14:20] = 1.0
    spill_arr[0, 0, 14:20, 13] = 1.0
    spill_arr[0, 0, 14:20, 20] = 1.0
    spill = torch.from_numpy(spill_arr)
    l_exact = boundary_dou_loss(exact, exact, from_logits=False)
    l_spill = boundary_dou_loss(spill, exact, from_logits=False)
    assert l_spill > l_exact, (l_exact.item(), l_spill.item())
    print(f"PASS 小目标外溢方向性：外溢 {l_spill.item():.4f} > 贴合 {l_exact.item():.4f}")

    # 9) C==1 二类分支（sigmoid 扩展）+ 取值范围 [0,1]
    bin_logits = torch.randn(2, 1, 16, 16, requires_grad=True)
    bin_gt = (torch.rand(2, 1, 16, 16) > 0.8).float()
    l_bin = BoundaryDoULoss(from_logits=True)(bin_logits, bin_gt)
    l_bin.backward()
    assert torch.isfinite(l_bin) and torch.isfinite(bin_logits.grad).all()
    assert 0.0 <= l_bin.item() <= 1.0, l_bin.item()
    print(f"PASS C==1 二类分支：loss={l_bin.item():.4f} ∈ [0,1]，梯度有限")

    # 10) reduction 口径与逐类形状："none" 的均值 == "mean"；固定 α ≥ 1 报错
    x10 = torch.randn(2, 4, 8, 8)
    t10 = torch.randint(0, 4, (2, 1, 8, 8))
    ln = BoundaryDoULoss(reduction="none")(x10, t10)
    lm = BoundaryDoULoss()(x10, t10)
    assert ln.shape == (4,), ln.shape
    assert abs(ln.mean().item() - lm.item()) < 1e-6, (ln.mean().item(), lm.item())
    try:
        boundary_dou_loss(
            torch.randn(1, 2, 8, 8), torch.randint(0, 2, (1, 1, 8, 8)),
            adaptive=False, alpha=1.0,
        )
        raise AssertionError("α=1 未被拒绝")
    except ValueError:
        pass
    print("PASS reduction/形状一致性与 α≥1 校验")

    print("PASS 全部自测通过")
