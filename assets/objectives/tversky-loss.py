"""tversky-loss —— Tversky 损失函数 [loss]（资产卡：assets/objectives/tversky-loss.md）

论文支撑（统一四问，依据论文原文 arXiv v1 全文，链接见下）：
- 哪篇论文用了它：Seyed Sadegh Mohseni Salehi, Deniz Erdogmus, Ali Gholipour，
  "Tversky loss function for image segmentation using 3D fully convolutional
  deep networks"，MICCAI 2017 MLMI workshop（Springer LNCS 10541, pp. 379-387），
  https://arxiv.org/abs/1706.05721
- 目的：多发性硬化（MS）病灶 MRI 分割中非病灶体素超过病灶 500 倍以上，Dice loss
  作为精确率与召回率的调和平均对 FP/FN 等权，训练出的网络偏向高精确、低召回
  （漏检）；临床中假阴性比假阳性更不可接受。论文以 Tversky 指数为损失，用 α/β
  显式惩罚 FP/FN，把训练推向可调的 precision-recall 平衡。
- 效果：15 例 T1/T2/FLAIR MRI 二折交叉验证，(α,β)=(0.3,0.7) 相比 Dice loss
  (α,β)=(0.5,0.5)：DSC 53.42%→56.42%（约 +3），F2 51.77%→57.32%（约 +5.5），
  PR 曲线下面积 APR 52.57%→56.04%（约 +3.5）；低病灶密度病例上 Dice 网络完全
  漏检、β=0.7 网络成功检出。
- 结论：Tversky 指数是 Dice（α=β=0.5）、Tanimoto（α=β=1）、F_β（α+β=1）的
  广义框架；β 越大敏感度单调上升、特异度微降，综合指标在 β=0.7 最优。属概念
  验证研究（仅 15 人数据、未与当时 SOTA 直接对比），作者建议在更大标准数据集
  与多类任务上进一步验证。

实现说明
- 关键公式与论文式子编号的对应（所读版本：arXiv v1，2017-06-18）：
  * 论文式(2) Tversky 指数 S(P,G;α,β) = |P∩G| / (|P∩G| + α|P\\G| + β|G\\P|)，
    α 是 FP（|P\\G|，预测为病灶而真值不是）的权重、β 是 FN（|G\\P|，漏掉的
    病灶）的权重 —— 见 tversky_index_numpy（集合/软计数口径）与张量实现；
  * 论文式(3) softmax 输出形式 T(α,β) = Σ_i p0i·g0i / (Σ_i p0i·g0i +
    α·Σ_i p0i·g1i + β·Σ_i p1i·g0i)，p0i 为体素 i 属病灶的概率、g0i/g1i 为真值
    指示 —— tversky_loss；论文式(3)定义指数 T 并要求 cost minimization，
    最小化其补 1 - T 是通行推断（与主流实现一致）；
  * 论文式(4)(5) 给出对 p0i/p1i 的解析梯度，本实现直接用 torch 自动微分，
    不手写梯度。
- 参考的开源实现：
  * MONAI（Project-MONAI/MONAI，monai/losses/tversky.py，dev 分支；由 NiftyNet
    版本适配而来）：score = 1 - (tp + smooth_nr) / (tp + alpha*fp + beta*fn +
    smooth_dr)，α=FP 权重、β=FN 权重，与论文式(2)(3)一致；
  * 本仓库素材 /ssd/lx/metrics/metrics.py:47-84 的 hard Dice 指标
    （(2tp+smooth)/(2tp+fp+fn+1e-8+smooth)，smooth=1），参考其平滑写法。
- 与开源实现不一致处（以论文为准）：
  * MONAI 在分子另加 smooth_nr=1e-5；本实现只在分母加 eps——论文式(3)没有
    平滑项，分子加平滑会系统性抬高指数、对小目标语义改变更大，故仅在分母加
    eps 防 0/0（语义上仍是无平滑的式(3)）；
  * MONAI 默认把背景通道一并算指数；论文只对病灶（前景）定义指数，本实现默认
    include_background=False，只在前景通道上计算；
  * 本地 metrics.py 的 smooth=1 属 hard 指标的做法，对软损失过大（小目标时
    会显著拉高指数），未采用；
  * 聚合口径：论文式(3)把 Σi 写成对体素的整体求和；训练实现通行做法是按
    (样本, 通道) 各算一个指数再 reduction（MONAI DiceLoss/TverskyLoss 默认
    batch=False 即逐样本），本实现从之，并在自测第 1/2 项按同一口径对拍
    （mean(比值) ≠ 比值(总和)，整批池化对拍是不成立的）。
- 数值稳定：分母加 eps=1e-7。依据：分母是"体素软计数"（TP + α·FP + β·FN），
  单个体素计数量级为 1，1e-7 远小于任何非空目标的计数和（fp32 下不改变损失
  值），同时保证全零输入（TP=FP=FN=0）时分母非零，避免 0/0；概率来自
  sigmoid/softmax 天然落在 (0,1)，无显式 log/exp/幂运算，无需额外钳制。

布局约定：2D=(B,C,H,W)，3D=(B,C,D,H,W)；numpy 指标用预处理后的 numpy 数组，
布局见函数 docstring。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["tversky_loss", "TverskyLoss", "tversky_index_numpy"]

# 分母平滑系数：见文件头"数值稳定"注释
_DEFAULT_EPS = 1e-7


def _to_prob(pred: torch.Tensor, from_logits: bool) -> torch.Tensor:
    """把网络原始输出转成每通道概率。

    参数：
        pred: torch.Tensor，2D=(B,C,H,W) 或 3D=(B,C,D,H,W)。
        from_logits: True 表示 pred 是未过激活的 logit——C==1 时过 sigmoid，
            C>1 时过 softmax（通道维）；False 表示 pred 已是概率。
    返回：
        与 pred 同形状的概率张量，每通道值域 (0,1)。
    """
    if not from_logits:
        return pred
    if pred.shape[1] == 1:
        return torch.sigmoid(pred)
    return F.softmax(pred, dim=1)


def _expand_target(target: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
    """把真值对齐到概率张量形状 (B,C,...)。

    参数：
        target: 两种形式之一——
            (a) 与 prob 同形的 one-hot/probability 标签 (B,C,S...)；
            (b) 单通道二值标签 (B,1,S...)（按通道广播：每个前景通道对同一
                二值标签做二分类）。
        prob: 已转成概率的预测张量 (B,C,...)。
    返回：
        与 prob 同形状的浮点标签张量。
    """
    if target.shape == prob.shape:
        return target.to(dtype=prob.dtype)
    if target.dim() == prob.dim() and target.shape[1] == 1:
        return target.to(dtype=prob.dtype).expand_as(prob)
    raise ValueError(
        f"target 形状 {tuple(target.shape)} 既不是 one-hot {tuple(prob.shape)}，"
        f"也不是单通道二值标签 {tuple((prob.shape[0], 1, *prob.shape[2:]))}"
    )


def tversky_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
    from_logits: bool = False,
    eps: float = _DEFAULT_EPS,
    reduction: str = "mean",
    include_background: bool = False,
) -> torch.Tensor:
    """Tversky 损失：1 - Tversky 指数（论文式(3)，最小化其补）。

    α 是假阳 FP 的权重、β 是假阴 FN 的权重；α=β=0.5 时退化为 soft Dice loss
    （论文式(1)对应的训练目标）。漏检代价高的小病灶分割建议 β>α（论文最优
    (0.3, 0.7)）。

    参数：
        pred: 2D=(B,C,H,W) 或 3D=(B,C,D,H,W)。C==1 为前景 logit/概率；
            C>1 为多类 logit/概率（每个前景通道独立按二分类算指数，再平均）。
        target: 与 pred 空间形状匹配的真值——同形 one-hot (B,C,...) 或单通道
            二值 (B,1,...)，元素为 0/1。
        alpha: FP 惩罚权重，默认 0.5。
        beta: FN 惩罚权重，默认 0.5（即 Dice）。
        from_logits: pred 是否为 logit（C==1 走 sigmoid，C>1 走 softmax）。
        eps: 分母平滑系数，防全零输入 0/0，默认 1e-7（取值依据见文件头）。
        reduction: "mean"（标量，默认）/ "sum" / "none"（返回 (B,C_fg)）。
        include_background: 是否把通道 0（背景）也作为独立前景参与计算，
            默认 False（论文只对病灶定义指数）。

    返回：
        torch.Tensor，reduction 决定标量或 (B,C_fg)。

    异常：
        ValueError：target 形状与 pred 不匹配，或 reduction 非法。
    """
    prob = _to_prob(pred, from_logits)
    gt = _expand_target(target, prob)
    if not include_background and prob.shape[1] > 1:
        prob = prob[:, 1:]
        gt = gt[:, 1:]

    # 空间维聚合出每 (B, 通道) 的软计数，对应论文式(3)的四个求和项
    axes = list(range(2, prob.dim()))
    tp = (prob * gt).sum(axes)             # Σ p0·g0 = |P∩G|
    fp = (prob * (1 - gt)).sum(axes)       # Σ p0·g1 = |P\G|（假阳软计数）
    fn = ((1 - prob) * gt).sum(axes)       # Σ p1·g0 = |G\P|（假阴软计数）

    # 论文式(3)：仅在分母加 eps 防 0/0（取值依据见文件头）
    tversky_index = tp / (tp + alpha * fp + beta * fn + eps)
    loss = 1.0 - tversky_index

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"reduction 只支持 mean/sum/none，收到 {reduction!r}")


class TverskyLoss(nn.Module):
    """Tversky 损失的 nn.Module 封装，参数同 tversky_loss。

    参数：
        alpha: FP 惩罚权重（论文式(2)），默认 0.5。
        beta: FN 惩罚权重，默认 0.5。
        from_logits: forward 收到的 pred 是否为 logit，默认 False。
        eps: 分母平滑系数，默认 1e-7。
        reduction: "mean" / "sum" / "none"，默认 "mean"。
        include_background: 是否计入背景通道，默认 False。
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        from_logits: bool = False,
        eps: float = _DEFAULT_EPS,
        reduction: str = "mean",
        include_background: bool = False,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.from_logits = from_logits
        self.eps = eps
        self.reduction = reduction
        self.include_background = include_background

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算 Tversky 损失，参数语义同 tversky_loss。"""
        return tversky_loss(
            pred,
            target,
            alpha=self.alpha,
            beta=self.beta,
            from_logits=self.from_logits,
            eps=self.eps,
            reduction=self.reduction,
            include_background=self.include_background,
        )


def tversky_index_numpy(
    prob_fg: np.ndarray,
    gt: np.ndarray,
    alpha: float = 0.3,
    beta: float = 0.7,
    eps: float = _DEFAULT_EPS,
) -> float:
    """numpy 版 Tversky 指数（论文式(2)的软计数口径），用于验证/评估。

    参数：
        prob_fg: numpy.ndarray，前景概率图，值域 [0,1]。布局 2D=(H,W) 或
            3D=(D,H,W)（即去掉批次与通道维，或单通道 (1,H,W)/(1,D,H,W) 自动
            去掉通道维）。
        gt: numpy.ndarray，与 prob_fg 空间形状相同的二值真值（0/1）。
        alpha: FP 惩罚权重，默认 0.3（论文最优组合）。
        beta: FN 惩罚权重，默认 0.7（论文最优组合）。
        eps: 分母平滑系数，默认 1e-7（同文件头说明）。

    返回：
        float，Tversky 指数 ∈ [0, 1]。

    异常：
        ValueError：空间形状不匹配或数值不在 [0,1]。
    """
    prob_fg = np.asarray(prob_fg, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if prob_fg.ndim == gt.ndim + 1 and prob_fg.shape[0] == 1:
        prob_fg = prob_fg[0]  # 去掉单通道维，对齐 (H,W) / (D,H,W)
    if prob_fg.shape != gt.shape:
        raise ValueError(f"形状不匹配：prob_fg {prob_fg.shape} vs gt {gt.shape}")
    if prob_fg.min() < 0.0 or prob_fg.max() > 1.0:
        raise ValueError("prob_fg 应为 [0,1] 内的概率")
    gt = (gt > 0.5).astype(np.float64)  # 容忍 0/1 之外的整型标签输入

    tp = float((prob_fg * gt).sum())            # |P∩G|（软计数）
    fp = float((prob_fg * (1.0 - gt)).sum())    # |P\G|：假阳软计数
    fn = float(((1.0 - prob_fg) * gt).sum())    # |G\P|：假阴软计数
    return tp / (tp + alpha * fp + beta * fn + eps)


# ---------------------------------------------------------------------------
# 自测：CPU 可跑、固定随机种子、数秒内完成
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    # 1) 退化一致性：α=β=0.5 时应等于 soft Dice loss 1 - 2TP/(ΣP+ΣG)（论文
    #    式(1)/(2) 的退化关系）。按实现口径（每样本每通道一个指数，见文件头）
    #    用 reduction="none" 逐条与 numpy 闭式对拍
    prob = torch.rand(2, 1, 8, 8)
    gt = (torch.rand(2, 1, 8, 8) > 0.7).float()
    loss_none = tversky_loss(prob, gt, alpha=0.5, beta=0.5, reduction="none")
    p = prob.numpy().astype(np.float64)
    g = gt.numpy().astype(np.float64)
    tp = (p * g).sum(axis=(2, 3))
    dice_closed = 1.0 - 2.0 * tp / (p.sum(axis=(2, 3)) + g.sum(axis=(2, 3)))
    assert torch.isfinite(loss_none).all()
    assert np.allclose(loss_none.numpy(), dice_closed, atol=1e-6), (loss_none, dice_closed)
    print(f"PASS 退化一致性 α=β=0.5 ≡ soft Dice loss（逐样本对拍，max|Δ|={np.abs(loss_none.numpy()-dice_closed).max():.2e}）")

    # 2) 公式对拍：(α,β)=(0.3,0.7) 时与论文式(3)的 numpy 闭式逐样本一致
    loss37 = tversky_loss(prob, gt, alpha=0.3, beta=0.7, reduction="none")
    fp = (p * (1.0 - g)).sum(axis=(2, 3))
    fn = ((1.0 - p) * g).sum(axis=(2, 3))
    closed_37 = 1.0 - tp / (tp + 0.3 * fp + 0.7 * fn)
    assert np.allclose(loss37.numpy(), closed_37, atol=1e-6)
    print(f"PASS 式(3) 对拍 (α,β)=(0.3,0.7)：逐样本一致，max|Δ|={np.abs(loss37.numpy()-closed_37).max():.2e}")

    # 3) α/β 语义方向：固定一个漏检一半病灶的欠召回预测，β 越大损失必须越大
    #    （β 惩罚 FN——这是最容易弄反的点，直接验证）
    recall_half = gt.clone()
    recall_half[:, :, 4:] = 0.0  # 预测丢掉下半空间 → FN 加倍
    l_small_b = tversky_loss(recall_half, gt, alpha=0.7, beta=0.3)
    l_big_b = tversky_loss(recall_half, gt, alpha=0.3, beta=0.7)
    assert l_big_b > l_small_b, (l_small_b.item(), l_big_b.item())
    print(f"PASS β 惩罚 FN：欠召回预测下 loss(β=0.7)={l_big_b.item():.4f} > loss(β=0.3)={l_small_b.item():.4f}")

    # 4) 边界值：完美预测→损失≈0；全反预测→损失≈1；输出均有限
    perfect = tversky_loss(gt, gt)
    assert torch.isfinite(perfect) and perfect.item() < 1e-6, perfect.item()
    wrong_prob = torch.where(gt > 0.5, torch.zeros_like(gt), torch.ones_like(gt))
    all_wrong = tversky_loss(wrong_prob, gt)
    assert torch.isfinite(all_wrong) and abs(all_wrong.item() - 1.0) < 1e-3, all_wrong.item()
    print(f"PASS 边界值：完美 {perfect.item():.2e}≈0，全错 {all_wrong.item():.4f}≈1，均有限")

    # 5) 多类 softmax + logit 入口 + backward：梯度必须有限（无 NaN/Inf）
    logits = torch.randn(2, 3, 16, 16, requires_grad=True)  # 2D 多类 (B,C,H,W)
    tgt2d = torch.randint(0, 3, (2, 1, 16, 16))             # 单通道类别索引
    onehot = F.one_hot(tgt2d[:, 0].long(), 3).permute(0, 3, 1, 2).float()
    loss5 = TverskyLoss(alpha=0.3, beta=0.7, from_logits=True)(logits, onehot)
    loss5.backward()
    assert torch.isfinite(loss5), loss5
    assert torch.isfinite(logits.grad).all(), "logits.grad 含 NaN/Inf"
    print(f"PASS 2D 多类 forward+backward：loss={loss5.item():.4f}，梯度有限")

    # 6) 完整小模型 2D：(B,C,H,W)=(2,2,32,32)，forward+backward，打印参数量
    net2d = nn.Sequential(
        nn.Conv2d(2, 8, 3, padding=1), nn.ReLU(),
        nn.Conv2d(8, 4, 3, padding=1), nn.ReLU(),
        nn.Conv2d(4, 1, 1),
    )
    x2d = torch.randn(2, 2, 32, 32)
    y2d = (torch.rand(2, 1, 32, 32) > 0.9).float()  # 极不平衡：前景约 10%
    loss6 = TverskyLoss(alpha=0.3, beta=0.7, from_logits=True)(net2d(x2d), y2d)
    loss6.backward()
    assert torch.isfinite(loss6)
    print(f"PASS 2D 小模型端到端：参数量 {sum(q.numel() for q in net2d.parameters())}，loss={loss6.item():.4f}")

    # 7) 完整小模型 3D：(B,C,D,H,W)=(1,2,8,16,16)，forward+backward，打印参数量
    net3d = nn.Sequential(
        nn.Conv3d(2, 4, 3, padding=1), nn.ReLU(),
        nn.Conv3d(4, 1, 1),
    )
    x3d = torch.randn(1, 2, 8, 16, 16)
    y3d = (torch.rand(1, 1, 8, 16, 16) > 0.9).float()
    loss7 = TverskyLoss(alpha=0.3, beta=0.7, from_logits=True)(net3d(x3d), y3d)
    loss7.backward()
    assert torch.isfinite(loss7)
    print(f"PASS 3D 小模型端到端：参数量 {sum(q.numel() for q in net3d.parameters())}，loss={loss7.item():.4f}")

    # 8) numpy 指标：典型用例形状/数值范围 + 完美/全空极端用例
    gt_np = (np.random.rand(16, 16, 16) > 0.9).astype(np.float64)  # 3D (D,H,W)
    prob_np = np.clip(gt_np + np.random.normal(0, 0.2, gt_np.shape), 0.0, 1.0)
    ti = tversky_index_numpy(prob_np, gt_np)
    assert np.isscalar(ti) or isinstance(ti, float)
    assert 0.0 <= ti <= 1.0, ti
    assert abs(tversky_index_numpy(gt_np, gt_np) - 1.0) < 1e-9        # 完美预测
    assert tversky_index_numpy(np.zeros_like(gt_np), gt_np) == 0.0    # 全空预测：0/eps
    print(f"PASS numpy 指标：随机 3D 用例 TI={ti:.4f}∈[0,1]，完美≈1，全空=0")

    print("PASS 全部自测通过")
