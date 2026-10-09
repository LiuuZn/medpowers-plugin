"""dice-loss —— [loss] 可导 Dice 损失（V-Net 首提）。

统一四问（论文支撑）：
- 哪篇论文用了它：Milletari、Navab、Ahmadi，《V-Net: Fully Convolutional Neural
  Networks for Volumetric Medical Image Segmentation》（V-Net），2016，3DV。
  https://arxiv.org/abs/1606.04797
- 目的：3D 医疗影像中目标解剖结构常只占极小区域，交叉熵（多项逻辑损失）易陷入
  偏向背景的局部极小、前景漏检；论文提出直接以 Dice 系数为优化目标，无需对
  前景/背景体素做样本重加权。
- 效果：PROMISE 2012 前列腺 MRI（训练 50 卷 / 测试 30 卷）：V-Net + Dice loss
  平均 Dice 0.869±0.033、Hausdorff 5.71±1.20 mm、挑战赛得分 82.39；同一网络换成
  带样本重加权的多项逻辑损失仅 0.739±0.088、10.55±5.38 mm、63.30。
- 结论：直接优化 Dice 的可导损失在前景/背景强烈不平衡的二值分割上明显优于加权
  交叉熵；论文只处理二值前景/背景，多类"逐类算 Dice 取平均"是后续扩展，非原文。

实现说明：
- 关键公式与论文对应（论文第 3 节 "Dice loss layer"，arXiv PDF 中两式无编号）：
  1) 可导 Dice 系数：D = 2·Σ_i p_i·g_i / (Σ_i p_i² + Σ_i g_i²)，i 遍历 N 个体素；
     训练中最大化 D（等价于最小化损失 L = 1 − D）。
     → DiceLoss 默认 squared_pred=True 复现该式。
  2) 梯度闭式：∂D/∂p_j = 2·[g_j·(Σp_i²+Σg_i²) − 2·p_j·Σp_i·g_i] / (Σp_i²+Σg_i²)²。
     → 文末自测 PASS 3 用 autograd 与该闭式对拍（float64，max_err<1e-10）。
- 参考的开源实现（本地）：
  * nnunetv2 MemoryEfficientSoftDiceLoss：
    /ssd/lx/totalsegmentator/nnunetv2/training/loss/dice.py:58
    （非平方软形式、smooth 加在分子分母、分母 clamp 1e-8、do_bg/batch_dice 开关、
    gt 的 one-hot 在 no_grad 中构建省显存——本实现同样在计算交/并前固定 gt）；
  * nnunetv2 DC_and_CE_loss：
    /ssd/lx/totalsegmentator/nnunetv2/training/loss/compound_losses.py（CE+Dice
    各权重 1 的组合；本文件只实现 Dice 侧，不含组合损失）；
  * /ssd/lx/metrics/metrics.py:47 dice_test（评价端硬 Dice，smooth=1）。
- 与开源实现不一致处（以论文为准）：
  * 默认 squared_pred=True 采用论文形式（分母 Σp²）；nnU-Net 主流实现用非平方
    Σp（软预测下两者不同，硬 0/1 预测下相等）。保留 squared_pred=False 复现
    nnU-Net 行为便于对照，默认以论文为准。
  * 论文原式无平滑项；本实现默认 smooth=1e-5 并对分母 clamp(1e-8)（与 nnU-Net
    一致）：论文未处理预测/真值皆空的零除情形，工程上必须加，设 0 可精确复现
    论文（需保证分母非零）。
  * 论文只做二值（softmax 两通道）；本实现扩展支持多类逐类平均与 do_bg 排除
    背景类（nnU-Net 做法，非原文），docstring 中已标明。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["DiceLoss", "dice_loss", "soft_dice_coefficient", "hard_dice_score"]


def _to_probs(pred: torch.Tensor, from_logits: bool) -> torch.Tensor:
    """把网络输出转成概率（内部工具）。

    pred: (B, C, H, W) 或 (B, C, D, H, W)。C==1 视为二值分割走 sigmoid，
    C>1 视为多类别在 channel 维 softmax。from_logits=True 时 pred 是未过
    非线性的 logits，否则已是概率。
    """
    if not from_logits:
        return pred
    if pred.shape[1] == 1:
        return torch.sigmoid(pred)
    return torch.softmax(pred, dim=1)


def _align_target(target: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """把真值对齐成与 probs 同布局的 (B, C, ...) 浮点张量（内部工具）。

    接受两种 target：
    - 类索引图 (B, *spatial)：long/int，值域 [0, C)；自动 one-hot 扩展；
    - 已对齐的 one-hot / 二值 mask (B, C, *spatial)：channel 数须与 probs 相同。
    （约定同 nnU-Net dice.py:141 "shape 相同则视为已 one-hot"。）
    """
    if target.ndim == probs.ndim:
        if target.shape[1] != probs.shape[1]:
            raise ValueError(
                f"target channel 数 {target.shape[1]} 与 pred 的 {probs.shape[1]} 不一致；"
                "类索引 target 应去掉 channel 维传入 (B, *spatial)"
            )
        return target.to(dtype=probs.dtype)
    gt = F.one_hot(target.long(), probs.shape[1])  # (B, *spatial, C)
    return gt.movedim(-1, 1).to(dtype=probs.dtype)  # (B, C, *spatial)


def soft_dice_coefficient(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    smooth: float = 1e-5,
    eps: float = 1e-8,
    squared_pred: bool = True,
    from_logits: bool = True,
    batch_dice: bool = False,
) -> torch.Tensor:
    """逐类软 Dice 系数（可导）。

    布局：pred 为 (B, C, H, W) 或 (B, C, D, H, W)；target 为类索引 (B, *spatial)
    或已对齐 one-hot/mask (B, C, *spatial)。

    参数：
    - smooth：加在分子与分母上的平滑项。默认 1e-5（nnU-Net 训练默认，
      nnUNetTrainer.py:400）；设 0 可精确复现论文原式（需保证分母非零）。
    - eps：分母下限 clamp，防零除。默认 1e-8（同 nnU-Net dice.py:116 的
      torch.clip(..., 1e-8)）：float32 最小正规数约 1.2e-38，1e-8 足以约束
      "预测与真值皆空"时的梯度爆炸，又远小于正常分母不影响数值。
    - squared_pred：True 用论文形式（分母 Σp²，D = 2Σpg/(Σp²+Σg²)）；False 用
      nnU-Net 软形式（分母 Σp）。g 为 0/1 标签故 Σg²=Σg，两种形式 g 侧相同。
    - from_logits：pred 是否为 logits（True 时按 C==1 sigmoid / C>1 softmax）。
    - batch_dice：False 逐样本计算，返回 (B, C)；True 跨 batch 池化，返回 (C,)。

    返回：每类 Dice ∈ [0, 1]，形状 (B, C) 或 (C,)。
    """
    probs = _to_probs(pred, from_logits)
    gt = _align_target(target, probs)
    axes = tuple(range(2, probs.ndim))
    if batch_dice:
        axes = (0,) + axes

    inter = (probs * gt).sum(dim=axes)
    if squared_pred:  # 论文式：分母含 Σp²（g 硬标签时 Σg²=Σg）
        denom_p = (probs * probs).sum(dim=axes)
        denom_g = (gt * gt).sum(dim=axes)
    else:  # nnU-Net 软形式：分母用 Σp、Σg（dice.py:100-101,116）
        denom_p = probs.sum(dim=axes)
        denom_g = gt.sum(dim=axes)

    numerator = 2.0 * inter + smooth
    denominator = torch.clamp(denom_p + denom_g + smooth, min=eps)
    return numerator / denominator


class DiceLoss(nn.Module):
    """可导 Dice 损失：L = 1 − mean(D)（论文：最大化 D，等价最小化 1−D）。

    多类时对各类（可经 do_bg 排除背景类）取平均——逐类平均是 nnU-Net 式扩展，
    论文原文仅二值前景/背景单一 Dice。
    """

    def __init__(
        self,
        *,
        smooth: float = 1e-5,
        eps: float = 1e-8,
        squared_pred: bool = True,
        from_logits: bool = True,
        do_bg: bool = False,
        batch_dice: bool = False,
    ) -> None:
        # 参数说明见 forward / 模块 docstring
        super().__init__()
        self.smooth = smooth
        self.eps = eps
        self.squared_pred = squared_pred
        self.from_logits = from_logits
        self.do_bg = do_bg
        self.batch_dice = batch_dice

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算 Dice 损失（标量）。

        pred: (B, C, H, W) 或 (B, C, D, H, W)，logits（from_logits=True）或概率。
        target: 类索引 (B, *spatial) 或 one-hot/mask (B, C, *spatial)。

        参数含义（构造时给定）：
        - smooth / eps / squared_pred / batch_dice：同 soft_dice_coefficient；
        - do_bg：多类时是否剔除第 0 类（背景）再平均；False 为 nnU-Net 训练默认
          （nnUNetTrainer.py:400 do_bg=False）。C==1（二值单通道）时自动忽略。
        """
        dc = soft_dice_coefficient(
            pred,
            target,
            smooth=self.smooth,
            eps=self.eps,
            squared_pred=self.squared_pred,
            from_logits=self.from_logits,
            batch_dice=self.batch_dice,
        )
        if not self.do_bg and dc.shape[-1] > 1:  # C==1 时无背景类可剔
            dc = dc[..., 1:]
        return 1.0 - dc.mean()


def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    smooth: float = 1e-5,
    eps: float = 1e-8,
    squared_pred: bool = True,
    from_logits: bool = True,
    do_bg: bool = False,
    batch_dice: bool = False,
) -> torch.Tensor:
    """DiceLoss 的函数式封装（参数与布局同 DiceLoss.forward 及其构造参数）。"""
    return DiceLoss(
        smooth=smooth,
        eps=eps,
        squared_pred=squared_pred,
        from_logits=from_logits,
        do_bg=do_bg,
        batch_dice=batch_dice,
    )(pred, target)


def hard_dice_score(
    pred_mask: np.ndarray, gt_mask: np.ndarray, smooth: float = 1.0, eps: float = 1e-8
) -> float:
    """硬（二值化后）Dice 指标——与 Dice loss 配对观察的评价量（numpy 实现）。

    布局：pred_mask 与 gt_mask 同形状的任意维二值数组（非零即前景），如
    (H, W)、(D, H, W) 或 (B, D, H, W)；对全部元素算单一 Dice，批量样本请逐个
    传入。公式同 /ssd/lx/metrics/metrics.py:67：(2·TP + smooth) /
    (2·TP + FP + FN + smooth + eps)。

    参数：
    - smooth：评价端默认 1.0（metrics.py:54 的写法；注意与训练端 1e-5 不同）。
    - eps：分母防零除下限（同上文件 1e-8，防止两边皆空时除零）；预测与真值皆空
      时返回 ≈1.0（eps 影响，实际为 0.99999999；smooth=1 约定下空对空视为一致，
      与 metrics.py 行为相同）。

    返回：float ∈ [0, 1]。
    """
    if pred_mask.shape != gt_mask.shape:
        raise ValueError(f"形状不一致：pred {pred_mask.shape} vs gt {gt_mask.shape}")
    p = pred_mask.astype(bool)
    g = gt_mask.astype(bool)
    tp = float(np.count_nonzero(p & g))
    fp = float(np.count_nonzero(p & ~g))
    fn = float(np.count_nonzero(~p & g))
    return (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth + eps)


# ---------------------------------------------------------------------------
# 自测：CPU 可跑、数秒内完成。python dice-loss.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    # 1. 前向：2D/3D 形状与值域（squared 形式下 2Σpg ≤ Σp²+Σg²，故 D∈[0,1]、loss∈[0,1]）
    pred3d = torch.randn(2, 3, 16, 16, 16)
    y3d = torch.randint(0, 3, (2, 16, 16, 16))
    loss3d = dice_loss(pred3d, y3d)
    assert loss3d.ndim == 0 and torch.isfinite(loss3d)
    assert 0.0 <= float(loss3d) <= 1.0, f"loss 越界: {float(loss3d)}"
    pred2d = torch.randn(2, 3, 32, 32)
    y2d = torch.randint(0, 3, (2, 32, 32))
    loss2d = dice_loss(pred2d, y2d)
    assert 0.0 <= float(loss2d) <= 1.0
    print(f"PASS 1 前向形状/值域 3D={float(loss3d):.4f} 2D={float(loss2d):.4f}")

    # 2. 理想预测 → 0，完全错开 → ≈1（概率直接喂入，from_logits=False）
    y_one = F.one_hot(y3d, 3).movedim(-1, 1).float()  # (2,3,16,16,16)
    loss_perfect = dice_loss(y_one, y_one, from_logits=False)
    assert float(loss_perfect) < 1e-6, f"理想预测 loss 应≈0: {float(loss_perfect)}"
    wrong = torch.zeros_like(y_one)
    wrong.scatter_(1, ((y3d + 1) % 3).unsqueeze(1), 1.0)  # 概率全押在错误的类上
    loss_worst = dice_loss(wrong, y_one, from_logits=False)
    assert float(loss_worst) > 0.99, f"完全错开 loss 应≈1: {float(loss_worst)}"
    print(
        f"PASS 2 极值行为 perfect={float(loss_perfect):.2e} worst={float(loss_worst):.4f}"
    )

    # 3. 论文梯度闭式对拍：autograd vs ∂D/∂p_j = 2[g_j·Q − 2p_j·A]/Q²，
    #    Q=Σp²+Σg²，A=Σpg（论文第 3 节第二式；float64 保证精度）
    p_flat = torch.rand(4, 7, dtype=torch.float64, requires_grad=True)  # 展平的体素
    g_flat = (torch.rand(4, 7, dtype=torch.float64) > 0.5).double()
    d = soft_dice_coefficient(
        p_flat,
        g_flat,
        smooth=0.0,
        eps=0.0,
        squared_pred=True,
        from_logits=False,
        batch_dice=True,
    ).sum()
    (grad_auto,) = torch.autograd.grad(d, p_flat)
    # 论文闭式逐类套用（每类是与该类 one-hot 之间的二值 Dice，代数同论文；多类逐类为扩展约定）：
    # ∂D_c/∂p_{b,c} = 2·[g_{b,c}·Q_c − 2·p_{b,c}·A_c] / Q_c²，Q_c=Σp²+Σg²、A_c=Σpg（类内求和）
    p_d = p_flat.detach()
    a_c = (p_d * g_flat).sum(dim=0)  # (C,)
    q_c = (p_d**2).sum(dim=0) + (g_flat**2).sum(dim=0)  # (C,)
    grad_paper = 2.0 * (g_flat * q_c - 2.0 * p_d * a_c) / q_c**2
    max_err = float((grad_auto - grad_paper).abs().max())
    assert max_err < 1e-10, f"梯度与论文闭式不符: {max_err}"
    print(f"PASS 3 论文梯度闭式对拍 max_err={max_err:.2e}")

    # 4. squared 与非 squared 在硬 0/1 预测下相等（Σp²=Σp），且等于硬 Dice 公式
    p_hard = (torch.rand(1, 1, 8, 8, 8) > 0.5).float()
    g_hard = (torch.rand(1, 1, 8, 8, 8) > 0.5).float()
    l_sq = dice_loss(p_hard, g_hard, smooth=0.0, eps=1e-8, squared_pred=True, from_logits=False, do_bg=True)
    l_nosq = dice_loss(p_hard, g_hard, smooth=0.0, eps=1e-8, squared_pred=False, from_logits=False, do_bg=True)
    tp = float((p_hard * g_hard).sum())
    hard_loss = 1.0 - (2.0 * tp) / (float(p_hard.sum()) + float(g_hard.sum()))
    assert abs(float(l_sq) - float(l_nosq)) < 1e-7
    assert abs(float(l_sq) - hard_loss) < 1e-7
    print(f"PASS 4 平方/非平方硬预测等值 loss={float(l_sq):.4f}")

    # 5. 类索引 target 与 one-hot target 等价
    l_idx = dice_loss(pred3d, y3d)
    l_oh = dice_loss(pred3d, y_one, from_logits=True)
    assert abs(float(l_idx) - float(l_oh)) < 1e-6
    print("PASS 5 类索引与 one-hot target 等价")

    # 6. 类别不平衡（前景≈0.5%）：损失/梯度有限；全预测背景时 loss≈1 且梯度不 NaN
    vol = torch.zeros(1, 2, 32, 32, 32)
    vol[0, 0] = 1.0  # 全背景
    vol[0, 0, :4, :5, :2] = 0.0  # 抠掉 40 个体素给前景
    vol[0, 1, :4, :5, :2] = 1.0  # 前景 40/32768≈0.12%
    vol = vol * 12.0 - 6.0  # 放大成 logits
    vol.requires_grad_(True)
    lb = dice_loss(vol, torch.zeros(1, 32, 32, 32).long())  # gt 全背景，排除 bg 后考验 fg 类
    lb.backward()
    assert torch.isfinite(lb) and torch.isfinite(vol.grad).all()
    lb_val = float(lb.detach())
    p_zero = torch.full((1, 1, 16, 16, 16), -20.0)  # 二值：sigmoid→≈0，全预测背景
    gt_full = torch.zeros(1, 16, 16, 16).long()  # C==1 类索引全 0 → one_hot(0,1)=前景全 1
    lz = dice_loss(p_zero, gt_full, from_logits=True, do_bg=True)
    assert torch.isfinite(lz) and float(lz) > 0.99
    print(f"PASS 6 不平衡场景 loss={lb_val:.4f}，全背景预测 loss={float(lz):.4f} 有限")

    # 7. 小分割模型端到端 forward+backward，打印参数量，断言梯度有限
    class _TinySeg3D(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv3d(1, 4, 3, padding=1), nn.ReLU(inplace=True),
                nn.Conv3d(4, 3, 3, padding=1),
            )

        def forward(self, x):
            return self.net(x)

    model = _TinySeg3D()
    n_params = sum(p.numel() for p in model.parameters())
    out = model(torch.randn(1, 1, 16, 16, 16))
    loss_m = dice_loss(out, y3d[:1, :16, :16, :16] % 3)
    loss_m.backward()
    grads_finite = all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert grads_finite
    print(f"PASS 7 小模型端到端 参数量={n_params} loss={float(loss_m.detach()):.4f} 梯度有限")

    # 8. numpy 硬 Dice 指标典型用例：形状/值域/手算对拍
    gt_np = np.zeros((16, 16), dtype=np.uint8)
    gt_np[4:8, 4:8] = 1  # 16 个体素
    pr_np = np.zeros((16, 16), dtype=np.uint8)
    pr_np[5:9, 4:8] = 1  # 与 gt 重叠 12，Σp=Σg=16 → (24+1)/(32+1+1e-8)
    expected = 25.0 / 33.0
    dc = hard_dice_score(pr_np, gt_np)
    assert 0.0 <= dc <= 1.0
    assert abs(dc - expected) < 1e-6, f"硬 Dice 手算不符: {dc} vs {expected}"
    empty_both = hard_dice_score(np.zeros((8, 8)), np.zeros((8, 8)))
    assert 0.0 <= empty_both <= 1.0 and np.isfinite(empty_both)
    print(f"PASS 8 硬 Dice 指标 dc={dc:.4f}（手算 {expected:.4f}），双空有限={empty_both:.4f}")

    print("PASS 全部自测通过")
