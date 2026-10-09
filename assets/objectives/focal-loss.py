"""focal-loss — [loss]（可导损失；资产卡：assets/objectives/focal-loss.md）

论文支撑（统一四问，简体中文）：
- 哪篇论文用了它：Lin, Goyal, Girshick, He & Dollár（FAIR），《Focal Loss for Dense
  Object Detection》（使用该损失的模型叫 RetinaNet），ICCV 2017，
  https://arxiv.org/abs/1708.02002 （本实现对照 arXiv v2 全文即 ICCV 版；
  公式编号依该版：式 (4) 为 focal loss，式 (5) 为 α 平衡版）。
- 目的：单阶段密集检测训练时每张图枚举约 10 万个候选位置，绝大多数是易分类背景，
  其损失总量淹没少数难例；focal loss 用 (1-p_t)^γ 因子压低易样本贡献，聚焦难样本。
- 效果：同一 ResNet-50-FPN 网络，α-balanced CE 31.1 AP → focal loss 34.0 AP
  （+2.9 AP，表 1b 最优 γ=2、α=0.25）；对比最优 OHEM（32.8 AP）focal loss
  36.0 AP（ResNet-101-FPN，表 1d，领先 3.2 AP）；最终模型 COCO test-dev 40.8 AP。
- 结论：γ=0 退化为 CE；γ=2、α=0.25 最优且对 γ∈[0.5,5] 稳健，α 随 γ 增大应调小。

实现说明（关键公式 ↔ 论文式子，编号依 arXiv v2）：
- 式 (2)：p_t = p（y=1）或 1-p（y=0），p=σ(x)；式 (2) 后正文句：CE(p_t) = -log(p_t)。
- 式 (4)：FL(p_t) = -(1-p_t)^γ log(p_t)。
- 式 (5)：FL(p_t) = -α_t (1-p_t)^γ log(p_t)，论文实验采用形式；α_t 定义同式 (3)
  （α 当 y=1，1-α 当 y=0）。
- softmax_focal_loss 是把式 (4) 中 p_t 换成 softmax 概率的常见推广，非论文原式
  （论文为 sigmoid one-vs-all），函数 docstring 已注明。
- 梯度：p_t 不做 detach，调制因子 (1-p_t)^γ 对 logits 的导数保留在反传中——
  这与论文附录式 (9)–(11) 给出的对 x 的真导数一致；自测中的有限差分验证的正是该真导数。
- 参考实现：torchvision.ops.sigmoid_focal_loss（torchvision 源码注明源自
  facebookresearch/fvcore 的 fvcore/nn/focal_loss.py；本地落点
  /ssd/lx/totalsegmentator/.venv/lib/python3.11/site-packages/torchvision/ops/focal_loss.py）。
- 与 torchvision 不一致处（均以论文为准或为数值稳定）：
  1) -log(p_t) 一项用 binary_cross_entropy_with_logits（log-sum-exp，内部稳定）计算，
     不显式 log(sigmoid(x))，避免 log(0)。torchvision 同样如此，属共同选择。
  2) 调制因子 (1-p_t)^γ 先对 1-p_t 做下界 0 钳制。依据（实测，float32）：
     合规标签 t∈[0,1] 时 p_t 是 p 与 1-p 的凸组合、必在 [0,1] 内，x∈[-300,300] 稠密扫描
     （60 万点，硬标签与 (0,1) 软标签网格）p_t>1 出现 0 次——钳制对合规输入是无操作；
     它防的是越界/异常标签（如 t=2 时 p_t=3p-1，1-p_t 最低到 -1：γ 为非整数时负底数幂
     产生 NaN，γ 为整数时 torch 静默返回错误权重），钳到 0 使行为良定义（该处 FL=0）。
     torchvision 未钳制，此为防御性增强。
  3) alpha=None 表示不用式 (5) 的 α 加权（即式 (4)）；torchvision 用 alpha=-1 表示
     同一含义，本实现改用 None，语义更直白。
- 数值稳定：log 走 log-sum-exp（BCE with logits / log_softmax）；幂运算先钳 1-p_t ≥ 0；
  softmax 版 p_t 由 logp.exp() 得到，无除法、无 log(0)。

自测（CPU，数秒，固定种子）：见文件末 __main__——与朴素双精度参考实现逐元素对拍、
γ=0 退化为 CE 的检查、与 torchvision 的对拍（可用时）、有限差分梯度验证、
小卷积模型 forward+backward 并打印参数量；逐项 print("PASS ...")。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["focal_loss", "softmax_focal_loss", "FocalLoss", "SoftmaxFocalLoss"]


def focal_loss(
    logits: Tensor,
    targets: Tensor,
    alpha: float | None = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> Tensor:
    """二分类/多标签 focal loss（论文式 (4)，alpha 非 None 时为式 (5)）。

    布局：logits 与 targets 同形状，逐元素独立计算，具体布局不强制，常规输入为
    - 2D 图像：logits=(B, C, H, W)——C=1 二分类（如分割前景概率图），或 C 个独立标签的多标签；
    - 3D 体数据：logits=(B, C, D, H, W)；
    - 逐样本分类：logits=(B, C)。
    targets 为与 logits 同形的 float 张量，取值 0/1（硬标签）或 [0,1]（软标签）。

    参数：
        logits: 未过 sigmoid 的原始输出（float 张量）。
        targets: 与 logits 同形的 float 标签，0/1 或 [0,1]。
        alpha: 式 (5) 中 α_t 的 α ∈ [0,1]（y=1 时权重 α，y=0 时 1-α）；None 表示不加权、
            用式 (4)。论文表 1b 最优 α=0.25。
        gamma: focusing parameter γ ≥ 0，论文最优 2.0；γ=0 退化为 CE（式 (4)）或
            α-平衡 CE（式 (5)/式 (3)）。
        reduction: "none"（逐元素返回）| "mean"（全元素均值）| "sum"。注意论文 §4.1 的训练
            口径是对全部 ~10 万 anchor 的 focal loss 求和后按"分配到 ground-truth 框的
            anchor 数"归一化（正样本数，而非全元素均值）；复现检测训练口径时请按正样本数
            自行归一化，本参数不做该处理。

    返回：
        损失张量；reduction="none" 时与 logits 同形。
    """
    if gamma < 0:
        raise ValueError(f"gamma 必须非负，得到 {gamma}")
    if alpha is not None and not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha 必须在 [0,1] 或为 None，得到 {alpha}")

    # 式 (2) 后正文句 CE(p_t) = -log(p_t)：用 log-sum-exp 形式计算，内部稳定、不触及 log(0)
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    # 式 (2)：p_t = p*t + (1-p)*(1-t)。不做 detach：调制因子对 logits 的导数保留在反传中，
    # 与论文附录式 (9)–(11) 的真导数一致。
    p = torch.sigmoid(logits)
    p_t = p * targets + (1.0 - p) * (1.0 - targets)

    # 式 (4) 的调制因子 (1-p_t)^γ。钳制依据（float32 实测）：合规标签 t∈[0,1] 时 p_t 是
    # p 与 1-p 的凸组合、必在 [0,1] 内（x∈[-300,300] 稠密扫描 p_t>1 出现 0 次），钳制是
    # 无操作；它防越界标签——如 t=2 时 1-p_t 最低 -1，γ 非整数 → NaN、γ 为整数 → torch
    # 静默返回错误权重，钳到 0 使行为良定义（该处样本视为完美分类、FL=0）。
    # γ=0 时钳后 0^0=1（torch/IEEE pow 语义），恰好退化为 CE。
    modulating = (1.0 - p_t).clamp(min=0.0).pow(gamma)
    loss = ce * modulating

    # 式 (5)：α_t = α（y=1）或 1-α（y=0）
    if alpha is not None:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss

    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f'reduction 仅支持 "none"/"mean"/"sum"，得到 {reduction!r}')


def softmax_focal_loss(
    logits: Tensor,
    targets: Tensor,
    alpha: float | Tensor | None = None,
    gamma: float = 2.0,
    reduction: str = "mean",
    ignore_index: int | None = None,
) -> Tensor:
    """多类单标签 focal loss（论文式 (4)/(5) 的 softmax 推广，非论文原式）。

    论文原式为 sigmoid one-vs-all（式 (4)/(5)）；本函数把其中 p_t 换成 softmax 概率
    log_softmax(logits)[target] 的指数，是医疗分割（逐像素多类）中的常见推广。

    布局：
        logits: (B, C, H, W)（2D 逐像素多类分割）或 (B, C, D, H, W)（3D），或 (B, C)（逐样本多类）。
        targets: 对应空间的 int64 类别索引：(B, H, W) / (B, D, H, W) / (B,)，取值 [0, C)
            （ignore_index 处任意，不参与均值）。

    参数：
        alpha: 类别权重。None 不加权；float 对所有类同权；长度为 C 的 1-D Tensor 为逐类权重。
            注意论文 α_t 是正/负两类的二分类权重，多类逐类权重是推广用法。
        gamma: focusing parameter γ ≥ 0；γ=0 退化为交叉熵。
        reduction: "none" | "mean" | "sum"；"mean" 时忽略 ignore_index 处样本。
        ignore_index: 需忽略的标签值（如背景/无效区），None 表示不忽略；
            ignore 位置损失置 0 且不参与 "mean" 的分母。

    返回：
        损失张量；reduction="none" 时形状为空间形状 (B, H, W) / (B, D, H, W) / (B,)。
    """
    if logits.dim() < 2:
        raise ValueError("logits 至少 2 维（B, C, ...）")
    if gamma < 0:
        raise ValueError(f"gamma 必须非负，得到 {gamma}")
    if targets.dtype not in (torch.int64, torch.int32):
        raise TypeError(
            f"targets 必须是 int64/int32 类别索引图，得到 {targets.dtype}（float 软标签请用 focal_loss）"
        )

    # log_softmax：log-sum-exp 内部稳定；取目标类 log 概率用 gather，避免 log(p) 触及 log(0)。
    # gather 的索引不能是负数，ignore_index（可为 -100 等哨兵值）先钳到合法范围，
    # ignore 位置随后由 valid 掩码置零、不参与均值（reduction="none" 时这些位置返回 0）。
    log_probs = F.log_softmax(logits, dim=1)
    gather_idx = targets.clamp(min=0, max=log_probs.size(1) - 1)
    logp_t = log_probs.gather(1, gather_idx.unsqueeze(1)).squeeze(1)
    p_t = logp_t.exp()  # 由 logp 指数还原 p_t，无除法、无 log(0)

    # 调制因子 (1-p_t)^γ：同 focal_loss，先钳下界 0——对合规 [0,1] 标签是无操作（softmax 概率
    # 本就在 [0,1]），防的是越界标签导致的负底数幂（γ 非整数 → NaN）
    modulating = (1.0 - p_t).clamp(min=0.0).pow(gamma)
    loss = -modulating * logp_t  # 式 (4)（p_t 取 softmax 概率）

    if alpha is not None:
        if isinstance(alpha, Tensor):
            a = alpha.view(1, -1, *([1] * (logits.dim() - 2))).to(logits)
            loss = loss * a
        else:
            loss = loss * float(alpha)

    valid: Tensor | None = None
    if ignore_index is not None:
        valid = targets != ignore_index
        loss = loss * valid

    if reduction == "none":
        return loss
    if reduction == "mean":
        if valid is not None:
            return loss.sum() / valid.sum().clamp(min=1)
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    raise ValueError(f'reduction 仅支持 "none"/"mean"/"sum"，得到 {reduction!r}')


class FocalLoss(nn.Module):
    """二分类/多标签 focal loss 的 nn.Module 封装，用作 criterion：loss = FocalLoss(...)(logits, targets)。

    参数与布局同 focal_loss（2D=(B,C,H,W)、3D=(B,C,D,H,W)、逐样本=(B,C)）：
        alpha: 式 (5) 的 α ∈ [0,1]，None 为式 (4)；gamma: γ≥0；reduction: "none"/"mean"/"sum"。
    """

    def __init__(self, alpha: float | None = 0.25, gamma: float = 2.0, reduction: str = "mean") -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return focal_loss(logits, targets, self.alpha, self.gamma, self.reduction)


class SoftmaxFocalLoss(nn.Module):
    """多类单标签 focal loss（softmax 推广）的 nn.Module 封装，用作 criterion。

    参数与布局同 softmax_focal_loss（logits=(B,C,H,W)/(B,C,D,H,W)/(B,C)，targets 为类索引图）：
        alpha: None/float/长度 C 的 Tensor；gamma: γ≥0；reduction: "none"/"mean"/"sum"；
        ignore_index: 忽略的标签值。
    """

    def __init__(
        self,
        alpha: float | Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int | None = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return softmax_focal_loss(logits, targets, self.alpha, self.gamma, self.reduction, self.ignore_index)


# ---------------------------------------------------------------- 自测（CPU，数秒）


def _naive_focal_ref(
    logits_np: np.ndarray, targets_np: np.ndarray, alpha: float | None, gamma: float
) -> np.ndarray:
    """自测用朴素双精度参考：严格按论文式 (2)(4)(5) 逐式直接计算。

    调用方保证 logits 幅度温和（|x|≤3 → p∈[0.047,0.953]），双精度下 log(p_t) 无下溢/上溢风险，
    故无需任何钳制——与主实现的稳定化路径互相独立，可作对拍基准。
    """
    p = 1.0 / (1.0 + np.exp(-logits_np))
    p_t = p * targets_np + (1.0 - p) * (1.0 - targets_np)  # 论文式 (2)
    ce = -np.log(p_t)  # CE(p_t) = -log(p_t)
    fl = (1.0 - p_t) ** gamma * ce  # 论文式 (4)
    if alpha is not None:
        alpha_t = alpha * targets_np + (1.0 - alpha) * (1.0 - targets_np)  # 论文式 (3) 的 α_t
        fl = alpha_t * fl  # 论文式 (5)
    return fl


def _finite_diff_grad(
    fn, x: Tensor, coords: list[tuple[int, ...]], eps: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """对 fn(x) 在指定坐标处做中心差分，返回 (数值梯度, autograd 梯度)。要求 x 为 float64、requires_grad。"""
    loss = fn(x)
    (ana,) = torch.autograd.grad(loss, x, retain_graph=False)

    num = np.zeros(len(coords), dtype=np.float64)
    x_np = x.detach().clone()
    for i, coord in enumerate(coords):
        xp = x_np.clone()
        xp[coord] += eps
        xm = x_np.clone()
        xm[coord] -= eps
        num[i] = (fn(xp).item() - fn(xm).item()) / (2.0 * eps)
    return num, ana.detach().numpy()[tuple(np.array(coords).T)]


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    # ---- 1. 与朴素双精度参考实现逐元素对拍（式 (4) 无 α 与式 (5) 含 α，2D/3D 布局） ----
    for shape in [(2, 1, 8, 8), (2, 3, 4, 4, 4)]:  # 2D=(B,C,H,W) 与 3D=(B,C,D,H,W)
        logits = torch.randn(*shape, dtype=torch.float64).clamp(-3, 3)
        targets = (torch.rand(*shape, dtype=torch.float64) < 0.2).double()
        for alpha, gamma in [(None, 2.0), (0.25, 2.0), (0.25, 0.0)]:
            got = focal_loss(logits, targets, alpha=alpha, gamma=gamma, reduction="none")
            ref = _naive_focal_ref(logits.numpy(), targets.numpy(), alpha, gamma)
            assert torch.isfinite(got).all(), "focal_loss 出现 NaN/Inf"
            assert np.abs(got.numpy() - ref).max() < 1e-10, "与论文式 (2)(4)(5) 的朴素实现不一致"
    print("PASS 与朴素双精度参考实现逐元素对拍（式 4/式 5、2D 与 3D 布局）")

    # ---- 2. γ=0 退化为 CE（式 4→CE；式 5→α-平衡 CE=式 3 加权的 BCE） ----
    logits = torch.randn(4, 1, 6, 6, dtype=torch.float64)
    targets = (torch.rand(4, 1, 6, 6, dtype=torch.float64) < 0.3).double()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    assert torch.allclose(focal_loss(logits, targets, alpha=None, gamma=0.0, reduction="none"), ce)
    alpha_t = 0.25 * targets + 0.75 * (1.0 - targets)
    assert torch.allclose(
        focal_loss(logits, targets, alpha=0.25, gamma=0.0, reduction="none"), alpha_t * ce
    )
    print("PASS γ=0 退化为 CE / α-平衡 CE")

    # ---- 3. 易样本降权量化：γ=2、p_t=σ(5)≈0.9933 时 FL/CE 应≈(1-p_t)²≈4.4e-5（论文 §3.2 的 100 倍量级） ----
    easy_logit = torch.tensor([5.0], dtype=torch.float64)
    easy_target = torch.tensor([1.0], dtype=torch.float64)
    ce_easy = float(F.binary_cross_entropy_with_logits(easy_logit, easy_target))
    ratio = float(focal_loss(easy_logit, easy_target, alpha=None, gamma=2.0, reduction="none")) / ce_easy
    assert 1e-6 < ratio < 1e-3, f"易样本降权比例异常: {ratio}"
    print(f"PASS 易样本降权：p_t=σ(5) 时 FL/CE = {ratio:.2e}（论文：γ=2、p_t=0.9 低 100 倍）")

    # ---- 4. softmax 版 γ=0 与（ignore 口径下的）交叉熵对拍 ----
    logits_mc = torch.randn(2, 5, 6, 6, 4, dtype=torch.float64)  # (B,C,D,H,W) 多类 3D
    targets_mc = torch.randint(0, 5, (2, 6, 6, 4), dtype=torch.int64)
    assert torch.allclose(
        softmax_focal_loss(logits_mc, targets_mc, gamma=0.0, reduction="mean"),
        F.cross_entropy(logits_mc, targets_mc),
    ), "softmax 版 γ=0 未退化到 cross_entropy"
    targets_mc_ig = targets_mc.clone()
    targets_mc_ig[0, 0, 0, :] = -100
    assert torch.allclose(
        softmax_focal_loss(logits_mc, targets_mc_ig, gamma=0.0, ignore_index=-100, reduction="mean"),
        F.cross_entropy(logits_mc, targets_mc_ig, ignore_index=-100),
    ), "ignore_index 口径与 cross_entropy 不一致"
    print("PASS softmax 版：γ=0 退化为 CE（含 ignore_index 口径）")

    # ---- 5. 与 torchvision 官方实现逐元素对拍（可用时；本 venv 的 torchvision 源自 fvcore 实现） ----
    try:
        from torchvision.ops import sigmoid_focal_loss as _tv_focal

        logits32 = torch.randn(2, 1, 8, 8)
        targets32 = (torch.rand(2, 1, 8, 8) < 0.3).float()
        for alpha, gamma in [(0.25, 2.0), (None, 2.0)]:
            got = focal_loss(logits32, targets32, alpha=alpha, gamma=gamma, reduction="none")
            ref = _tv_focal(logits32, targets32, alpha=-1.0 if alpha is None else alpha, gamma=gamma, reduction="none")
            assert (got - ref).abs().max().item() < 1e-6, "与 torchvision 实现不一致"
        print("PASS 与 torchvision.ops.sigmoid_focal_loss 逐元素对拍（float32，|Δ|max<1e-6）")
    except ImportError:
        print("SKIP torchvision 对拍（环境未装 torchvision；朴素双精度对拍已在第 1 项覆盖正确性）")

    # ---- 6. 有限差分验证真导数（含调制因子路径，对照论文附录式 (9)-(11) 的 dFL/dx） ----
    for alpha, gamma in [(None, 2.0), (0.25, 2.0)]:
        x = torch.randn(2, 1, 5, 5, dtype=torch.float64, requires_grad=True).clamp(-3, 3)
        x = x.detach().requires_grad_(True)
        t = (torch.rand(2, 1, 5, 5, dtype=torch.float64) < 0.4).double()
        coords = [(0, 0, i, j) for i in range(2) for j in range(2)] + [(1, 0, 3, 4)]
        num, ana = _finite_diff_grad(lambda z: focal_loss(z, t, alpha=alpha, gamma=gamma, reduction="sum"), x, coords)
        rel = np.abs(num - ana) / (np.abs(num) + np.abs(ana) + 1e-12)
        assert rel.max() < 1e-6, f"有限差分与 autograd 梯度不一致（max rel err={rel.max():.2e}）"

    xs = torch.randn(2, 5, 4, 4, dtype=torch.float64).clamp(-3, 3).detach().requires_grad_(True)
    ts = torch.randint(0, 5, (2, 4, 4), dtype=torch.int64)
    num, ana = _finite_diff_grad(lambda z: softmax_focal_loss(z, ts, gamma=2.0, reduction="sum"), xs, [(0, 1, 2, 3), (1, 3, 0, 1), (0, 0, 0, 0)])
    rel = np.abs(num - ana) / (np.abs(num) + np.abs(ana) + 1e-12)
    assert rel.max() < 1e-6, f"softmax 版梯度不一致（max rel err={rel.max():.2e}）"
    print("PASS 有限差分验证 sigmoid 版与 softmax 版的真导数（rel err < 1e-6）")

    # ---- 7. float32 反向梯度有限性 ----
    logits = torch.randn(2, 1, 8, 8, requires_grad=True)
    targets = (torch.rand(2, 1, 8, 8) < 0.3).float()
    loss = FocalLoss(alpha=0.25, gamma=2.0)(logits, targets)
    loss.backward()
    assert torch.isfinite(loss).all() and torch.isfinite(logits.grad).all(), "float32 反向出现 NaN/Inf"
    assert logits.grad.abs().sum() > 0, "梯度全零"
    print(f"PASS float32 forward+backward 梯度有限（loss={loss.item():.6f}）")

    # ---- 8. 小卷积模型 forward+backward 并打印参数量 ----
    model = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 1, 1),
    )
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(1, 1, 16, 16)
    y = (torch.rand(1, 1, 16, 16) < 0.2).float()  # 模拟 20% 正类的极不平衡分割标签
    loss = FocalLoss(alpha=0.25, gamma=2.0)(model(x), y)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters()), "模型梯度出现 NaN/Inf"
    # SoftmaxFocalLoss 冒烟：多类头 + int64 类别索引标签
    logits_mc = torch.randn(2, 5, 4, 4, requires_grad=True)
    targets_mc = torch.randint(0, 5, (2, 4, 4))
    loss_mc = SoftmaxFocalLoss(gamma=2.0)(logits_mc, targets_mc)
    loss_mc.backward()
    assert torch.isfinite(loss_mc).all() and torch.isfinite(logits_mc.grad).all(), "softmax 版梯度出现 NaN/Inf"
    print(f"PASS 小模型 forward+backward（参数量 {n_params}；loss={loss.item():.6f}，softmax 版 loss={loss_mc.item():.6f}）")

    print("PASS 全部自测通过")
