"""dice-ce-loss — [loss]（可导损失；资产卡：assets/objectives/dice-ce-loss.md）

论文支撑（统一四问，简体中文）：
- 哪篇论文用了它：Isensee, Jaeger, Kohl, Petersen & Maier-Hein（德国癌症研究中心
  DKFZ），《nnU-Net: a self-configuring method for deep learning-based biomedical
  image segmentation》，Nature Methods 2021，
  https://www.nature.com/articles/s41592-020-01008-z （Dice+CE 复合损失是 nnU-Net
  的默认训练损失）。所读版本：Nature 官方文章页（正文付费墙，读摘要与数据集清单）
  + 官方免费补充材料 MOESM1（Supplementary Notes 1–9，44 页 PDF，损失论述在
  Note 2 "Training Scheme"）+ arXiv:1809.10486（同框架早期论文，Dice 公式的
  式 (1)(2) 出处）。
- 目的：生物医学数据类别不平衡严重，稀有前景类欠代表、易被忽略；Dice 损失直接
  对齐评价指标并缓解不平衡，但 patch 式训练下只是近似优化该指标，单独用不稳定；
  CE 提供稠密逐像素梯度稳定收敛，Dice 项把优化对齐到 Dice 指标。补充材料 Note 2
  原文："Empirically, combining the Dice loss with a cross-entropy loss improved
  training stability and segmentation accuracy. Therefore, the two loss terms are
  simply averaged."（两损失项简单取平均）。
- 效果：作为 nnU-Net 默认配置，nnU-Net 不做人工干预即在 23 个公开数据集上超过
  大多数现有方法（含高度特化方案）（摘要）；例：ACDC 五折 CV 前景平均 Dice
  2D/3D 全分辨率/集成 = 0.9165/0.9181/0.9228（补充材料 §3.1 的 ACDC 应用示例，
  原文 "the 2D U-Net obtains a score of 0.9165, the 3D full resolution a score
  of 0.9181 and the ensemble of the two a score of 0.9228"；完整五折表在
  §6.6 Table SN6.26）。论文未对损失项单独消融（补充材料 SN8 只消融激活/上采样/
  深监督），"组合提升稳定性与精度"是经验结论。
- 结论：Dice+CE 简单平均是 nnU-Net 默认训练损失，已成医学分割事实标准；适用
  边界——softmax 多类单标签分割；区域式任务（体素可属多结构）nnU-Net v2 改用
  BCE+Dice 变体（代码 DC_and_BCE_loss）。

实现说明（关键公式 ↔ 论文式子）：
- 复合损失：L = 0.5·L_dice + 0.5·L_ce。权重 0.5/0.5 依 Nature 版补充材料
  Note 2 原文"the two loss terms are simply averaged"（简单平均）。
- soft Dice loss（arXiv:1809.10486 式 (2)，Nature 版正文付费墙未含显式公式，
  由同框架早期论文给出、口径与官方补充材料及开源 nnU-Net 一致）：
      L_dc = -(1/|K|) · Σ_{k∈K} [ 2·Σ_i u_i^k·v_i^k ] / [ Σ_i u_i^k + Σ_i v_i^k ]
  其中 u = softmax(logits)、v = one-hot 真值、i 遍历批内体素；K 为类别集合
  （arXiv 原文对 K 仅写 "k∈K being the classes"，未明示排除背景；不含背景是
  nnU-Net 代码约定 do_bg=False，本实现默认遵循，见"不一致处 4"）。注意式 (2)
  是 -DC：**无 "+1-DC" 常数项**，完美分割时 L_dc = -1
  （负极值），与 nnunetv2 返回 -dc.mean() 一致——nnU-Net 训练日志中损失为负属
  正常现象，比较曲线时不要把负值当异常。式 (1) 为总式 L_total = L_dice + L_CE
  （早期版为求和；Nature 版改为平均，见下面"不一致处 1"）。
- CE：逐像素标准交叉熵（含背景类），用 F.cross_entropy（log_softmax，
  log-sum-exp 内部稳定，不显式 log(p)）。
- 参考实现（本地落点）：
  /ssd/lx/totalsegmentator/nnunetv2/training/loss/compound_losses.py
  （DC_and_CE_loss，类定义在第 8 行）；
  /ssd/lx/totalsegmentator/nnunetv2/training/loss/dice.py（SoftDiceLoss 与
  MemoryEfficientSoftDiceLoss：dc = (2*intersect + smooth) / clip(sum_gt +
  sum_pred + smooth, 1e-8)，do_bg=False 时对 dc[1:] 取均值）；
  /ssd/lx/totalsegmentator/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:399
  （实例化：DC_and_CE_loss({'batch_dice': ..., 'smooth': 1e-5, 'do_bg': False},
  {}, weight_ce=1, weight_dice=1, dice_class=MemoryEfficientSoftDiceLoss)）。
- 与开源实现不一致处（均以论文为准或为数值稳定）：
  1) 权重：Nature 版补充材料明言两损失项"简单平均"（0.5/0.5）；nnunetv2 代码
     默认 weight_ce=1, weight_dice=1（求和）。本实现默认 0.5/0.5 以论文为准，
     weight_ce/weight_dice 参数可改回 1/1。两者最优解相同、只差常数因子 2（改变
     损失尺度从而与 lr 耦合），比较实验时固定口径即可。
  2) 平滑项：arXiv 式 (2) 无平滑；nnunetv2 加 smooth=1e-5 并对分母
     clamp(min=1e-8)。本实现保留 smooth 参数（默认 1e-5，nnunetv2 trainer 同值）
     与分母钳制（eps=1e-8，nnunetv2 dice.py 同值）——两者均为纯数值稳定措施
     （防空前景类 0/0 与零除 NaN），不属于论文公式语义；smooth=0.0 时与式 (2)
     严格一致（钳制仍兜底）。自测第 1 项以 smooth=0.0 路径对拍论文公式。
  3) batch dice：论文文字——对"近全患者输入"的 3D U-Net 逐样本计 dice 再对批
     平均；其余网络把整批当伪体素合并计 dice（式 (2) 的 Σ_{i∈I} 即伪体素口径，
     对应 nnunetv2 的 batch_dice=True）。本实现默认 batch_dice=True（伪体素，
     与式 (2) 一致），可切 False（逐样本逐类计 dice 再对全部取均值）。
  4) do_bg：本实现 |K| 取前景类集合（不含背景），对应 nnunetv2 默认
     do_bg=False（nnUNetTrainer.py:400、dice.py:48/96 的 do_bg 分支）。注意这
     是代码约定而非公式原文：arXiv 式 (2) 对 K 的原文仅 "k∈K being the
     classes"，未明示排除背景。exclude_background=False 时含背景类（nnunetv2 的
     do_bg=True，仅用于区域式 BCE+Dice 组合，不在本条目默认路径）。
- 数值稳定清单：dice 分母 (den + smooth).clamp(min=eps)（1e-8，nnunetv2 同值，
  防零除）；smooth=1e-5（nnunetv2 trainer 同值，防空前景 0/0）；CE/softmax 走
  log-sum-exp（F.cross_entropy / F.softmax），无显式 log(p)、无 exp 上溢风险
  （softmax 内部减最大值）。ignore_index 仅作用于 CE 项（nnunetv2 同口径：
  dice 侧用 loss_mask 而非 ignore，本实现不实现 loss_mask）。

自测（CPU，数秒，固定种子）：见文件末 __main__——与朴素双精度论文公式对拍
（2D/3D、batch_dice 两种口径、权重退化）、完美预测达式 (2) 负极值 -1、缺席
前景类有限性、ignore_index 与 F.cross_entropy 口径对齐、有限差分梯度验证、
float32 与极值 logits 的 forward+backward 有限性、小卷积模型 forward+backward
并打印参数量；逐项 print("PASS ...")。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["soft_dice_loss", "dice_ce_loss", "DiceCELoss"]


def _prepare_target(logits: Tensor, target: Tensor) -> Tensor:
    """把 target 规整成与 logits 空间形状匹配的 int64 类别索引图。

    布局：logits 为 (B, C, H, W)（2D）或 (B, C, D, H, W)（3D）；target 接受
    (B, H, W)/(B, D, H, W) 的类别索引（值 [0, C)），或 nnUNet 风格的
    (B, 1, H, W)/(B, 1, D, H, W)（自动去掉大小为 1 的通道维）。
    """
    if target.dim() == logits.dim() and target.shape[1] == 1:
        target = target.squeeze(1)
    if target.dim() != logits.dim() - 1:
        raise ValueError(
            f"target 形状 {tuple(target.shape)} 与 logits 形状 {tuple(logits.shape)} "
            "不匹配：期望 (B, *spatial) 或 (B, 1, *spatial) 的类别索引图"
        )
    if target.shape != logits.shape[:1] + logits.shape[2:]:
        raise ValueError(
            f"target 空间形状 {tuple(target.shape)} 与 logits {tuple(logits.shape)} 不一致"
        )
    return target.long()


def soft_dice_loss(
    logits: Tensor,
    target: Tensor,
    batch_dice: bool = True,
    exclude_background: bool = True,
    smooth: float = 1e-5,
    eps: float = 1e-8,
) -> Tensor:
    """多类 soft Dice loss（论文式 (2)，arXiv:1809.10486，含数值平滑）。

    布局：
        logits: (B, C, H, W)（2D）或 (B, C, D, H, W)（3D），未过 softmax 的原始输出；
        target: (B, *spatial) 或 (B, 1, *spatial) 的 int64 类别索引，值 [0, C)。

    参数：
        batch_dice: True 时把整批当伪体素合并计每类 Dice（式 (2) 的 Σ_{i∈I} 口径，
            nnU-Net 默认）；False 时逐样本计再对全部取均值。
        exclude_background: True 时仅对前景类 {1, …, C-1} 平均（nnU-Net 代码约定
            do_bg=False；arXiv 式 (2) 的 K 原文只写 "the classes"，未明示排除
            背景）；False 时含背景类。
        smooth: 平滑项，默认 1e-5（nnunetv2 trainer 同值）；纯数值稳定（防空前景
            0/0），论文式 (2) 无此项，置 0.0 即严格论文公式。
        eps: 分母钳制下界 1e-8（nnunetv2 dice.py 同值），防零除。

    返回：
        标量损失 = -mean_k Dice_k（式 (2) 原式，无 "+1-DC" 的常数项），取值约
        [-1, 0]：完美分割（每类 Dice→1）时取 -1，完全未预测出前景时趋于 0。
        常数偏移不影响梯度，nnunetv2 同样返回 -dc.mean()。
    """
    if logits.dim() < 3:
        raise ValueError("logits 至少 3 维（B, C, *spatial）")
    target = _prepare_target(logits, target)
    n_classes = logits.shape[1]

    probs = F.softmax(logits, dim=1)
    # one-hot：(B, *spatial) -> (B, C, *spatial)，float 以参与概率乘积
    onehot = F.one_hot(target, num_classes=n_classes).to(probs.dtype)
    onehot = onehot.permute(0, -1, *range(1, target.dim()))

    if exclude_background:
        probs = probs[:, 1:]
        onehot = onehot[:, 1:]

    # 分子 2Σuv、分母 Σu+Σv，逐类计算（式 (2) 的分式内部）
    if batch_dice:
        dims: tuple[int, ...] = (0, *range(2, probs.dim()))  # 批内体素合并
    else:
        dims = tuple(range(2, probs.dim()))  # 逐样本
    inter = (probs * onehot).sum(dim=dims)
    sum_pred = probs.sum(dim=dims)
    sum_gt = onehot.sum(dim=dims)

    # 数值稳定：+smooth 防空前景类 0/0；clamp(min=eps) 防零除（取值依据见 docstring）
    dc = (2.0 * inter + smooth) / (sum_pred + sum_gt + smooth).clamp(min=eps)
    return -dc.mean()


def dice_ce_loss(
    logits: Tensor,
    target: Tensor,
    weight_ce: float = 0.5,
    weight_dice: float = 0.5,
    batch_dice: bool = True,
    exclude_background: bool = True,
    smooth: float = 1e-5,
    eps: float = 1e-8,
    ignore_index: int | None = None,
) -> Tensor:
    """Dice + 交叉熵复合损失（nnU-Net 默认训练损失；Nature 版论文口径为 0.5/0.5 平均）。

    L = weight_ce * L_ce + weight_dice * L_dice，默认 weight_ce=weight_dice=0.5
    （补充材料 Note 2 原文"the two loss terms are simply averaged"；nnunetv2 代码
    用 1/1，等价差常数因子 2）。

    布局：
        logits: (B, C, H, W)（2D）或 (B, C, D, H, W)（3D），未过 softmax；
        target: (B, *spatial) 或 (B, 1, *spatial) 的 int64 类别索引，值 [0, C)
            （ignore_index 处任意，CE 项不统计）。

    参数：
        weight_ce / weight_dice: 两损失项权重，不必和为 1（nnunetv2 注释同口径）；
            置 0 时跳过对应项的计算。
        batch_dice / exclude_background / smooth / eps: 同 soft_dice_loss。
        ignore_index: CE 项忽略的标签值（如无效区）；仅作用于 CE（nnunetv2 同口径）。

    返回：
        标量损失张量。注意取值**可以为负**：CE 项 ∈ [0, ∞)，Dice 项 ∈ [-1, 0]
        （式 (2) 是 -DC，无 "+1" 常数），完美预测时复合损失 → 0.5·0 + 0.5·(-1)
        = -0.5；nnU-Net 训练日志中损失为负属正常现象。
    """
    if logits.dim() < 3:
        raise ValueError("logits 至少 3 维（B, C, *spatial）")
    target = _prepare_target(logits, target)

    loss = torch.zeros((), dtype=logits.dtype, device=logits.device)
    if weight_ce != 0:
        # CE 含背景类、逐像素平均；log_softmax 内部稳定。
        # 该 torch 版本的 cross_entropy 不接受 ignore_index=None，故条件传参
        ce_kw = {} if ignore_index is None else {"ignore_index": ignore_index}
        ce = F.cross_entropy(logits, target, **ce_kw)
        loss = loss + weight_ce * ce
    if weight_dice != 0:
        dl = soft_dice_loss(
            logits, target,
            batch_dice=batch_dice, exclude_background=exclude_background,
            smooth=smooth, eps=eps,
        )
        loss = loss + weight_dice * dl
    return loss


class DiceCELoss(nn.Module):
    """Dice+CE 复合损失的 nn.Module 封装，用作 criterion：loss = DiceCELoss(...)(logits, target)。

    参数与布局同 dice_ce_loss（logits=(B,C,H,W)/(B,C,D,H,W)，target 为类别索引图）。
    """

    def __init__(
        self,
        weight_ce: float = 0.5,
        weight_dice: float = 0.5,
        batch_dice: bool = True,
        exclude_background: bool = True,
        smooth: float = 1e-5,
        eps: float = 1e-8,
        ignore_index: int | None = None,
    ) -> None:
        super().__init__()
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.batch_dice = batch_dice
        self.exclude_background = exclude_background
        self.smooth = smooth
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return dice_ce_loss(
            logits, target,
            weight_ce=self.weight_ce, weight_dice=self.weight_dice,
            batch_dice=self.batch_dice, exclude_background=self.exclude_background,
            smooth=self.smooth, eps=self.eps, ignore_index=self.ignore_index,
        )


# ---------------------------------------------------------------- 自测（CPU，数秒）


def _naive_dice_ce_ref(
    logits_np: np.ndarray,
    target_np: np.ndarray,
    weight_ce: float,
    weight_dice: float,
    batch_dice: bool,
    exclude_background: bool,
) -> float:
    """自测用朴素双精度参考：严格按论文式 (2) 与 CE 定义直接计算，无 smooth/clamp。

    调用方保证 logits 幅度温和、每类分母远离 0，双精度下无 0/0 风险，
    故可不加平滑——与主实现的稳定化路径互相独立，可作对拍基准。
    """
    b, c = logits_np.shape[:2]
    # softmax（减最大值防溢出，纯实现细节）
    z = logits_np - logits_np.max(axis=1, keepdims=True)
    e = np.exp(z)
    u = e / e.sum(axis=1, keepdims=True)
    # one-hot：(B, *s) -> (B, C, *s)
    v = np.moveaxis(np.eye(c, dtype=np.float64)[target_np], -1, 1)

    # CE 先用完整类别维收集 log-prob（含背景类），再做前景切片（切片只影响 Dice 项）
    logp = np.log(np.take_along_axis(u, target_np[:, None], axis=1)[:, 0])
    ce = -logp.mean()

    if exclude_background:
        u, v = u[:, 1:], v[:, 1:]
    voxel_axes: tuple[int, ...] = tuple(range(2, u.ndim))
    if batch_dice:
        axes: tuple[int, ...] = (0, *voxel_axes)  # 批内合并
    else:
        axes = voxel_axes  # 逐样本
    dc = 2.0 * (u * v).sum(axis=axes) / (u.sum(axis=axes) + v.sum(axis=axes))  # 式 (2) 分式
    dice_loss = -dc.mean()

    return weight_ce * ce + weight_dice * dice_loss


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

    # ---- 1. 与朴素双精度论文公式对拍（式 (2)+CE；2D/3D、batch_dice 两口径、含/不含背景、权重退化；smooth=0 走严格论文路径） ----
    for shape in [(2, 4, 8, 8), (1, 3, 6, 5, 4)]:  # 2D=(B,C,H,W) 与 3D=(B,C,D,H,W)
        logits = torch.randn(*shape, dtype=torch.float64)
        target = torch.randint(0, shape[1], shape[:1] + shape[2:], dtype=torch.int64)
        for batch_dice in (True, False):
            for excl_bg in (True, False):
                for wce, wdc in [(0.5, 0.5), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0)]:
                    got = dice_ce_loss(logits, target, weight_ce=wce, weight_dice=wdc,
                                       batch_dice=batch_dice, exclude_background=excl_bg, smooth=0.0)
                    ref = _naive_dice_ce_ref(logits.numpy(), target.numpy(), wce, wdc, batch_dice, excl_bg)
                    assert torch.isfinite(got).all(), "dice_ce_loss 出现 NaN/Inf"
                    err = abs(got.item() - ref) / (abs(ref) + 1e-12)
                    assert err < 1e-10, f"与论文式 (2) 朴素实现不一致：rel err={err:.2e}"
    print("PASS 与朴素双精度论文公式对拍（式 2+CE，2D/3D、batch_dice 两口径、4 组权重）")

    # ---- 2. 权重退化：weight_dice=0 时与 F.cross_entropy 一致；ignore_index 口径对齐 ----
    logits = torch.randn(2, 5, 6, 6, 4, dtype=torch.float64)
    target = torch.randint(0, 5, (2, 6, 6, 4), dtype=torch.int64)
    assert torch.allclose(
        dice_ce_loss(logits, target, weight_ce=1.0, weight_dice=0.0),
        F.cross_entropy(logits, target),
    ), "weight_dice=0 未退化到标准交叉熵"
    t_ig = target.clone()
    t_ig[0, 0, 0, :] = -100
    assert torch.allclose(
        dice_ce_loss(logits, t_ig, weight_ce=1.0, weight_dice=0.0, ignore_index=-100),
        F.cross_entropy(logits, t_ig, ignore_index=-100),
    ), "ignore_index 口径与 cross_entropy 不一致"
    print("PASS weight_dice=0 退化为 CE；ignore_index 口径与 F.cross_entropy 一致")

    # ---- 3. 完美预测达负极值：式 (2) 是 -DC（无 +1 常数），完美分割 → Dice 项 -1、CE 项 0；±30 极值 logits 触发钳制路径 ----
    onehot_t = F.one_hot(target, num_classes=5).permute(0, -1, *range(1, target.dim()))
    perfect = (30.0 * onehot_t - 30.0 * (1.0 - onehot_t)).double()  # 强 logits（softmax 饱和到 1）
    dice_only = dice_ce_loss(perfect, target, weight_ce=0.0, weight_dice=1.0)
    ce_only = dice_ce_loss(perfect, target, weight_ce=1.0, weight_dice=0.0)
    assert torch.isfinite(dice_only) and -1.0 - 1e-2 < dice_only.item() <= -1.0 + 1e-2, \
        f"完美预测 Dice 项应≈-1: {dice_only.item()}"
    assert torch.isfinite(ce_only) and 0.0 <= ce_only.item() < 1e-2, \
        f"完美预测 CE 项应≈0: {ce_only.item()}"
    loss_p = dice_ce_loss(perfect, target)
    assert abs(loss_p.item() - 0.5 * (ce_only.item() + dice_only.item())) < 1e-9
    print(f"PASS 完美预测：Dice 项={dice_only.item():.4f}（式 2 负极值 -1）、CE 项={ce_only.item():.2e}、复合={loss_p.item():.4f}，极值 logits 无 NaN")

    # ---- 4. 缺席前景类有限性：GT 全背景时 Dice 项仍有限（分母钳制 + smooth 防空前景 0/0） ----
    target_bg = torch.zeros(2, 5, 5, dtype=torch.int64)  # GT 全为背景（类 0），前景类缺席
    loss_bg = dice_ce_loss(torch.randn(2, 4, 5, 5, dtype=torch.float64), target_bg)
    assert torch.isfinite(loss_bg), "全背景 GT 下损失出现 NaN/Inf"
    loss_bg_s0 = dice_ce_loss(torch.randn(2, 4, 5, 5, dtype=torch.float64), target_bg, smooth=0.0)
    assert torch.isfinite(loss_bg_s0), "smooth=0 时分母钳制失效"
    print(f"PASS 缺席前景类（GT 全背景）损失有限：smooth=1e-5 → {loss_bg.item():.4f}，smooth=0 → {loss_bg_s0.item():.4f}")

    # ---- 5. 有限差分验证真导数（默认 smooth=1e-5 路径，2D 与 3D 各测若干坐标） ----
    for shape in [(2, 4, 5, 5), (1, 3, 4, 4, 4)]:
        x = torch.randn(*shape, dtype=torch.float64).detach().requires_grad_(True)
        t = torch.randint(0, shape[1], shape[:1] + shape[2:], dtype=torch.int64)
        # 取若干跨类/跨空间坐标
        coords = [(0, 0, 0, 0), (0, 1, 2, 3), (1, 2, 4, 1), (1, 3, 0, 4)]
        coords = [c for c in coords if all(i < s for i, s in zip(c, x.shape))]
        if x.dim() == 5:
            coords = [c + (1,) for c in coords]
        num, ana = _finite_diff_grad(
            lambda z: dice_ce_loss(z, t, batch_dice=False), x, coords
        )
        rel = np.abs(num - ana) / (np.abs(num) + np.abs(ana) + 1e-12)
        assert rel.max() < 1e-6, f"有限差分与 autograd 不一致（max rel err={rel.max():.2e}）"
    print("PASS 有限差分验证复合损失真导数（2D/3D，rel err < 1e-6）")

    # ---- 6. float32 forward+backward 梯度有限（损失函数本体） ----
    logits = torch.randn(2, 4, 16, 16, requires_grad=True)
    target = torch.randint(0, 4, (2, 16, 16), dtype=torch.int64)
    loss = dice_ce_loss(logits, target)
    loss.backward()
    assert torch.isfinite(loss).all() and torch.isfinite(logits.grad).all(), "float32 反向出现 NaN/Inf"
    assert logits.grad.abs().sum() > 0, "梯度全零"
    print(f"PASS float32 forward+backward 梯度有限（loss={loss.item():.6f}）")

    # ---- 7. 小卷积模型 forward+backward 并打印参数量（2D 多类分割 + DiceCELoss 封装类） ----
    model = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 4, 1),
    )
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(1, 1, 32, 32)
    y = torch.randint(0, 4, (1, 32, 32), dtype=torch.int64)
    criterion = DiceCELoss()  # 默认 0.5/0.5（论文"简单平均"口径）
    loss = criterion(model(x), y)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters()), "模型梯度出现 NaN/Inf"
    print(f"PASS 小模型 forward+backward（2D，参数量 {n_params}；loss={loss.item():.6f}）")

    # ---- 8. 3D 布局模型冒烟：Conv3d 输出 (B,C,D,H,W) 走同一损失并反传 ----
    model3d = nn.Conv3d(1, 3, 3, padding=1)
    n_params3d = sum(p.numel() for p in model3d.parameters())
    x3d = torch.randn(1, 1, 8, 8, 8)
    y3d = torch.randint(0, 3, (1, 8, 8, 8), dtype=torch.int64)
    loss3d = DiceCELoss(batch_dice=False)(model3d(x3d), y3d)
    loss3d.backward()
    assert torch.isfinite(loss3d) and torch.isfinite(model3d.weight.grad).all(), "3D 模型梯度出现 NaN/Inf"
    print(f"PASS 3D 布局模型 forward+backward（参数量 {n_params3d}；loss={loss3d.item():.6f}）")

    print("PASS 全部自测通过")
