"""class-weighted-dice-ce-loss — 类加权 Dice + 交叉熵复合损失 [loss]（资产卡：assets/objectives/class-weighted-dice-ce-loss.md）

论文支撑（统一四问，简体中文）：
- 哪篇论文用了它：Guo、Gong、Jiang、Yang、Zheng 等（华中科技大学同济医学院
  附属协和医院等，Thy-uNET），《Development and validation of AI-based
  automatic segmentation and measurement of thymus on chest CT scans》，
  BMC Medical Imaging 2025;25:249，
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12220788/ 。所读版本：PMC 开放获取
  全文 + 官方补充材料 DOCX（Table S1–S3，2026-10-08 解析）。CE 逐类权重机制
  本身是标准做法（F.cross_entropy 的 weight 参数；nnunetv2 的 DC_and_CE_loss
  即支持），无单独原始论文；Thy-uNET 是"小器官分割场景用它抗类不平衡"的直接
  论文证据。
- 目的：胸腺等小目标在全图（全程胸部 CT）中体素占比极低，未加权时其损失梯度
  被背景淹没，粗分割网络易整体忽略小类——而两阶段 coarse-to-fine 流水线里
  粗分割的召回就是系统上限。论文原文："To address the issue of class imbalance
  arising from the small size of the thymus, we adjusted the loss function by
  increasing the weight assigned to the thymus category"。
- 效果：该加权支撑粗分割阶段"to better learn the thymic region and capture its
  approximate location"；整系统（两阶段 nnU-Net）内部测试 Dice 0.85 / IoU 0.75，
  多中心外部 0.82/0.70（286 例）。注意：论文**未做加权 on/off 消融，也未公开
  权重数值**（正文、补充材料、官方代码仓库均未给出）——"加权有效"是作者机制
  陈述而非消融证据，权重数值须任务仓库自调。
- 结论：小占比目标场景的低成本第一杠杆（改损失不动数据流）；适用边界——softmax
  单标签多类分割。权重过大会使背景欠拟合、假阳上升：在"输出只当 ROI 定位用、
  假阳可被下游精分割纠正"的粗分割阶段可放大，精分割阶段慎用。

实现说明（关键公式 ↔ 论文/代码口径）：
- 复合损失沿用 Dice+CE 家族（见 assets/objectives/dice-ce-loss.md）：
      L = weight_ce · L_ce^w + weight_dice · L_dice^w（默认 0.5/0.5）
- 类加权 CE（= F.cross_entropy(weight=w) 的标准语义）：
      L_ce^w = Σ_i w_{y_i}·(-log softmax(logits)_{i,y_i}) / Σ_i w_{y_i}
  （i 遍历体素；含背景类，背景权重通常保持 1）。
- 类加权 Dice（本实现的可加权扩展；nnunetv2 原版 Dice 项无权重）：
  逐类 soft Dice（式同 dice-ce-loss 家族的 -DC 口径）后取权重归一化平均：
      L_dice^w = -Σ_k w_k·DC_k / Σ_k w_k   （k 遍历参与平均的类，
      batch_dice=True 时各类跨批合并，False 时按 (样本, 类) 元素平均）
- 与参考实现不一致处（均注明依据）：
  1) 论文只说 "increasing the weight assigned to the thymus category"，未说明
     权重作用于 CE、Dice 还是两者。本实现默认两者同时加权
     （apply_to_ce=True, apply_to_dice=True），可按需关闭；与 nnunetv2 对齐时
     关 apply_to_dice（其 DC_and_CE_loss 仅 CE 支持 weight）。
  2) Dice 权重默认做归一化平均（dice_weight_norm=True，权重只改类间相对比重，
     不改变损失整体尺度、不与 lr 耦合）；代价是**单一前景类的二类分割下加权
     Dice 项是恒等操作**（比值上下同乘抵消），此时加权只能靠 CE 项——需要
     Dice 项也生效可设 dice_weight_norm=False（改为 Σ_k w_k·DC_k/|K| 的
     非归一化口径，会放大损失尺度、与 lr 耦合，对比实验须固定口径）。
  3) 权重数值无文献锚点（论文未公开），默认 class_weights=None 即全 1，严格
     退化为本仓库 dice-ce-loss 家族口径（自测第 2 项验证退化一致）。
- 参考实现（本地落点）：
  /ssd/lx/totalsegmentator/nnunetv2/training/loss/compound_losses.py
  （DC_and_CE_loss：CE 项 weight 即逐类权重；Dice 项无权重）；
  本仓库 assets/objectives/dice-ce-loss.py（未加权家族口径，本文件不 import
  它——资产代码自包含，仅口径对齐）。

布局与数值稳定：
- logits: (B, C, H, W)（2D）或 (B, C, D, H, W)（3D），未过 softmax；target:
  (B, *spatial) 或 (B, 1, *spatial) 的 int64 类别索引，值 [0, C)。
- class_weights: 长度 C 的一维正数序列，索引即类别（背景=0，通常保持 1）。
- 数值稳定清单：Dice 分母 (sum_pred + sum_gt + smooth).clamp(min=eps)，smooth
  默认 1e-5、eps 1e-8（nnunetv2 同值，防零除与空前景 0/0）；CE/softmax 走
  F.cross_entropy（log-sum-exp 内部稳定）；权重校验非正值/长度错/NaN 即抛
  ValueError，不静默。

自测（CPU，数秒，固定种子）：见文件末 __main__——与朴素双精度参考对拍
（2D/3D、batch_dice 两口径、apply 组合、归一化两口径；smooth=0 走严格无平滑
路径）、全 1 权重退化与未加权口径一致、加权方向性（坏预测类的权重放大使损失
上升）、完美预测负极值、缺席前景类有限性、有限差分验证真导数、float32
forward+backward、小卷积模型端到端并打印参数量、非法权重校验；逐项
print("PASS ...")。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["class_weighted_dice_ce_loss", "ClassWeightedDiceCELoss"]

_EPS_DENOM = 1e-8  # Dice 分母钳制下界（nnunetv2 dice.py 同值，防零除）

# class_weights 的公开参数类型：长度 C 的正数序列（背景=0），None → 全 1
ClassWeightsLike = Tensor | tuple[float, ...] | list[float] | np.ndarray | None


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


def _prepare_class_weights(
    class_weights: ClassWeightsLike, n_classes: int,
    dtype: torch.dtype, device: torch.device,
) -> Tensor:
    """校验并规整类权重为长度 n_classes 的正数张量；None → 全 1（退化口径）。"""
    if class_weights is None:
        return torch.ones(n_classes, dtype=dtype, device=device)
    w = torch.as_tensor(class_weights, dtype=dtype, device=device)
    if w.shape != (n_classes,):
        raise ValueError(
            f"class_weights 长度 {tuple(w.shape)} 应为 ({n_classes},)（背景=0，目标类≥1）"
        )
    if not torch.isfinite(w).all() or not (w > 0).all():
        raise ValueError("class_weights 必须全为正的有限数（F.cross_entropy 语义不允许非正权重）")
    return w


def class_weighted_dice_ce_loss(
    logits: Tensor,
    target: Tensor,
    class_weights: ClassWeightsLike = None,
    weight_ce: float = 0.5,
    weight_dice: float = 0.5,
    apply_to_ce: bool = True,
    apply_to_dice: bool = True,
    dice_weight_norm: bool = True,
    batch_dice: bool = True,
    exclude_background: bool = True,
    smooth: float = 1e-5,
    eps: float = _EPS_DENOM,
    ignore_index: int | None = None,
) -> Tensor:
    """类加权 Dice + 交叉熵复合损失（Thy-uNET 粗分割阶段口径；默认全 1 权重
    严格退化为本仓库 dice-ce-loss 家族）。

    布局：
        logits: (B, C, H, W)（2D）或 (B, C, D, H, W)（3D），未过 softmax；
        target: (B, *spatial) 或 (B, 1, *spatial) 的 int64 类别索引，值 [0, C)
            （若使用 ignore_index，target 含该值时必须 weight_dice=0——Dice 项的
            one_hot 不接受负标签，家族同口径）。

    参数：
        class_weights: 长度 C 的正数序列，w[k] 为第 k 类权重（背景=0 通常保持
            1；目标类取 >1 抗不平衡，Thy-uNET 未公开数值，2–10 起步网格搜）。
            None → 全 1。
        weight_ce / weight_dice: 两损失项权重（0.5/0.5 为家族论文"简单平均"
            口径；nnunetv2 代码 1/1 等价差常数因子 2）；置 0 跳过对应项。
        apply_to_ce / apply_to_dice: 权重是否作用于对应项（论文未说明作用项，
            默认两者都加；与 nnunetv2 对齐时关 apply_to_dice）。
        dice_weight_norm: True（默认）= Dice 项权重归一化平均（只改类间相对
            比重，不动损失尺度；单一前景类时是恒等操作）；False = 非归一化
            Σ w_k·DC_k/|K|（二类分割下加权也生效，但放大尺度、与 lr 耦合）。
        batch_dice: True 时整批当伪体素合并计每类 Dice（家族/nnU-Net 默认）；
            False 时逐样本逐类计再对全部元素平均。
        exclude_background: True 时 Dice 项仅统计前景类 {1..C-1}（nnU-Net 代码
            约定 do_bg=False；CE 始终含背景类）。
        smooth: Dice 平滑项 1e-5（nnunetv2 trainer 同值，防空前景 0/0；置 0
            即严格无平滑公式）。
        eps: Dice 分母钳制下界 1e-8（nnunetv2 dice.py 同值，防零除）。
        ignore_index: CE 项忽略的标签值（如无效区）；仅作用于 CE，target 含该值
            时须 weight_dice=0（Dice 项 one_hot 不接受负标签，家族同口径）。

    返回：
        标量损失张量，取值**可为负**：Dice 项是 -DC（家族口径无 "+1" 常数），
        完美预测时 Dice 项 → -1、CE 项 → 0，复合损失 → 0.5·0+0.5·(-1) = -0.5
        （默认 0.5/0.5 权重下）。
    """
    if logits.dim() < 3:
        raise ValueError("logits 至少 3 维（B, C, *spatial）")
    target = _prepare_target(logits, target)
    n_classes = logits.shape[1]
    w = _prepare_class_weights(class_weights, n_classes, logits.dtype, logits.device)

    loss = torch.zeros((), dtype=logits.dtype, device=logits.device)

    if weight_ce != 0:
        ce_kw: dict = {"weight": w} if apply_to_ce else {}
        if ignore_index is not None:
            ce_kw["ignore_index"] = ignore_index
        loss = loss + weight_ce * F.cross_entropy(logits, target, **ce_kw)

    if weight_dice != 0:
        probs = F.softmax(logits, dim=1)
        onehot = F.one_hot(target, num_classes=n_classes).to(probs.dtype)
        onehot = onehot.permute(0, -1, *range(1, target.dim()))
        if exclude_background:
            probs = probs[:, 1:]
            onehot = onehot[:, 1:]
            w_dice = w[1:]
        else:
            w_dice = w

        # 逐类 Dice：分子 2Σuv、分母 Σu+Σv（batch_dice=True 时跨批合并为伪体素）
        dims: tuple[int, ...] = (0, *range(2, probs.dim())) if batch_dice else tuple(range(2, probs.dim()))
        inter = (probs * onehot).sum(dim=dims)
        dc = (2.0 * inter + smooth) / (probs.sum(dim=dims) + onehot.sum(dim=dims) + smooth).clamp(min=eps)

        if apply_to_dice:
            w_b = w_dice.expand_as(dc)  # (B,K) 逐样本口径下按类广播
            if dice_weight_norm:
                dl = -(dc * w_b).sum() / w_b.sum()
            else:
                dl = -(dc * w_b).sum() / dc.numel()
        else:
            dl = -dc.mean()
        loss = loss + weight_dice * dl

    return loss


class ClassWeightedDiceCELoss(nn.Module):
    """类加权 Dice+CE 复合损失的 nn.Module 封装，用作 criterion：
    loss = ClassWeightedDiceCELoss(class_weights=[1, 5])(logits, target)。

    参数与布局同 class_weighted_dice_ce_loss（logits=(B,C,H,W)/(B,C,D,H,W)，
    target 为类别索引图；class_weights 可在构造时固定，也可为 None）。
    """

    def __init__(
        self,
        class_weights: ClassWeightsLike = None,
        weight_ce: float = 0.5,
        weight_dice: float = 0.5,
        apply_to_ce: bool = True,
        apply_to_dice: bool = True,
        dice_weight_norm: bool = True,
        batch_dice: bool = True,
        exclude_background: bool = True,
        smooth: float = 1e-5,
        eps: float = _EPS_DENOM,
        ignore_index: int | None = None,
    ) -> None:
        super().__init__()
        self.class_weights = class_weights
        self.weight_ce = weight_ce
        self.weight_dice = weight_dice
        self.apply_to_ce = apply_to_ce
        self.apply_to_dice = apply_to_dice
        self.dice_weight_norm = dice_weight_norm
        self.batch_dice = batch_dice
        self.exclude_background = exclude_background
        self.smooth = smooth
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, logits: Tensor, target: Tensor) -> Tensor:
        return class_weighted_dice_ce_loss(
            logits, target,
            class_weights=self.class_weights,
            weight_ce=self.weight_ce, weight_dice=self.weight_dice,
            apply_to_ce=self.apply_to_ce, apply_to_dice=self.apply_to_dice,
            dice_weight_norm=self.dice_weight_norm,
            batch_dice=self.batch_dice, exclude_background=self.exclude_background,
            smooth=self.smooth, eps=self.eps, ignore_index=self.ignore_index,
        )


# ---------------------------------------------------------------- 自测（CPU，数秒）


def _naive_weighted_ref(
    logits_np: np.ndarray,
    target_np: np.ndarray,
    w_np: np.ndarray,
    weight_ce: float,
    weight_dice: float,
    apply_ce: bool,
    apply_dice: bool,
    dice_norm: bool,
    batch_dice: bool,
    excl_bg: bool,
) -> float:
    """自测用朴素双精度参考：严格按公式直接计算，无 smooth/clamp。

    调用方保证 logits 幅度温和、每类分母远离 0，双精度下无 0/0 风险——与
    主实现的稳定化路径互相独立，可作对拍基准。
    """
    b, c = logits_np.shape[:2]
    z = logits_np - logits_np.max(axis=1, keepdims=True)
    e = np.exp(z)
    u = e / e.sum(axis=1, keepdims=True)
    v = np.moveaxis(np.eye(c, dtype=np.float64)[target_np], -1, 1)

    logp = np.log(np.take_along_axis(u, target_np[:, None], axis=1)[:, 0])
    if apply_ce:
        wv = w_np[target_np]
        ce = -(wv * logp).sum() / wv.sum()
    else:
        ce = -logp.mean()

    if excl_bg:
        u, v, w_k = u[:, 1:], v[:, 1:], w_np[1:]
    else:
        w_k = w_np
    voxel_axes = tuple(range(2, u.ndim))
    axes = (0, *voxel_axes) if batch_dice else voxel_axes
    dc = 2.0 * (u * v).sum(axis=axes) / (u.sum(axis=axes) + v.sum(axis=axes))
    if apply_dice:
        wb = np.broadcast_to(w_k, dc.shape)
        if dice_norm:
            dice_loss = -(dc * wb).sum() / wb.sum()
        else:
            dice_loss = -(dc * wb).sum() / dc.size
    else:
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

    # ---- 1. 与朴素双精度公式对拍（2D/3D、batch_dice 两口径、apply 组合、归一化两口径；smooth=0 走严格公式路径） ----
    for shape in [(2, 4, 8, 8), (1, 3, 6, 5, 4)]:
        logits = torch.randn(*shape, dtype=torch.float64)
        target = torch.randint(0, shape[1], shape[:1] + shape[2:], dtype=torch.int64)
        w = torch.tensor([1.0, 3.0, 0.5, 2.0][: shape[1]], dtype=torch.float64)
        for batch_dice in (True, False):
            for excl_bg in (True, False):
                for ace, adi in [(True, True), (True, False), (False, True)]:
                    for dnorm in (True, False):
                        got = class_weighted_dice_ce_loss(
                            logits, target, class_weights=w,
                            batch_dice=batch_dice, exclude_background=excl_bg,
                            apply_to_ce=ace, apply_to_dice=adi,
                            dice_weight_norm=dnorm, smooth=0.0,
                        )
                        ref = _naive_weighted_ref(
                            logits.numpy(), target.numpy(), w.numpy(),
                            0.5, 0.5, ace, adi, dnorm, batch_dice, excl_bg,
                        )
                        assert torch.isfinite(got).all(), "损失出现 NaN/Inf"
                        err = abs(got.item() - ref) / (abs(ref) + 1e-12)
                        assert err < 1e-10, f"与朴素参考不一致：rel err={err:.2e}"
    print("PASS 与朴素双精度公式对拍（2D/3D、batch_dice 两口径、apply 3 组合 × 归一化 2 口径）")

    # ---- 2. 全 1 权重严格退化为未加权家族口径（smooth=0 走严格公式；构造 target 保证各类在场，朴素参考无 0/0） ----
    logits = torch.randn(2, 5, 6, 6, 4, dtype=torch.float64)
    target = torch.randint(0, 4, (2, 6, 6, 4), dtype=torch.int64)
    target[:, 0, 0, 0] = 4  # 保证第 5 类在场（随机 288 体素下其余类在场的概率极高）
    got1 = class_weighted_dice_ce_loss(logits, target, class_weights=None, smooth=0.0)
    got2 = class_weighted_dice_ce_loss(
        logits, target, class_weights=torch.ones(5, dtype=torch.float64), smooth=0.0
    )
    ref2 = _naive_weighted_ref(
        logits.numpy(), target.numpy(), np.ones(5), 0.5, 0.5,
        True, True, True, True, True,
    )
    assert abs(got1.item() - got2.item()) < 1e-14, "None 与全 1 权重不一致"
    assert abs(got1.item() - ref2) / (abs(ref2) + 1e-12) < 1e-10, "全 1 权重未退化到未加权口径"
    print(f"PASS None ≡ 全 1 权重 ≡ 未加权家族口径（loss={got1.item():.6f}）")

    # ---- 3. 加权方向性：坏预测类的权重放大 → 损失上升（CE 项与 Dice 项分别验证） ----
    # 构造：3 类，类 1 预测差（低概率）、类 2 预测好；背景占多数。
    t3 = torch.zeros(1, 16, 16, dtype=torch.int64)
    t3[0, 2:5, 2:5] = 1      # 类 1 小区域
    t3[0, 8:14, 8:14] = 2    # 类 2 较大区域
    onehot3 = F.one_hot(t3, num_classes=3).permute(0, -1, 1, 2).double()
    l3 = 4.0 * onehot3 - 1.0  # 温和偏向真值……再对类 1 通道压低制造坏预测
    l3[:, 1] -= 5.0
    for ace, adi, tag in [(True, False, "仅 CE"), (False, True, "仅 Dice"), (True, True, "CE+Dice")]:
        w_low = class_weighted_dice_ce_loss(
            l3, t3, class_weights=[1.0, 1.0, 1.0], apply_to_ce=ace, apply_to_dice=adi, batch_dice=False,
        )
        w_high = class_weighted_dice_ce_loss(
            l3, t3, class_weights=[1.0, 10.0, 1.0], apply_to_ce=ace, apply_to_dice=adi, batch_dice=False,
        )
        assert w_high.item() > w_low.item(), f"{tag}：坏预测类加权后损失应上升（{w_low.item()} → {w_high.item()}）"
    print("PASS 加权方向性：坏预测类权重放大使 CE/Dice/复合损失均上升")

    # ---- 4. 完美预测负极值（家族 -DC 口径无 +1 常数）；±30 极值 logits 触发钳制路径 ----
    perfect = 30.0 * onehot3 - 30.0 * (1.0 - onehot3)
    dl = class_weighted_dice_ce_loss(perfect, t3, class_weights=[1.0, 10.0, 1.0], weight_ce=0.0, weight_dice=1.0)
    cl = class_weighted_dice_ce_loss(perfect, t3, class_weights=[1.0, 10.0, 1.0], weight_ce=1.0, weight_dice=0.0)
    assert torch.isfinite(dl) and -1.0 - 1e-2 < dl.item() <= -1.0 + 1e-2, f"完美预测 Dice 项应≈-1: {dl.item()}"
    assert torch.isfinite(cl) and 0.0 <= cl.item() < 1e-2, f"完美预测 CE 项应≈0: {cl.item()}"
    print(f"PASS 完美预测：Dice 项={dl.item():.4f}（-DC 负极值 -1）、CE 项={cl.item():.2e}，极值 logits 无 NaN")

    # ---- 5. 缺席前景类有限性（GT 全背景：clamp + smooth 防空前景 0/0；smooth=0 仍有限） ----
    t_bg = torch.zeros(2, 5, 5, dtype=torch.int64)
    lb = class_weighted_dice_ce_loss(
        torch.randn(2, 4, 5, 5, dtype=torch.float64), t_bg, class_weights=[1.0, 7.0, 1.0, 1.0]
    )
    lb0 = class_weighted_dice_ce_loss(
        torch.randn(2, 4, 5, 5, dtype=torch.float64), t_bg, class_weights=[1.0, 7.0, 1.0, 1.0], smooth=0.0
    )
    assert torch.isfinite(lb) and torch.isfinite(lb0), "全背景 GT 下损失出现 NaN/Inf"
    print(f"PASS 缺席前景类损失有限：smooth=1e-5 → {lb.item():.4f}，smooth=0 → {lb0.item():.4f}")

    # ---- 6. 有限差分验证真导数（默认 smooth=1e-5 路径，2D 与 3D 各测若干坐标） ----
    for shape in [(2, 4, 5, 5), (1, 3, 4, 4, 4)]:
        x = torch.randn(*shape, dtype=torch.float64).detach().requires_grad_(True)
        t = torch.randint(0, shape[1], shape[:1] + shape[2:], dtype=torch.int64)
        w = torch.arange(1.0, shape[1] + 1.0)  # [1, 2, ...] 非平凡权重
        coords = [(0, 0, 0, 0), (0, 1, 2, 3), (1, 2, 4, 1), (1, 3, 0, 4)]
        coords = [c for c in coords if all(i < s for i, s in zip(c, x.shape))]
        if x.dim() == 5:
            coords = [c + (1,) for c in coords]
        num, ana = _finite_diff_grad(
            lambda z: class_weighted_dice_ce_loss(z, t, class_weights=w, batch_dice=False), x, coords
        )
        rel = np.abs(num - ana) / (np.abs(num) + np.abs(ana) + 1e-12)
        assert rel.max() < 1e-6, f"有限差分与 autograd 不一致（max rel err={rel.max():.2e}）"
    print("PASS 有限差分验证真导数（2D/3D，rel err < 1e-6）")

    # ---- 7. float32 forward+backward 梯度有限 ----
    logits = torch.randn(2, 4, 16, 16, requires_grad=True)
    target = torch.randint(0, 4, (2, 16, 16), dtype=torch.int64)
    loss = class_weighted_dice_ce_loss(logits, target, class_weights=[1.0, 5.0, 1.0, 1.0])
    loss.backward()
    assert torch.isfinite(loss).all() and torch.isfinite(logits.grad).all(), "float32 反向出现 NaN/Inf"
    assert logits.grad.abs().sum() > 0, "梯度全零"
    print(f"PASS float32 forward+backward 梯度有限（loss={loss.item():.6f}）")

    # ---- 8. 小卷积模型 forward+backward 并打印参数量（封装类 + ignore_index 冒烟） ----
    model = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(8, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 3, 1),
    )
    n_params = sum(p.numel() for p in model.parameters())
    x = torch.randn(1, 1, 32, 32)
    y = torch.randint(0, 3, (1, 32, 32), dtype=torch.int64)
    criterion = ClassWeightedDiceCELoss(class_weights=[1.0, 5.0, 1.0])
    loss = criterion(model(x), y)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters()), "模型梯度出现 NaN/Inf"
    y_ig = y.clone()
    y_ig[0, :4, :] = -100
    loss_ig = ClassWeightedDiceCELoss(class_weights=[1.0, 5.0, 1.0], weight_dice=0.0, ignore_index=-100)(
        model(x).detach().requires_grad_(True), y_ig
    )
    assert torch.isfinite(loss_ig), "ignore_index 路径出现 NaN/Inf"
    print(f"PASS 小模型 forward+backward（2D，参数量 {n_params}；loss={loss.item():.6f}；ignore_index+纯 CE 有限）")

    # ---- 9. 非法权重校验（长度错 / 非正值 / NaN 均抛 ValueError，不静默） ----
    for bad in [torch.ones(2), torch.tensor([1.0, 0.0, 1.0]), torch.tensor([1.0, float("nan"), 1.0])]:
        try:
            class_weighted_dice_ce_loss(torch.randn(1, 3, 4, 4), torch.zeros(1, 4, 4, dtype=torch.int64), class_weights=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"非法权重 {bad} 未抛 ValueError")
    print("PASS 非法权重（长度错/零值/NaN）均抛 ValueError")

    print("PASS 全部自测通过")
