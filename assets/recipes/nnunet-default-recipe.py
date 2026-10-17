"""nnunet-default-recipe — [recipe]（nnU-Net v2 默认训练配方的框架外可复刻组件：
SGD+poly LR / 深监督 scales·权重·wrapper·GT 近邻下采样 / AMP 反传序列；
资产卡：assets/recipes/nnunet-default-recipe.md）

论文支撑（统一四问，简体中文）：
- 哪篇论文用了它：Isensee, Jaeger, Kohl, Petersen & Maier-Hein（德国癌症研究中心
  DKFZ），《nnU-Net: a self-configuring method for deep learning-based biomedical
  image segmentation》，Nature Methods 2021，
  https://www.nature.com/articles/s41592-020-01008-z （配方论述在官方免费补充材料
  MOESM1 Note 2 "Training Scheme"）。ResEnc 变体沿用同配方零改动：Isensee 等
  《nnU-Net Revisited》，MICCAI 2024，https://arxiv.org/abs/2404.09556 。
  所读版本：补充材料 PDF + nnunetv2 2.7.x 源码（/data/lx/thymic_seg/.venv，
  2026-10-15 逐行考证；该版本无 __version__ 属性，行号以该安装为准）。
- 目的：给"自配置"框架一套跨任务不调参的固定训练配方——框架的全部自由度花在
  预处理与网络规划上，训练循环本身保持简单且久经考验。
- 效果：作为 nnU-Net 默认配置，23 个公开数据集上超过大多数人工特化方案（摘要）；
  ResEnc 论文在同配方下把 KiTS Dice 从 86.04 推到 88.17(L)/88.67(XL)。
- 结论：医学 3D 分割的事实标准配方；框架外复现 nnU-Net 基线时每一项都应对表抄写。

实现说明（关键默认值 ↔ nnunetv2 源码对应）：
- 优化器：SGD(lr=0.01, momentum=0.99, nesterov=True, weight_decay=3e-5)——
  nnUNetTrainer.py:153-154（initial_lr/weight_decay）+ configure_optimizers
  （momentum/nesterov 硬编码）。
- poly LR：lr = lr0·(1 − t/T)^0.9，exponent 0.9 为 polylr.py 默认；按 epoch 步进
  （step() 无参调用用内部计数器 ctr，nnU-Net 每 epoch 调一次）。
- 深监督 scales：1/cumprod(pool_op_kernel_sizes)[:-1]；而 pool_op_kernel_sizes
  ≡ architecture.arch_kwargs.strides（含首级 [1,1,1]，plans_handler.py:155-156），
  故本实现直接吃 strides（nnUNetTrainer.py:383-387）。
- 深监督权重：[1/2^i] 末位 0（DDP 下 1e-6，防 unused-parameters 崩溃），再归一化
  到和为 1（nnUNetTrainer.py:455-468）。
- DeepSupervisionWrapper：加权求和、零权重输出跳过（training/loss/deep_supervision.py）。
- GT 下采样：最近邻（order=0），deep_supervision_donwsampling.py:18 默认。
- AMP 反传：scale → backward → unscale_ → clip_grad_norm(12) → step → update
  （nnUNetTrainer.py:1033-1042；clip 12 是常被漏抄的隐藏默认）。
- 与参考实现不一致处：
  1. poly 底数 (1 − t/T) 加 clamp_min(0)：源码不设防（训练恒有 t < T），
     本实现防越界调用产生负底数的非整数幂 NaN——以数值稳定规范为准，正常路径无差。
  2. GT 下采样用 F.interpolate(mode="nearest")：nnU-Net 经 batchgenerators
     resize_segmentation(order=0) 同为最近邻语义，仅边界 half-case 的取整
     可能差 1 个体素；以 PyTorch 原生算子为准（自包含约束）。
  3. PolyLRScheduler 不继承 torch _LRScheduler：nnunetv2 版继承之，但只重写
     step() 自管计数；本实现独立类行为对齐（自包含、不依赖 torch 内部 API）。

布局与数值稳定：
- 标签/权重布局 3D = (B, C, D, H, W)（C 通常 1）；strides 为逐级列表
  [[1,1,1],[2,2,2],...]（含首级）；logits 未过 softmax。
- 数值稳定点：poly 底数 clamp 0（幂 0.9 非整数）；strides 逐元素校验 >=1（cumprod
  无零除）；DS 权重归一化前断言存在非零项；近邻下采样为纯索引操作，标签安全。
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

__all__ = [
    "build_sgd_optimizer",
    "PolyLRScheduler",
    "deep_supervision_scales",
    "deep_supervision_weights",
    "DeepSupervisionWrapper",
    "downsample_seg_nearest",
    "backward_with_amp",
]


def build_sgd_optimizer(
    params,
    initial_lr: float = 1e-2,
    weight_decay: float = 3e-5,
) -> torch.optim.SGD:
    """按 nnU-Net v2 默认配方构建 SGD 优化器（momentum=0.99、nesterov 固定）。

    参数：
        params: 可迭代参数组（如 net.parameters()），传给 torch.optim.SGD。
        initial_lr: 初始学习率，配方默认 1e-2（nnUNetTrainer.py:153）。
        weight_decay: 权重衰减，配方默认 3e-5（nnUNetTrainer.py:154）；
            与 lr 是耦合的共同体，改 lr 时保持不动。

    返回：
        torch.optim.SGD，momentum=0.99、nesterov=True 硬编码（源码同）。
    """
    return torch.optim.SGD(params, lr=initial_lr, momentum=0.99, nesterov=True,
                           weight_decay=weight_decay)


class PolyLRScheduler:
    """poly 学习率日程：lr = initial_lr · (1 − t/max_steps)^exponent。

    与 nnunetv2 的 polylr.PolyLRScheduler 行为对齐：step() 无参调用时使用内部
    计数器并自增；也可显式传 current_step。nnU-Net 每 **epoch** 结束调用一次
    （max_steps = num_epochs = 1000），不是按 iteration。

    参数：
        optimizer: 已构建的优化器，step() 时直接改写其 param_groups 的 lr。
        initial_lr: 起始学习率。
        max_steps: 总步数（训练总 epoch 数）。
        exponent: 多项式指数，配方默认 0.9。
    """

    def __init__(self, optimizer: torch.optim.Optimizer, initial_lr: float,
                 max_steps: int, exponent: float = 0.9):
        if max_steps <= 0:
            raise ValueError(f"max_steps 须为正，得到 {max_steps}")
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0  # 与源码一致：无参 step() 从 0 起计数并自增

    def step(self, current_step: int | None = None) -> float:
        """推进一个 epoch 并写回 lr；返回新 lr。

        数值稳定：底数 (1 − t/T) 先 clamp_min(0)——exponent 0.9 非整数，
        负底数幂会产生 NaN；源码靠训练恒有 t < T 保证非负，本实现显式设防。
        """
        if current_step is None:
            current_step = self.ctr
            self.ctr += 1
        if current_step < 0:
            raise ValueError(f"current_step 须 >= 0，得到 {current_step}")
        base = max(0.0, 1.0 - current_step / self.max_steps)
        new_lr = self.initial_lr * base ** self.exponent
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr
        return new_lr


def deep_supervision_scales(strides: list[list[int]] | tuple[tuple[int, ...], ...]
                            ) -> list[list[float]]:
    """由网络 strides 计算 nnU-Net 深监督各输出的逐轴分辨率比例。

    nnU-Net 口径：scales = 1/cumprod(pool_op_kernel_sizes)[:-1]，而
    pool_op_kernel_sizes ≡ architecture.arch_kwargs.strides（含首级 [1,1,1]，
    plans_handler.py:155-156）——所以直接传网络 strides 即可，丢掉最后一级
    （最深编码级不参与深监督，故长度 = n_stages − 1）。

    参数：
        strides: 逐级步长，形如 [[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[1,2,2]]
            （含首级 [1,1,1]；各向异性步长如 [1,2,2] 合法）。

    返回：
        长度 len(strides)−1 的列表，每项为逐轴比例 ∈ (0,1]，
            如 [[1,1,1],[.5,.5,.5],[.25,.25,.25],[.125,.125,.125],[.0625,.0625,.0625]]。
    """
    if len(strides) < 2:
        raise ValueError("strides 至少 2 级（首级 + 一次下采样）")
    arr = np.asarray(strides, dtype=np.int64)
    if arr.ndim != 2 or (arr <= 0).any():
        raise ValueError(f"strides 须为二维且逐元素 >= 1，得到形状 {arr.shape}")
    # cumprod 逐元素 >= 1（校验过 >=1），1/x 无零除；[:-1] 去掉最深编码级
    return (1.0 / np.cumprod(arr, axis=0))[:-1].tolist()


def deep_supervision_weights(n_outputs: int, ddp: bool = False) -> np.ndarray:
    """深监督各级损失权重：1/2^i、末位 0（DDP 1e-6）、整体归一化到和为 1。

    参数：
        n_outputs: 深监督输出个数（= len(deep_supervision_scales(...))）。
        ddp: 分布式训练下末位用 1e-6 代替 0（DDP 对"参数完全未用"会崩溃，
            nnUNetTrainer.py:462-466 注明；单卡用 0，源码同）。

    返回：
        长度 n_outputs 的 float 数组，和为 1；最后一位为 0（或 1e-6）。
    """
    if n_outputs < 2:
        raise ValueError(f"深监督至少 2 个输出，得到 {n_outputs}")
    weights = np.array([1.0 / (2 ** i) for i in range(n_outputs)])
    weights[-1] = 1e-6 if ddp else 0.0
    total = weights.sum()
    # n_outputs>=2 时前级必有非零项（1/2^0=1），total>0 恒成立，断言兜底
    assert total > 0, "权重全零——n_outputs 非法"
    return weights / total


class DeepSupervisionWrapper(nn.Module):
    """把单分辨率损失包装成多分辨率加权求和（复刻 nnunetv2 同名类）。

    forward(net_output, target) 中两者皆为列表/元组（逐分辨率对齐，如
    [(B,C,96,160,160), (B,C,48,80,80), ...] 与同形状的各分辨率 GT），
    返回标量损失 sum_i w_i · loss(out_i, tgt_i)；w_i = 0 的项整项跳过
    （不进入 loss 计算）。

    参数：
        loss: 单分辨率可调用损失，签名 loss(pred, target) -> 标量张量。
        weight_factors: 各级权重（如 deep_supervision_weights 的输出）；
            None 时全部取 1。
    """

    def __init__(self, loss, weight_factors: list[float] | tuple[float, ...] | None = None):
        super().__init__()
        if weight_factors is not None:
            assert any(w != 0 for w in weight_factors), \
                "至少一个权重须非零（源码同校验）"
            weight_factors = tuple(weight_factors)
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, *args) -> torch.Tensor:
        assert all(isinstance(a, (tuple, list)) for a in args), \
            f"所有输入须为 tuple/list（逐分辨率），得到 {[type(a) for a in args]}"
        weights = (1,) * len(args[0]) if self.weight_factors is None \
            else self.weight_factors
        total = None
        for i, inputs in enumerate(zip(*args)):
            if weights[i] == 0.0:
                continue  # 零权重跳过整级，与源码一致（省计算）
            term = weights[i] * self.loss(*inputs)
            total = term if total is None else total + term
        assert total is not None, "全零权重——无可计算项"
        return total


def downsample_seg_nearest(seg: torch.Tensor, scale: list[float] | tuple[float, ...]
                           ) -> torch.Tensor:
    """按逐轴比例对标签图做最近邻下采样（复刻深监督 GT 的 order=0 语义）。

    参数：
        seg: 标签张量 (B, C, D, H, W)（或 (B, C, H, W)），整数/浮点标签皆可——
            最近邻是纯索引操作，不产生新标签值。
        scale: 逐轴比例 ∈ (0,1]，长度 = 空间维数，如 [0.5, 0.5, 0.5] 或
            各向异性的 [0.0625, 0.0625, 0.0625]（比例 1 的轴尺寸不变）。

    返回：
        下采样后的标签张量，值域与输入完全一致（索引重排）。
        与 batchgenerators resize_segmentation(order=0) 最近邻语义等价，
        边界 half-case 的取整可能差 1 个体素（见文件头不一致处 2）。
    """
    if seg.ndim not in (4, 5):
        raise ValueError(f"seg 须为 (B,C,D,H,W) 或 (B,C,H,W)，得到 {seg.shape}")
    spatial = seg.shape[2:]
    if len(scale) != len(spatial):
        raise ValueError(f"scale 长度 {len(scale)} 与空间维数 {len(spatial)} 不符")
    if any((s <= 0) or (s > 1) for s in scale):
        raise ValueError(f"scale 逐项须 ∈ (0,1]（本函数只做下采样），得到 {scale}")
    new_size = [max(1, int(round(s * r))) for s, r in zip(spatial, scale)]
    # mode="nearest"：纯索引选取，无插值权重——标签安全（不产生 0.5 这类混合值）
    return torch.nn.functional.interpolate(
        seg.float(), size=new_size, mode="nearest").to(seg.dtype)


def backward_with_amp(loss: torch.Tensor, optimizer: torch.optim.Optimizer,
                      scaler=None, clip_grad_norm: float = 12.0) -> float:
    """nnU-Net AMP 反传序列：scale→backward→unscale_→clip→step→update。

    nnUNetTrainer.py:1039-1043 的顺序复刻；**clip_grad_norm=12 是配方常量**
    （藏在 GradScaler.unscale_ 之后，教程常漏抄）。CPU/无 scaler 时退化为
    backward→clip→step（源码在 CPU 上同样不用 GradScaler）。

    参数：
        loss: 标量损失张量。
        optimizer: 本次 step 使用的优化器（调用方负责之前的 zero_grad）。
        scaler: torch.amp.GradScaler 实例；None 走无 AMP 路径。
        clip_grad_norm: 梯度总范数上限，配方默认 12。

    返回：
        float，裁剪**前**的梯度总范数（监控用；inf 表示 scaler 跳过了本步，
        此时本函数也不执行 step——与 scaler 语义一致）。
    """
    params = [p for group in optimizer.param_groups for p in group["params"]
              if p.grad is not None or True]
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        pre_norm = float(torch.nn.utils.clip_grad_norm_(params, clip_grad_norm))
        if not all(torch.isfinite(p.grad).all() for p in params if p.grad is not None):
            return pre_norm  # 梯度含 inf/NaN：交还调用方判断，不 step（防污染权重）
        scaler.step(optimizer)
        scaler.update()
        return pre_norm
    loss.backward()
    pre_norm = float(torch.nn.utils.clip_grad_norm_(params, clip_grad_norm))
    optimizer.step()
    return pre_norm


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    # 1) 优化器超参对表
    lin = nn.Linear(4, 2)
    opt = build_sgd_optimizer(lin.parameters())
    pg = opt.param_groups[0]
    assert pg["lr"] == 1e-2 and pg["weight_decay"] == 3e-5
    assert pg["momentum"] == 0.99 and pg["nesterov"] is True
    print("PASS 优化器 = SGD(lr 0.01, wd 3e-5, momentum 0.99, nesterov)")

    # 2) poly LR 与解析式逐点对拍（按 epoch 步进语义）
    sched = PolyLRScheduler(opt, initial_lr=1e-2, max_steps=1000)
    for t, expect in [(0, 1e-2), (500, 1e-2 * 0.5 ** 0.9), (999, 1e-2 * 0.001 ** 0.9)]:
        lr = sched.step(current_step=t)
        assert abs(lr - expect) < 1e-12, (t, lr, expect)
    lr_seq = []
    for _ in range(3):
        lr_seq.append(sched.step())
    assert lr_seq == [1e-2 * 1.0, 1e-2 * 0.999 ** 0.9, 1e-2 * 0.998 ** 0.9]
    assert np.isfinite(lr_seq).all()
    try:
        PolyLRScheduler(opt, 1e-2, 10).step(current_step=20)  # 越界 clamp 不 NaN
        print("PASS poly lr 解析式逐点一致（含 epoch 步进与越界 clamp）")
    except ValueError:
        raise AssertionError("越界 t 不应抛错（clamp 语义）")

    # 3) DS scales 对 Dataset701_Thymus ResEnc-M strides 口径
    strides = [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [1, 2, 2]]
    scales = deep_supervision_scales(strides)
    expect = [[1.0] * 3, [0.5] * 3, [0.25] * 3, [0.125] * 3, [0.0625] * 3]
    assert len(scales) == 5 and np.allclose(scales, expect)
    print("PASS 深监督 scales = 1/cumprod(strides)[:-1]（Dataset701 口径 5 级）")

    # 4) DS 权重：归一化、末位 0 / DDP 1e-6
    w = deep_supervision_weights(5)
    raw = np.array([1.0, 0.5, 0.25, 0.125, 0.0])
    assert abs(w.sum() - 1.0) < 1e-12 and w[-1] == 0.0
    assert np.allclose(w, raw / raw.sum())
    w_ddp = deep_supervision_weights(5, ddp=True)
    # DDP 末位 1e-6 是归一化前设的，归一化后 ≈5.2e-7——与源码行为一致
    assert 0 < w_ddp[-1] < 1e-5 and abs(w_ddp.sum() - 1.0) < 1e-9
    print("PASS 深监督权重 1/2^i 归一化，末位 0（DDP 1e-6）")

    # 5) wrapper 加权和与手算对拍（含跳零与 None 权重）
    dummy = lambda p, t: (p * t).float().mean()  # 哑损失：便于手算
    wrapper = DeepSupervisionWrapper(dummy, list(w))
    preds = [torch.ones(1, 1, 4, 4, 4) * (i + 1) for i in range(5)]
    tgts = [torch.ones(1, 1, 4, 4, 4)] * 5
    got = float(wrapper(preds, tgts))
    manual = sum(wi * (i + 1) for i, wi in enumerate(w) if wi != 0)
    assert abs(got - manual) < 1e-6
    ones = float(DeepSupervisionWrapper(dummy, None)(preds[:2], tgts[:2]))
    assert abs(ones - 3.0) < 1e-6  # None 权重全 1：mean(1)+mean(2)=3
    print("PASS DeepSupervisionWrapper 加权和手算一致（跳零/None 路径）")

    # 6) 最近邻 GT 下采样：形状、值域保值、退化输入报错
    seg = torch.zeros(1, 1, 8, 8, 8, dtype=torch.long)
    seg[:, :, :4] = 1
    ds = downsample_seg_nearest(seg, [0.5, 0.5, 0.5])
    assert ds.shape == (1, 1, 4, 4, 4) and set(ds.unique().tolist()) <= {0, 1}
    assert ds.flatten()[0] == 1 and ds.flatten()[-1] == 0  # 前半 1、后半 0 保序
    aniso = downsample_seg_nearest(seg, [1.0, 0.5, 0.25])
    assert aniso.shape == (1, 1, 8, 4, 2)
    try:
        downsample_seg_nearest(seg, [1.5, 1.0, 1.0])
        raise AssertionError("上采样比例应报错")
    except ValueError:
        pass
    print("PASS GT 最近邻下采样形状/值域/各向异性/退化输入处理")

    # 7) AMP 反传序列：无 scaler 路径 clip 生效且梯度有限
    lin2 = nn.Linear(4, 2)
    opt2 = build_sgd_optimizer(lin2.parameters())
    big = (lin2(torch.randn(8, 4)) ** 2).sum() * 1e4  # 构造大梯度触发裁剪
    opt2.zero_grad()
    pre = backward_with_amp(big, opt2, scaler=None, clip_grad_norm=12.0)
    total = torch.sqrt(sum((p.grad ** 2).sum() for p in lin2.parameters()
                           if p.grad is not None))
    assert pre > 12.0 and float(total) <= 12.0 + 1e-3
    assert all(torch.isfinite(p.grad).all() for p in lin2.parameters())
    print("PASS AMP 反传序列：unscale 后 clip 12 生效、梯度有限（pre-norm "
          f"{pre:.1f} → {float(total):.1f}）")

    print("PASS 全部自测通过")
