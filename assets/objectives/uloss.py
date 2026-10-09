"""uloss — 不确定性/难度加权分割损失（PGU + REH，合成 uLoss）[loss]（资产卡：assets/objectives/uloss.md）

论文支撑（统一四问，依据 PMC 全文 PMC12691699（NIHMS2122123），2026-10-09 抓取全文 XML，
链接见下）：
- 哪篇论文用了它：Yanyan Zheng, Yabo Wu, Jie Chen, Xiaoguo Yang, Hao Zhang, Quanyong Yi,
  Jiantao Pu, Lei Wang（温州医科大学附属眼视光医院 / KCL / 匹兹堡大学等），"Uncertainty-
  and hardness-weighted loss functions for medical image segmentation"，Engineering
  Applications of Artificial Intelligence 163(Pt.3):113118, 2026（2025-11-10 在线），
  https://doi.org/10.1016/j.engappai.2025.113118 ，PMC 全文 https://www.ncbi.nlm.nih.gov/
  pmc/articles/PMC12691699/ 。论文自命名 CeW1/CeW2/DiceW1/DiceW2（代码仓库名 uhLoss；
  "uLoss" 一词不出现在论文正文）。
- 目的：医学影像分割中 CE/Dice 等损失对所有像素等权，忽略像素级预测不确定性（uncertainty）
  与难度（hardness）差异，误差集中在目标边界。论文把两种逐像素权重嵌入 CE 与 Dice：
  概率引导不确定性权重 PGU（低不确定像素保持、高不确定像素按 |y-p| 区分并降权）与区域
  增强难度权重 REH（直接放大 FP/FN 区域像素权重），在不大改训练管线的条件下提升边界精度。
- 效果（平均 DIS，即 Dice 相似度）：REFUGE 视盘/视杯 CeW1 平均 0.9223、最优组合
  CeW2+DiceW2 视盘 0.9733（Dice loss 平均 0.9154）；RETA 视网膜血管 CeW1 0.8639 vs
  CE 0.8337 / Focal 0.8044 / Dice 0.8476 / CE+Dice 0.8564；OCT 角膜层 CeW1 0.8974 vs
  CE 0.8693，且跨 Swin-Unet/U-Net++/UNeXt/U-KAN 一致占优；3D 左心房（ASC，V-Net）
  CeW1+DiceW1 0.9133 vs CE 0.8926。配对 t 检验 p<0.05 视为显著（REFUGE 上最优组合对
  CeW1/CeW2/DiceW1/hDice 达 p<0.0002）。
- 结论：PGU 配 CE（CeW1）、REH 配 Dice（DiceW）各自最有效，组合互补；跨 2D/3D、四数据集、
  四架构有效。局限（作者自述）：μ/η 最优值随任务变化大、调参难；充分优化后部分加权损失仅
  与原版相当；权重只用了像素级概率差，未用符号距离场等其他差异信息。

实现说明
- 关键公式与论文式子编号的对应（所读版本：PMC12691699 全文 XML，NIHMS2122123）：
  * 概览框架式(1)：L = Σf(y,p)·w / Σw = (1/C)Σ_c[Σ_i f(y_ic,p_ic)·w_ic / Σ_i w_ic]——
    注意这只是概览；实际两个加权损失用的是不同的权重嵌入策略（论文原话："tailored to
    the characteristics of logarithmic and fractional operations"），见式(5)(6)；
  * PGU 权重式(3)：d = |y_ic − p_ic|（概率差取绝对值，原文为竖线围栏），τ 为小阈值
    （论文实验 τ=0.005）：d≤τ（低不确定）时 w_ic = μ；d>τ 时 w_ic = μ·d + p_ic；
    式(2)是示性函数形式的简单不确定性权重（d≤τ 取 1，否则 0），是式(3)的前身；
  * REH 权重式(4)：w_ic = 1 + η·fp + fn = 1 + η·p_ic + y_ic − (η+1)·p_ic·y_ic，
    其中 fp = (1−y_ic)·p_ic、fn = (1−p_ic)·y_ic（像素级软计数；原文排版漏了括号，
    但式(4)右端恒等式唯一确定此读法），η 为放大系数（论文举例 2 或 3），
    w 取值范围 [1, η+2]；
  * 加权 CE 式(5)：CeW = −Σ_c Σ_i y_ic·log(p_ic)·w_ic——对类与像素的纯加权和，
    无任何归一化（MathML 中无分式结构，已核）；
  * 加权 Dice 式(6)：DiceW = 1 − (1/C)Σ_c [2Σ_i y·p·w] / [Σ_i y + p·w]——权重同时
    进分子（y·p·w）与分母的 p 侧（p·w），y 侧不加权；
  * 组合：CeW_w + DiceW_w 简单相加（Table 9 的 CeW1+DiceW1 等组合族，无组合系数）。
- 参考的开源实现：
  * 论文官方仓库 https://github.com/wmuLei/uhLoss ——截至 2026-10-09 仅含一句 README
    （"The PyTorch implementation will be available once the paper is accepted."），
    单分支单 commit、无 tag、无 fork（本地克隆 /tmp/mtskills-ref-uloss 已核），**无任何
    代码可对照**；本实现因此只以论文公式为唯一依据，实现内朴素参考实现按式(3)~(6)逐字
    手写（float64、无稳定项），作为正确性主证据；
  * 风格对齐本仓库 assets/objectives/tversky-loss.py（入口形状/布局/reduction 约定）。
- 与论文/官方实现不一致处（编号）：
  * (1) 式(5) 的 log(p) 用 F.log_softmax / F.logsigmoid 实现（数学恒等、数值稳定）；
    传入已是概率的 pred 时对 p 钳制下限 1e-12 再取 log（防用户越界输入的 log(0)）；
  * (2) 式(6) 分母加 eps=1e-12 兜底（论文无平滑项）：正常分母是"像素软计数和"，
    非空通道量级远大于 1e-12，eps 不改变数值；只防"目标与加权预测全空"的 0/0（此时
    该类比值取 0，对应式(6)字面继续参与 1/C 平均）；
  * (3) 式(5)(6)按单幅图像定义；批处理时本实现逐样本计算，默认对批取平均
    （reduction="mean"），"none" 返回逐样本严格按式(5)(6)的值；
  * (4) 权重是否 detach（对 p 的梯度是否截断）论文未写明；本实现按公式字面让梯度流经
    权重（与 focal loss 的调制因子同做法）；需要常数权重行为时可用 pgu_weight/reh_weight
    自行 .detach() 后加权；
  * (5) 官方代码缺失导致 detach 口径、批归一化细节无法对拍核对——上述选择均为论文公式
    字面 + 常规工程惯例，待官方代码发布后应回到本条目复核；
  * (6) C==1（sigmoid 单通道）入口的 CeW 按式(5)字面只含前景项 −y·log p（无背景项）；
    论文实验用 Swin-Unet/V-Net 逐目标 softmax 双通道（含背景通道的全 CE 口径）。多类/
    二类任务建议用 C≥2 的 one-hot 入口。
- μ=0 / η=0 不退化为普通 CE+Dice（易误解点，自测第 5 项验证）：μ=0 时 PGU 权重
  w = p·𝟙(d>τ)≠常数；η=0 时 REH 权重 w = 1+fn≠1（仍加权 FN）。真正退化为无权重的
  路径是 PGU 的 τ≥1（全部低不确定）且 μ=1 → w≡1 → CeW=普通 CE、DiceW=普通 soft-Dice
  （DIS 式(7)形式）。

布局与数值稳定
- 布局约定：2D=(B,C,H,W)，3D=(B,C,D,H,W)；pred 为未过 softmax/sigmoid 的 logits
  （from_logits=True，默认推荐）或已归一化概率（from_logits=False）；target 为同形
  one-hot (B,C,S...)（元素 0/1）或类别索引图 (B,1,S...)（整型，内部转 one-hot）。
- eps / 钳制清单：Dice 分母 eps=1e-12（兜底防 0/0，见不一致处(2)）；概率路径 log 前
  clamp_min(1e-12)（仅 from_logits=False 路径需要，1e-12 对应 |log|≈27.6，远小于
  训练中正常 CE 量级）；logit 路径 log_softmax/logsigmoid 内部稳定，无裸 log/exp；
  权重有界（PGU ∈ [μ·τ, μ+1]，REH ∈ [1, η+2]），μ/η/τ 非法值抛 ValueError。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["pgu_weight", "reh_weight", "cew_loss", "dicew_loss", "uloss", "ULoss"]

# 式(6) 分母兜底 eps：见文件头"不一致处(2)"
_DEFAULT_EPS = 1e-12
# 已是概率的输入取 log 前的下限钳制：见文件头"不一致处(1)"
_DEFAULT_LOG_EPS = 1e-12
# 论文实验超参默认：τ=0.005（全部任务）、μ=4（论文举例 3~4）、η=2（论文举例 2~3）
_DEFAULT_TAU = 0.005


def _check_params(weight_kind: str, mu: float, eta: float, tau: float) -> None:
    """校验超参合法性，非法即抛 ValueError（防静默 NaN/无意义权重）。"""
    if weight_kind not in ("pgu", "reh"):
        raise ValueError(f"weight_kind 只支持 'pgu'/'reh'，收到 {weight_kind!r}")
    if mu < 0:
        raise ValueError(f"mu 必须 >= 0，收到 {mu}")
    if eta < 0:
        raise ValueError(f"eta 必须 >= 0，收到 {eta}")
    if tau < 0:
        raise ValueError(f"tau 必须 >= 0（τ≥1 时 PGU 退化为常数权重 μ），收到 {tau}")


def _to_prob(pred: torch.Tensor, from_logits: bool) -> torch.Tensor:
    """把网络原始输出转成每通道概率。

    参数：
        pred: torch.Tensor，2D=(B,C,H,W) 或 3D=(B,C,D,H,W)。
        from_logits: True 表示 pred 是未过激活的 logit——C==1 时过 sigmoid，
            C>1 时过 softmax（通道维）；False 表示 pred 已是概率。
    返回：
        与 pred 同形状的概率张量，每通道值域 [0,1]。
    """
    if not from_logits:
        return pred
    if pred.shape[1] == 1:
        return torch.sigmoid(pred)
    return F.softmax(pred, dim=1)


def _onehot_target(target: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
    """把真值对齐到概率张量形状 (B,C,S...)。

    参数：
        target: 两种形式之一——
            (a) 与 prob 同形的 one-hot 标签 (B,C,S...)，元素 0/1；
            (b) 单通道类别索引图 (B,1,S...)（整型，值域 [0,C)），内部转 one-hot。
        prob: 已转成概率的预测张量 (B,C,S...)。
    返回：
        与 prob 同形状的浮点 one-hot 标签张量。
    """
    if target.shape == prob.shape:
        return target.to(dtype=prob.dtype)
    if target.dim() == prob.dim() and target.shape[1] == 1:
        idx = target[:, 0].long()
        if int(idx.min()) < 0 or int(idx.max()) >= prob.shape[1]:
            raise ValueError(
                f"类别索引超出 [0,{prob.shape[1]})：min={int(idx.min())}, max={int(idx.max())}"
            )
        oh = F.one_hot(idx, prob.shape[1])  # (B,S...,C)
        order = [0, oh.dim() - 1] + list(range(1, oh.dim() - 1))
        return oh.permute(*order).to(dtype=prob.dtype)
    raise ValueError(
        f"target 形状 {tuple(target.shape)} 既不是 one-hot {tuple(prob.shape)}，"
        f"也不是单通道类别索引图 {tuple((prob.shape[0], 1, *prob.shape[2:]))}"
    )


def pgu_weight(
    prob: torch.Tensor,
    target: torch.Tensor,
    mu: float = 4.0,
    tau: float = _DEFAULT_TAU,
) -> torch.Tensor:
    """PGU（probability-guided uncertainty）权重，论文式(3)。

    d = |y − p| ≤ τ（低不确定像素，含全部预测正确的像素）时 w = μ；d > τ 时
    w = μ·d + p。μ>1 时中等不确定像素权重低于 μ（论文设计意图：降低高不确定像素
    对其他像素的影响，同时用 μ 放大高不确定像素之间的区分度）。

    参数：
        prob: 概率张量 2D=(B,C,H,W) / 3D=(B,C,D,H,W)，值域 [0,1]。
        target: 与 prob 同形的 one-hot 标签（0/1）。
        mu: 低不确定像素的基准权重，默认 4.0（论文举例 3~4；各任务最优 1~16，
            见资产卡"调参经验"）。
        tau: 不确定性阈值，默认 0.005（论文全部任务统一取值；τ≥1 时整体退化为
            常数权重 μ）。

    返回：
        与 prob 同形的权重张量，值域 [μ·τ, μ+1]。
    """
    _check_params("pgu", mu, 0.0, tau)
    d = (target - prob).abs()
    return torch.where(d <= tau, torch.full_like(prob, float(mu)), mu * d + prob)


def reh_weight(
    prob: torch.Tensor,
    target: torch.Tensor,
    eta: float = 2.0,
) -> torch.Tensor:
    """REH（region-enhanced hardness）权重，论文式(4)。

    w = 1 + η·fp + fn，其中 fp = (1−y)·p、fn = (1−p)·y 为像素级假阳/假阴软计数
    （原文排版漏括号，由式(4)右端恒等式 1 + η·p + y − (η+1)·p·y 唯一确定）。
    易分类像素 w≈1，FP/FN 区域像素权重最高 η+2。注意 η 只作用于 FP 项——η=0 时
    仍有 fn 加权，不退化为无权重。

    参数：
        prob: 概率张量 2D=(B,C,H,W) / 3D=(B,C,D,H,W)，值域 [0,1]。
        target: 与 prob 同形的 one-hot 标签（0/1）。
        eta: FP 放大系数，默认 2.0（论文举例 2~3；各任务最优 1~8）。

    返回：
        与 prob 同形的权重张量，值域 [1, η+2]。
    """
    _check_params("reh", 0.0, eta, 0.0)
    fp = (1.0 - target) * prob
    fn = (1.0 - prob) * target
    return 1.0 + eta * fp + fn


def _make_weight(
    prob: torch.Tensor,
    gt: torch.Tensor,
    weight_kind: str,
    mu: float,
    eta: float,
    tau: float,
) -> torch.Tensor:
    """按 weight_kind 生成逐像素权重（式(3)或式(4)）。"""
    if weight_kind == "pgu":
        return pgu_weight(prob, gt, mu=mu, tau=tau)
    return reh_weight(prob, gt, eta=eta)


def _stable_log_prob(pred: torch.Tensor, prob: torch.Tensor, from_logits: bool) -> torch.Tensor:
    """计算 log(p)：logit 路径走 log_softmax/logsigmoid（稳定）；概率路径钳制下限。

    C==1 时式(5)字面只有前景项 −y·log p，用 logsigmoid 即可（见文件头不一致处(6)）。
    """
    if from_logits:
        if pred.shape[1] == 1:
            return F.logsigmoid(pred)
        return F.log_softmax(pred, dim=1)
    return torch.log(prob.clamp_min(_DEFAULT_LOG_EPS))


def _reduce(val: torch.Tensor, reduction: str) -> torch.Tensor:
    """对逐样本值 (B,) 做 reduction。"""
    if reduction == "mean":
        return val.mean()
    if reduction == "sum":
        return val.sum()
    if reduction == "none":
        return val
    raise ValueError(f"reduction 只支持 mean/sum/none，收到 {reduction!r}")


def cew_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight_kind: str = "pgu",
    mu: float = 4.0,
    eta: float = 2.0,
    tau: float = _DEFAULT_TAU,
    from_logits: bool = False,
    reduction: str = "mean",
) -> torch.Tensor:
    """不确定性/难度加权交叉熵损失 CeW，论文式(5)。

    CeW = −Σ_c Σ_i y_ic·log(p_ic)·w_ic（对类与像素的纯加权和，论文无归一化——
    量级随图像像素数与 μ 线性增长，与 DiceW 组合时由学习率吸收，见资产卡"坑"）。
    权重 w 按 weight_kind 取 PGU（式(3)）或 REH（式(4)）。

    参数：
        pred: 2D=(B,C,H,W) 或 3D=(B,C,D,H,W)。C==1 为单通道前景 logit/概率
            （式(5)字面仅含前景项）；C>1 为多类 logit/概率。
        target: 与 pred 空间形状匹配的真值——同形 one-hot (B,C,...) 或单通道
            类别索引 (B,1,...)（整型）。
        weight_kind: "pgu"（式(3)，默认）/ "reh"（式(4)）。
        mu / eta / tau: 各权重超参，默认 4.0 / 2.0 / 0.005（论文设置）。
        from_logits: pred 是否为 logit（C==1 走 sigmoid，C>1 走 softmax）。
        reduction: "mean"（对批平均，默认）/ "sum"（对批求和）/ "none"
            （返回逐样本 (B,) 值——严格按式(5)）。

    返回：
        torch.Tensor 标量（mean/sum）或 (B,)（none）；值 >= 0
        （y·log p ≤ 0、w > 0），可为大数（无归一化，见上）。

    异常：
        ValueError：超参非法、target 形状不匹配或 reduction 非法。
    """
    _check_params(weight_kind, mu, eta, tau)
    prob = _to_prob(pred, from_logits)
    gt = _onehot_target(target, prob)
    w = _make_weight(prob, gt, weight_kind, mu, eta, tau)
    logp = _stable_log_prob(pred, prob, from_logits)
    per_sample = -(gt * logp * w).sum(dim=tuple(range(1, prob.dim())))  # (B,)
    return _reduce(per_sample, reduction)


def dicew_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight_kind: str = "pgu",
    mu: float = 4.0,
    eta: float = 2.0,
    tau: float = _DEFAULT_TAU,
    from_logits: bool = False,
    eps: float = _DEFAULT_EPS,
    reduction: str = "mean",
) -> torch.Tensor:
    """不确定性/难度加权 Dice 损失 DiceW，论文式(6)。

    DiceW = 1 − (1/C)Σ_c [2Σ_i y·p·w] / [Σ_i (y + p·w)]：权重进分子与分母的 p 侧，
    y 侧不加权（论文"针对分数运算特性"的嵌入方式）。注意 w≠1 时即使完美预测
    DiceW 也不为 0（PGU：1 − 2μ/(1+μ)，μ>1 时为负——式(6)固有行为，非 bug）。

    参数：
        pred / target / weight_kind / mu / eta / tau / from_logits: 同 cew_loss。
        eps: 分母兜底系数，默认 1e-12（见文件头不一致处(2)）。
        reduction: "mean"（默认）/ "sum" / "none"。

    返回：
        torch.Tensor 标量（mean/sum）或 (B,)（none）；单类比值 ∈ [0, 1+ε)，均值后
        可能 ≤ 0（见上）；目标与预测全空的通道该类比值按 0 计（eps 兜底）。

    异常：
        ValueError：超参非法、target 形状不匹配或 reduction 非法。
    """
    _check_params(weight_kind, mu, eta, tau)
    prob = _to_prob(pred, from_logits)
    gt = _onehot_target(target, prob)
    w = _make_weight(prob, gt, weight_kind, mu, eta, tau)
    axes = tuple(range(2, prob.dim()))
    num = 2.0 * (gt * prob * w).sum(dim=axes)                  # (B,C)
    den = (gt + prob * w).sum(dim=axes).clamp_min(eps)         # (B,C)
    ratio = num / den                                          # (B,C)
    per_sample = 1.0 - ratio.mean(dim=1)                       # (B,)，式(6)的 1/C 平均
    return _reduce(per_sample, reduction)


def uloss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight_kind: str = "pgu",
    mu: float = 4.0,
    eta: float = 2.0,
    tau: float = _DEFAULT_TAU,
    from_logits: bool = False,
    eps: float = _DEFAULT_EPS,
    reduction: str = "mean",
) -> torch.Tensor:
    """uLoss：加权 CE + 加权 Dice 简单相加（论文 Table 9 组合族，无组合系数）。

    即 uloss = cew_loss(...) + dicew_loss(...)（同一 weight_kind 与超参）。
    论文结论：PGU 配 CE 最有效、REH 配 Dice 最有效，组合互补；具体组合
    CeW1+DiceW1 / CeW2+DiceW2 在四数据集上均稳健。

    参数与返回值语义同 cew_loss / dicew_loss。

    异常：
        ValueError：同上。
    """
    return cew_loss(
        pred, target, weight_kind=weight_kind, mu=mu, eta=eta, tau=tau,
        from_logits=from_logits, reduction=reduction,
    ) + dicew_loss(
        pred, target, weight_kind=weight_kind, mu=mu, eta=eta, tau=tau,
        from_logits=from_logits, eps=eps, reduction=reduction,
    )


class ULoss(nn.Module):
    """uLoss 的 nn.Module 封装，参数同 uloss。

    参数：
        weight_kind: "pgu" / "reh"，默认 "pgu"。
        mu: PGU 基准权重（式(3)），默认 4.0。
        eta: REH 放大系数（式(4)），默认 2.0。
        tau: PGU 不确定性阈值，默认 0.005。
        from_logits: forward 收到的 pred 是否为 logit，默认 False。
        eps: Dice 分母兜底系数，默认 1e-12。
        reduction: "mean" / "sum" / "none"，默认 "mean"。
    """

    def __init__(
        self,
        weight_kind: str = "pgu",
        mu: float = 4.0,
        eta: float = 2.0,
        tau: float = _DEFAULT_TAU,
        from_logits: bool = False,
        eps: float = _DEFAULT_EPS,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.weight_kind = weight_kind
        self.mu = mu
        self.eta = eta
        self.tau = tau
        self.from_logits = from_logits
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算 uLoss，参数语义同 uloss。"""
        return uloss(
            pred,
            target,
            weight_kind=self.weight_kind,
            mu=self.mu,
            eta=self.eta,
            tau=self.tau,
            from_logits=self.from_logits,
            eps=self.eps,
            reduction=self.reduction,
        )


# ---------------------------------------------------------------------------
# 朴素双精度参考实现：严格按论文式(3)~(6)手写（float64、无稳定项），
# 独立于主实现的稳定化路径，作为正确性对拍基准。
# ---------------------------------------------------------------------------
def _naive_uloss_numpy(
    logits: np.ndarray,
    gt_onehot: np.ndarray,
    weight_kind: str,
    mu: float,
    eta: float,
    tau: float,
) -> tuple[float, float, float]:
    """按式(3)~(6)逐字计算单样本 CeW/DiceW/uLoss。

    参数：
        logits: numpy.ndarray，(C,S...)（单样本，无批维）logits。
        gt_onehot: numpy.ndarray，与 logits 同形 one-hot（0/1）。
        weight_kind / mu / eta / tau: 同主实现。
    返回：
        (cew, dicew, uloss) 三元组 float。
    """
    x = np.asarray(logits, dtype=np.float64)
    y = np.asarray(gt_onehot, dtype=np.float64)
    if x.shape[0] == 1:
        p = 1.0 / (1.0 + np.exp(-x))  # sigmoid
    else:
        e = np.exp(x - x.max(axis=0, keepdims=True))  # 减最大值防溢出
        p = e / e.sum(axis=0, keepdims=True)
    if weight_kind == "pgu":
        d = np.abs(y - p)
        w = np.where(d <= tau, mu, mu * d + p)  # 式(3)
    else:
        w = 1.0 + eta * (1.0 - y) * p + (1.0 - p) * y  # 式(4)
    cew = float(-(y * np.log(p) * w).sum())  # 式(5)（对拍用中等 logits，p 不为 0）
    axes = tuple(range(1, x.ndim))  # 无批维布局 (C,S...)：空间轴从 1 起
    num = 2.0 * (y * p * w).sum(axis=axes)
    den = (y + p * w).sum(axis=axes)
    dicew = float(1.0 - (num / den).mean())  # 式(6)
    return cew, dicew, cew + dicew


# ---------------------------------------------------------------------------
# 自测：CPU 可跑、固定随机种子、数秒内完成
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    # 1) 权重函数与手写公式一致 + 取值范围（PGU ∈ [μτ, μ+1]，REH ∈ [1, η+2]）
    prob = torch.rand(2, 3, 16, 16, dtype=torch.float64)
    gt = (torch.rand(2, 3, 16, 16, dtype=torch.float64) > 0.7).to(torch.float64)
    wp = pgu_weight(prob, gt, mu=4.0, tau=0.005)
    wr = reh_weight(prob, gt, eta=2.0)
    d_np = np.abs(gt.numpy() - prob.numpy())
    wp_ref = np.where(d_np <= 0.005, 4.0, 4.0 * d_np + prob.numpy())
    wr_ref = 1.0 + 2.0 * (1 - gt.numpy()) * prob.numpy() + (1 - prob.numpy()) * gt.numpy()
    assert np.allclose(wp.numpy(), wp_ref, atol=1e-12)
    assert np.allclose(wr.numpy(), wr_ref, atol=1e-12)
    assert wp.min() >= 4.0 * 0.005 - 1e-12 and wp.max() <= 5.0 + 1e-12
    assert wr.min() >= 1.0 - 1e-12 and wr.max() <= 4.0 + 1e-12
    print("PASS 权重函数公式对拍与取值范围（PGU、REH）")

    # 2) 朴素双精度参考实现对拍：2D C=2/C=4 与 3D、二类/多类、多组 μ/η，逐样本一致
    rng = np.random.default_rng(42)
    for (b, c, size) in [(2, 2, (16, 16)), (2, 4, (12, 12)), (1, 2, (6, 10, 10))]:
        logits64 = torch.from_numpy(rng.normal(0, 1.5, size=(b, c, *size))).to(torch.float64)
        idx = torch.from_numpy(rng.integers(0, c, size=(b, 1, *size)))
        oh = F.one_hot(idx[:, 0], c)                       # (B,*size,C)
        order = [0, oh.dim() - 1] + list(range(1, oh.dim() - 1))
        onehot = oh.permute(*order).to(torch.float64)      # (B,C,*size)
        for kind in ("pgu", "reh"):
            for mu, eta, tau in [(4.0, 2.0, 0.005), (16.0, 8.0, 0.005), (1.0, 1.0, 0.005)]:
                got = uloss(logits64, onehot, weight_kind=kind, mu=mu, eta=eta, tau=tau,
                            from_logits=True, reduction="none").numpy()
                cew_got = cew_loss(logits64, onehot, weight_kind=kind, mu=mu, eta=eta, tau=tau,
                                   from_logits=True, reduction="none").numpy()
                dice_got = dicew_loss(logits64, onehot, weight_kind=kind, mu=mu, eta=eta, tau=tau,
                                      from_logits=True, reduction="none").numpy()
                for i in range(b):
                    cew_r, dice_r, u_r = _naive_uloss_numpy(
                        logits64[i].numpy(), onehot[i].numpy(), kind, mu, eta, tau)
                    assert abs(cew_got[i] - cew_r) < 1e-9, (kind, mu, eta, cew_got[i], cew_r)
                    assert abs(dice_got[i] - dice_r) < 1e-9, (kind, mu, eta, dice_got[i], dice_r)
                    assert abs(got[i] - u_r) < 1e-9, (kind, mu, eta, got[i], u_r)
                    assert abs(got[i] - (cew_got[i] + dice_got[i])) < 1e-12
    print("PASS 朴素双精度式(3)~(6)对拍：2D C=2/C=4、3D、pgu/reh × μ∈{1,4,16} × η∈{1,2,8} 逐样本一致（max 偏差 < 1e-9）")

    # 3) C==1 sigmoid 入口对拍（式(5)字面仅前景项）
    logits1 = torch.from_numpy(rng.normal(0, 1.0, size=(2, 1, 12, 12))).to(torch.float64)
    gt1 = (torch.from_numpy(rng.random((2, 1, 12, 12))) > 0.7).to(torch.float64)
    for kind in ("pgu", "reh"):
        got1 = uloss(logits1, gt1, weight_kind=kind, from_logits=True, reduction="none").numpy()
        for i in range(2):
            _, _, u_r = _naive_uloss_numpy(logits1[i].numpy(), gt1[i].numpy(), kind, 4.0, 2.0, 0.005)
            assert abs(got1[i] - u_r) < 1e-9
    print("PASS C==1 sigmoid 入口对拍（式(5)仅前景项口径）")

    # 4) forward+backward：损失与梯度有限（2D 多类 + 3D + 极端 logits ±30）
    logits32 = torch.randn(2, 3, 24, 24, requires_grad=True)
    tgt_idx = torch.randint(0, 3, (2, 1, 24, 24))
    loss32 = ULoss("pgu", mu=4.0, from_logits=True)(logits32, tgt_idx)
    loss32.backward()
    assert torch.isfinite(loss32) and torch.isfinite(logits32.grad).all()
    logits3d = torch.randn(1, 2, 6, 12, 12, requires_grad=True)
    tgt3d = torch.randint(0, 2, (1, 1, 6, 12, 12))
    loss3d = ULoss("reh", eta=2.0, from_logits=True)(logits3d, tgt3d)
    loss3d.backward()
    assert torch.isfinite(loss3d) and torch.isfinite(logits3d.grad).all()
    extreme = torch.randn(2, 2, 8, 8, requires_grad=True)
    with torch.no_grad():
        extreme.copy_(torch.where(extreme > 0, 30.0, -30.0))
    loss_ext = ULoss("pgu", from_logits=True)(extreme, torch.randint(0, 2, (2, 1, 8, 8)))
    loss_ext.backward()
    assert torch.isfinite(loss_ext) and torch.isfinite(extreme.grad).all()
    print(f"PASS forward+backward 梯度有限：2D loss={loss32.item():.4f}，3D loss={loss3d.item():.4f}，极端 logits loss={loss_ext.item():.4f}")

    # 5) 退化路径：PGU τ=1（全低不确定）且 μ=1 → w≡1 → 严格等于普通 CE + 普通 soft-Dice；
    #    同时验证 μ=0 / η=0 不等于无权重（论文公式无此退化，易误解点）
    lg5 = torch.randn(2, 3, 16, 16, dtype=torch.float64)
    idx5 = torch.randint(0, 3, (2, 1, 16, 16))
    onehot3 = F.one_hot(idx5[:, 0], 3).permute(0, 3, 1, 2).to(torch.float64)  # (2,3,16,16)
    plain_ce = -(onehot3 * F.log_softmax(lg5, dim=1)).sum(dim=tuple(range(1, lg5.dim())))
    plain_dice_ps = 1.0 - ((2.0 * onehot3 * F.softmax(lg5, dim=1)).sum(dim=(2, 3))
                           / (onehot3 + F.softmax(lg5, dim=1)).sum(dim=(2, 3))).mean(dim=1)
    deg_ce = cew_loss(lg5, onehot3, weight_kind="pgu", mu=1.0, tau=1.0,
                      from_logits=True, reduction="none")
    deg_dice = dicew_loss(lg5, onehot3, weight_kind="pgu", mu=1.0, tau=1.0,
                          from_logits=True, reduction="none")
    assert torch.allclose(deg_ce, plain_ce, atol=1e-9), (deg_ce, plain_ce)
    assert torch.allclose(deg_dice, plain_dice_ps, atol=1e-9), (deg_dice, plain_dice_ps)
    w_mu0 = pgu_weight(torch.tensor([0.5]), torch.tensor([1.0]), mu=0.0, tau=0.005)
    w_eta0 = reh_weight(torch.tensor([0.5]), torch.tensor([1.0]), eta=0.0)
    assert abs(w_mu0.item() - 0.5) < 1e-12   # μ=0: w = p = 0.5 ≠ 1
    assert abs(w_eta0.item() - 1.5) < 1e-12  # η=0: w = 1 + fn = 1.5 ≠ 1
    print(f"PASS 退化等价：τ=1&μ=1 时 CeW/DiceW ≡ 普通 CE/soft-Dice（max|Δ|="
          f"{max((deg_ce-plain_ce).abs().max().item(), (deg_dice-plain_dice_ps).abs().max().item()):.1e}）；"
          f"μ=0→w={w_mu0.item():.2f}、η=0→w={w_eta0.item():.2f}，均 ≠1（无此退化）")

    # 6) 方向性：REH 注入大 FP 区块后损失增量应大于无权重 CE+Dice（放大难样本）；
    #    PGU 对中等不确定像素降权（w = μd+p < μ），对易像素保持 μ（论文设计意图）
    torch.manual_seed(7)
    gt_bin = (torch.rand(2, 1, 32, 32) > 0.6).to(torch.float64)
    gt2 = torch.cat([1.0 - gt_bin, gt_bin], dim=1)  # (2,2,32,32) one-hot
    lg = torch.randn(2, 2, 32, 32, dtype=torch.float64) * 1.5
    lg_fp = lg.clone()
    fg_mask = gt_bin.squeeze(1) < 0.5
    blob = torch.zeros(2, 32, 32, dtype=torch.bool)
    blob[:, 20:28, 20:28] = True
    inject = blob & fg_mask  # 只往真值背景处注入 FP
    lg_fp[:, 0][inject] = -4.0   # 背景通道压低
    lg_fp[:, 1][inject] = 4.0    # 前景通道抬高 → 大 FP 区块
    u_reh = uloss(lg, gt2, "reh", eta=2.0, from_logits=True)
    u_reh_fp = uloss(lg_fp, gt2, "reh", eta=2.0, from_logits=True)
    u_base = uloss(lg, gt2, "pgu", mu=1.0, tau=1.0, from_logits=True)      # = 普通 CE+Dice
    u_base_fp = uloss(lg_fp, gt2, "pgu", mu=1.0, tau=1.0, from_logits=True)
    assert (u_reh_fp - u_reh) > (u_base_fp - u_base), (u_reh, u_reh_fp, u_base, u_base_fp)
    w_easy = pgu_weight(torch.tensor([1.0]), torch.tensor([1.0]), mu=4.0).item()   # d=0 → μ
    w_mid = pgu_weight(torch.tensor([0.5]), torch.tensor([1.0]), mu=4.0).item()    # 4*0.5+0.5=2.5
    assert abs(w_easy - 4.0) < 1e-12 and abs(w_mid - 2.5) < 1e-12 and w_mid < w_easy
    print(f"PASS 方向性：注入 FP 区块后 uLoss(REH) 增量 {u_reh_fp-u_reh:.4f} > 无权重增量 {u_base_fp-u_base:.4f}；"
          f"PGU 降权方向：中等不确定 w={w_mid:.2f} < 易像素 w={w_easy:.2f}")

    # 7) 边界与退化输入：全零目标有限；完美预测闭式（REH→0；PGU→CeW=0、DiceW=(1-μ)/(1+μ)）；
    #    reduction 三态形状
    empty = torch.zeros(2, 2, 8, 8, dtype=torch.float64)
    assert torch.isfinite(uloss(torch.randn(2, 2, 8, 8, dtype=torch.float64), empty, "reh", from_logits=False))
    perfect = gt2
    u_reh_p = uloss(perfect, perfect, "reh", from_logits=False)
    assert abs(u_reh_p.item()) < 1e-9, u_reh_p
    u_pgu_p = uloss(perfect, perfect, "pgu", mu=4.0, from_logits=False)
    assert abs(u_pgu_p.item() - (1.0 - 4.0) / (1.0 + 4.0)) < 1e-9, (u_pgu_p, (1-4)/(1+4))
    vals = uloss(lg, gt2, "reh", from_logits=True, reduction="none")
    assert vals.shape == (2,)
    assert abs(uloss(lg, gt2, "reh", from_logits=True, reduction="mean").item() - vals.mean().item()) < 1e-12
    assert abs(uloss(lg, gt2, "reh", from_logits=True, reduction="sum").item() - vals.sum().item()) < 1e-12
    print(f"PASS 边界退化：全零目标有限；完美预测 REH→{u_reh_p.item():.1e}≈0、"
          f"PGU→{u_pgu_p.item():.4f}=(1-μ)/(1+μ)（式(6)固有负值）；reduction 三态形状正确")

    # 8) 小模型端到端（2D/3D，from_logits=True，类别索引 target），打印参数量
    net2d = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 4, 1))
    x2d = torch.randn(2, 3, 32, 32)
    y2d = torch.randint(0, 4, (2, 1, 32, 32))
    loss8a = ULoss("pgu", mu=4.0, from_logits=True)(net2d(x2d), y2d)
    loss8a.backward()
    assert torch.isfinite(loss8a) and all(torch.isfinite(p.grad).all() for p in net2d.parameters() if p.grad is not None)
    net3d = nn.Sequential(nn.Conv3d(1, 4, 3, padding=1), nn.ReLU(), nn.Conv3d(4, 2, 1))
    x3d = torch.randn(1, 1, 8, 16, 16)
    y3d = (torch.rand(1, 1, 8, 16, 16) > 0.8).long()
    loss8b = ULoss("reh", eta=2.0, from_logits=True)(net3d(x3d), y3d)
    loss8b.backward()
    assert torch.isfinite(loss8b) and all(torch.isfinite(p.grad).all() for p in net3d.parameters() if p.grad is not None)
    print(f"PASS 2D 小模型端到端：参数量 {sum(q.numel() for q in net2d.parameters())}，loss={loss8a.item():.4f}")
    print(f"PASS 3D 小模型端到端：参数量 {sum(q.numel() for q in net3d.parameters())}，loss={loss8b.item():.4f}")

    print("PASS 全部自测通过")
