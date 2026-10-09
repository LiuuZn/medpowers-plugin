"""unified-focal-loss —— 统一 Focal 损失（对称 UFL 与非对称 A-UFL）[loss]（资产卡：assets/objectives/unified-focal-loss.md）

论文支撑（统一四问；所读版本：arXiv v4 全文 LaTeX 源（2021-11-24 提交）+ PMC 已出版全文
PMC8785124，两版公式排版一致）：
- 哪篇论文用了它：Michael Yeung, Evis Sala, Carola-Bibiane Schönlieb, Leonardo Rundo，
  "Unified Focal loss: Generalising Dice and cross entropy-based losses to handle class
  imbalanced medical image segmentation"，Computerized Medical Imaging and Graphics
  95:102026, 2022，https://doi.org/10.1016/j.compmedimag.2021.102026（预印本
  arXiv:2102.04525，PMC 全文 PMC8785124）。
- 目的：医学影像分割中病灶常远小于背景（输出不平衡），单个 batch 内前景体素占比低
  （输入不平衡）；CE 族损失只对输入不平衡稳健、Dice 族只对输出不平衡稳健。论文把两族
  损失统一到一个分层框架：δ 统一控制前景/背景权重（输出不平衡）、γ 统一控制 focal
  难例聚焦（输入不平衡）、λ 控制两分量配比，把原先六个超参压缩到三个。
- 效果：5 个公开不平衡数据集、对比 6 种 Dice/CE 族损失，非对称变体 DSC 全面第一：
  CVC-ClinicDB 0.909±0.023、DRIVE 0.803±0.006、BUS2017 0.824±0.063（2D 二类）、
  BraTS20 0.787±0.049（3D 二类，vs Focal Tversky 0.747±0.052）、KiTS19 肿瘤
  0.634±0.079（3D 多类，vs Dice 0.536±0.074、Combo 0.554±0.081）、肾脏 0.943±0.011。
- 结论：此前所有 Dice/CE 族损失都是本框架的特例（γ=0、δ=0.5 时 λ=1 得 CE、λ=0 得
  Dice）；实践中固定 λ=0.5、δ=0.6、只调 γ（2D 在 [0.1,0.9] 网格搜、3D 取 0.5）即可，
  γ=0.9 最差、0.5 最稳；非对称变体整体略优于对称。边界：未含边界类（boundary-based）
  损失对比、未在 nnU-Net 级强基线上验证、多类共享单个 γ 非最优（作者自述）。

实现说明
- 关键公式与论文式子编号的对应（所读版本：arXiv v4 LaTeX 全文，2021-11-24；PMC8785124
  为已出版同文，公式编号按期刊排版 §2.4）：
  * 修正 Focal 损失（式(16)）：L_mF = w_c·(1-p_t)^γ·(-log p_t)，逐类权重 w：前景类 δ、
    背景类 1-δ——focal 指数取 γ，γ∈[0,1]，γ=0 退化为加权 CE；
  * 修正 Tversky 指数（式(18)）与修正 Focal Tversky 损失（式(17)）：mTI_c = TP_c /
    (TP_c + δ·FN_c + (1-δ)·FP_c)，L_mFT = mean_c (1-mTI_c)^(1-γ)——focal 指数取 1-γ，
    γ=0 退化为加权 Tversky（δ=0.5 时即 Dice）；
  * 对称 UFL（式(19)）：L_sUF = λ·L_mF + (1-λ)·L_mFT；
  * 非对称 A-UFL（式(20)-(22)）：L_maF 去掉稀有类的 focal 调制（前景项只乘 δ）、
    L_maFT 去掉背景的 focal 调制（背景项取 1-mTI），L_aUF = λ·L_maF + (1-λ)·L_maFT。
- 参考的开源实现：
  * 官方 mlyg/unified-focal-loss（loss_functions.py，main 分支，Keras/TF，仅二类）：
    symmetric/asymmetric_focal_loss 用 (1-p)^γ 乘逐类 CE、δ 乘前景、1-δ 乘背景；
    symmetric/asymmetric_focal_tversky_loss 用 (1-TI)·(1-TI)^(-γ) = (1-TI)^(1-γ)；
    ε=K.epsilon()=1e-7 同时加在 TI 分子与分母；概率输入先 clip 到 [ε,1-ε] 再取 log。
    本地落点：/tmp/official_loss_functions.py（自 raw.githubusercontent.com 抓取）。
  * MONAI（Project-MONAI/MONAI，monai/losses/unified_focal_loss.py，dev 分支；本地落点
    /tmp/monai_unified_focal_loss.py）：AsymmetricUnifiedFocalLoss（torch，仅二类），
    λ 权重加在 Focal 分量上（weight·FL + (1-weight)·FTL），与论文式(19)一致；但其
    AsymmetricFocalLoss 对堆叠的两类逐体素张量直接取 mean（类维也被平均），FL 分量
    数值恰为官方 Keras/论文口径的一半（见"不一致处"第 8 条）。
  * 本仓库 assets/objectives/tversky-loss.py：mTI 是其 Tversky 指数的 δ 加权推广
    （两个条目的平滑口径不同，见"不一致处"第 5 条）。
- 与参考实现/论文不一致处（逐条编号，以论文正文与官方代码为准）：
  1. 论文排版式(16)(17)把两个 focal 指数印反了（mF 印成 (1-p)^(1-γ)、mFT 印成
     (1-mTI)^γ），与论文正文自己的特例声明（"γ=0、δ=0.5 时 λ=1 得 CE、λ=0 得 Dice"）
     矛盾；官方两份代码（mlyg、MONAI）均按 mF 指数 γ、mFT 指数 1-γ 实现。本实现从
     官方代码与论文正文特例声明，以正文为准。
  2. λ 的分配：论文式(19)(22)与 MONAI 为 λ·FL + (1-λ)·FTL；官方 Keras 仓库的
     sym/asym_unified_focal_loss 为 weight·FTL + (1-weight)·FL（两者在默认 λ=0.5 时
     数值相同）。本实现从论文/MONAI；与官方 Keras 仓库对拍时需以 weight=1-λ 代入。
  3. 官方两份实现都只支持二类（Keras 版逐类索引硬编码 [...,0]/[...,1]，实际仅 2D 正确；
     MONAI 文档写明 binary-only）。本实现按论文 Σ_c 形式推广到多类：通道 0 为背景、
     其余通道为前景；对称变体对全部 C 个类的 (1-mTI_c)^(1-γ) 取均值，非对称变体背景类
     不做 focal、全部前景类视为稀有类——论文 KiTS19 实验即把肾脏与肿瘤都按稀有类处理、
     共享单个 γ=0.5，本推广与其一致。
  4. 输入默认概率（与官方一致；clip 后取 log）；另提供 from_logits=True 的稳定入口
     （C==1 过 log_softmax([-x,x])、C>1 过 log_softmax，CE 项不再裸 log），属本实现的
     有意扩展；logits 路径下 log_prob 来自 log_softmax，但 focal/TI 使用的概率同样
     clip 到 [ε,1-ε]——极端 logit 饱和到 p=1 时 (1-p)^γ 的反向 -γ(1-p)^(γ-1) 会发散，
     clip 后底数严格为正，与官方概率口径行为一致。
  5. 平滑口径：官方在 mTI 分子、分母各加 ε=1e-7（K.epsilon()），本实现沿用；本仓库
     tversky-loss.py 只在分母加 eps——两卡口径不同，各自注明，属官方对齐取舍。
  6. FL 聚合 = 论文的 1/N 归一化：逐体素对类求和后对 (B,空间) 取均值；FTL 对 (B,C)
     取均值——与官方 Keras/MONAI 的 mean 语义逐算子一致。
  7. 官方 FTL 写成 (1-TI)·(1-TI)^(-γ)，本实现直接算 (1-TI)^(1-γ)：代数恒等，且避免
     完美预测（1-TI→0）时负指数上溢为 inf 的数值风险，属纯数值等价改写。
  8. MONAI 的 AsymmetricFocalLoss 把逐体素的两类损失堆叠后直接 mean（类维参与平均），
     其 FL 分量 = 官方 Keras/论文 sum-over-classes 口径的一半；本实现从官方/论文的
     1/N 归一化。等价对拍（/tmp/compare_unified_focal.py）：与官方 Keras 转写 51 组
     fp64 全部一致（max|Δ|≤1.1e-16）；MONAI 在还原该口径后 39 组也全部一致（max|Δ|=0），
     字面差异（最大 1.5e-1）完全由此聚合口径解释。
- 数值稳定（ε 清单，ε=1e-7，与官方 K.epsilon() 一致）：
  * 概率输入路径：p clip 到 [ε,1-ε] 再取 log——fp32 下 sigmoid/softmax 可精确饱和到
    0/1，不 clip 则 -log(p) 发散；ε=1e-7 远小于正常概率量级、只影响饱和样本；
  * mTI 分母加 ε 兜底、分子随官方加同值 ε：防全空类（TP=FN=FP=0）0/0；体素软计数
    量级 ≥1，1e-7 不改变非退化样本的损失值；
  * 幂运算：(1-p)^γ 底数 ∈ [ε,1-ε]；(1-mTI)^(1-γ) 底数 ∈ [0,1]（分母 ≥ 分子恒成立），
    均无需额外钳制；γ=0 时不走 pow 直接退化为加权 CE/Tversky，避免 0^0 歧义
    （torch 约定 x^0=1，与论文极限一致，见自测）。

布局约定：2D=(B,C,H,W)、3D=(B,C,D,H,W)；B 批、C 通道——C==1 为二类前景通道（内部自动
扩展为 [背景,前景] 两通道，与官方 Keras 二类形式一致），C≥2 时通道 0 为背景（softmax
多类）。from_logits=False 时 pred 为概率（默认，官方口径），True 时为未过激活的 logit。
真值 (B,1,S...) 二值掩码/整型类别索引或 (B,C,S...) one-hot。输出为标量损失（内部已
平均，官方无 reduction 参数）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "unified_focal_loss",
    "asymmetric_unified_focal_loss",
    "UnifiedFocalLoss",
    "AsymmetricUnifiedFocalLoss",
]

# ε：与官方 mlyg/unified-focal-loss 的 K.epsilon() 一致；用途见文件头"数值稳定"
_DEFAULT_EPS = 1e-7


def _check_hparams(gamma: float, delta: float, weight: float) -> None:
    """校验超参取值范围：γ≥0、δ∈[0,1]、λ∈[0,1]（论文定义域）。"""
    if gamma < 0.0:
        raise ValueError(f"gamma 应 ≥0（论文定义域 [0,1]），收到 {gamma}")
    if not 0.0 <= delta <= 1.0:
        raise ValueError(f"delta 应 ∈[0,1]，收到 {delta}")
    if not 0.0 <= weight <= 1.0:
        raise ValueError(f"weight(λ) 应 ∈[0,1]，收到 {weight}")


def _prepare(
    pred: torch.Tensor,
    target: torch.Tensor,
    from_logits: bool,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """对齐预测与真值到统一的 (B,C*,S...) 两通道/多通道形式。

    返回：
        prob: 概率张量（已在 [eps,1-eps] 内或来自 softmax/sigmoid）。
        log_prob: 对应的对数概率（概率路径经 clip 后取 log；logits 路径来自
            log_softmax，天然稳定）。
        y: 与 prob 同形的浮点 one-hot/掩码张量（二类时为 [1-y, y] 两通道）。
    """
    if pred.dim() not in (4, 5):
        raise ValueError(f"pred 应为 4D (B,C,H,W) 或 5D (B,C,D,H,W)，收到 {tuple(pred.shape)}")
    if target.dim() != pred.dim():
        raise ValueError(
            f"target 维度 {target.dim()} 与 pred 维度 {pred.dim()} 不一致；"
            "支持 (B,1,S...) 掩码/类别索引或 (B,C,S...) one-hot"
        )
    c = pred.shape[1]
    dtype = pred.dtype

    # --- 真值对齐 ---
    if c > 1 and target.shape[1] == 1 and target.shape != pred.shape:
        if target.dtype.is_floating_point and not torch.all((target == 0) | (target == 1)):
            raise ValueError(
                "多类（C>1）时单通道 target 应为整型类别索引张量，或取值仅为 0/1 的"
                "二值掩码；若为软标签请直接传 (B,C,S...) one-hot"
            )
        oh = F.one_hot(target[:, 0].long(), c)  # (B,S...,C)
        y = oh.permute(0, oh.dim() - 1, *range(1, oh.dim() - 1)).to(dtype)
    elif target.shape == pred.shape:
        y = target.to(dtype)
        if c == 1:
            y = torch.cat([1.0 - y, y], dim=1)  # 二类：扩展为 [背景,前景]
    else:
        raise ValueError(
            f"target 形状 {tuple(target.shape)} 与 pred {tuple(pred.shape)} 不匹配；"
            "支持同形 one-hot 或单通道 (B,1,S...) 形式"
        )

    # --- 概率与对数概率 ---
    if from_logits:
        # C==1：[sigmoid(-x), sigmoid(x)] ≡ log_softmax([-x,x])，与官方二类两通道一致
        log_prob = F.log_softmax(torch.cat([-pred, pred], dim=1) if c == 1 else pred, dim=1)
        # 概率同样 clip 到 [ε,1-ε]：极端 logit 下 sigmoid/softmax 会饱和到 0/1，
        # (1-p)^γ 在 p→1 的反向 -γ(1-p)^(γ-1) 发散，clip 后底数严格为正（见文件头）
        prob = log_prob.exp().clamp(min=eps, max=1.0 - eps)
    else:
        # 官方口径：概率先 clip 到 [ε,1-ε]（取值依据见文件头"数值稳定"）
        p = pred.clamp(min=eps, max=1.0 - eps)
        if c == 1:
            p = torch.cat([1.0 - p, p], dim=1)
        prob = p
        log_prob = p.clamp(min=eps).log()
    return prob, log_prob, y


def _modified_focal_loss(
    prob: torch.Tensor,
    log_prob: torch.Tensor,
    y: torch.Tensor,
    gamma: float,
    delta: float,
    asymmetric: bool,
) -> torch.Tensor:
    """修正（非对称）Focal 损失：论文式(16)/(20)，返回标量。

    逐类权重 w：通道 0（背景）1-δ、其余通道 δ；focal 调制 (1-p)^γ——对称变体作用于
    全部类，非对称变体仅作用于背景（稀有类去 focal，式(20)）。逐体素对类求和（y 为
    指示使非真值类为 0）再全均值，即论文的 1/N 归一化。
    """
    c = prob.shape[1]
    w = torch.full((c,), float(delta), device=prob.device, dtype=prob.dtype)
    w[0] = 1.0 - delta
    w = w.view(1, c, *([1] * (prob.dim() - 2)))

    ce = -y * log_prob
    if gamma != 0.0:  # γ=0：无 focal 调制，直接加权 CE（避免 0^0 歧义）
        factor = (1.0 - prob).pow(gamma)
        if asymmetric:  # 前景类去 focal：仅背景保留 (1-p)^γ
            factor = torch.cat([factor[:, :1], torch.ones_like(factor[:, 1:])], dim=1)
        ce = ce * factor
    return (w * ce).sum(dim=1).mean()


def _modified_focal_tversky_loss(
    prob: torch.Tensor,
    y: torch.Tensor,
    gamma: float,
    delta: float,
    asymmetric: bool,
    eps: float,
) -> torch.Tensor:
    """修正（非对称）Focal Tversky 损失：论文式(17)(18)/(21)，返回标量。

    逐 (样本,通道) 计算修正 Tversky 指数 mTI = (TP+ε)/(TP+δ·FN+(1-δ)·FP+ε)（ε 口径
    与官方一致，见文件头），对称变体取 mean_c (1-mTI_c)^(1-γ)；非对称变体背景类不做
    focal（取 1-mTI_0）、前景类取 (1-mTI_c)^(1-γ)（式(21)）。底数 ∈[0,1]（分母≥分子），
    幂运算安全；γ=1 时底数 0 处按 torch 约定 x^0=1（论文极限，分量退化为常数 1）。
    """
    axes = list(range(2, prob.dim()))
    tp = (prob * y).sum(axes)
    fn = (y * (1.0 - prob)).sum(axes)
    fp = ((1.0 - y) * prob).sum(axes)
    ti = (tp + eps) / (tp + delta * fn + (1.0 - delta) * fp + eps)
    base = 1.0 - ti
    if asymmetric:
        loss_c = torch.cat([base[:, :1], base[:, 1:].pow(1.0 - gamma)], dim=1)
    else:
        loss_c = base.pow(1.0 - gamma)
    return loss_c.mean()


def unified_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 0.5,
    delta: float = 0.6,
    weight: float = 0.5,
    from_logits: bool = False,
    eps: float = _DEFAULT_EPS,
) -> torch.Tensor:
    """对称 Unified Focal loss（论文式(19)）：L = λ·L_mF + (1-λ)·L_mFT。

    γ=0、δ=0.5 时：λ=1 严格退化为（0.5 加权的）交叉熵、λ=0 严格退化为各类平均的
    soft Dice loss（论文特例关系，自测第 6/7 项验证）。

    参数：
        pred: 2D=(B,C,H,W) 或 3D=(B,C,D,H,W)。C==1 为二类前景概率/logit；C≥2 为
            多类概率/logit（通道 0 为背景）。from_logits=False（默认）时为概率
            （官方口径），True 时为未过激活的 logit。
        target: 与 pred 空间形状匹配的真值——同形 one-hot (B,C,...)、二类掩码
            (B,1,...)（0/1，仅 C==1）或整型类别索引 (B,1,...)（仅 C≥1 的多类）。
        gamma: focal 参数 ∈[0,1]（论文推荐 2D 网格搜 [0.1,0.9]、3D 取 0.5；只调此参）。
            控制 mF 的难例聚焦强度与 mFT 的难例增强强度（指数 1-γ）。
        delta: 前景类权重 ∈[0,1]，背景取 1-δ（论文推荐 0.6；亦为 mTI 中 FN 的权重，
            (1-δ) 为 FP 的权重）。
        weight: 分量权重 λ∈[0,1]（论文推荐固定 0.5；λ 乘 Focal 分量、(1-λ) 乘
            Focal Tversky 分量，与论文式(19)/MONAI 一致）。
        from_logits: pred 是否为 logit（见文件头"不一致处"第 4 条）。
        eps: 数值稳定系数，默认 1e-7（与官方 K.epsilon() 一致，依据见文件头）。

    返回：
        torch.Tensor 标量损失（≥0，内部已对批/类/空间平均）。

    异常：
        ValueError：形状不匹配、target 形式非法或超参越界。
    """
    _check_hparams(gamma, delta, weight)
    prob, log_prob, y = _prepare(pred, target, from_logits, eps)
    fl = _modified_focal_loss(prob, log_prob, y, gamma, delta, asymmetric=False)
    ftl = _modified_focal_tversky_loss(prob, y, gamma, delta, asymmetric=False, eps=eps)
    return weight * fl + (1.0 - weight) * ftl


def asymmetric_unified_focal_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 0.5,
    delta: float = 0.6,
    weight: float = 0.5,
    from_logits: bool = False,
    eps: float = _DEFAULT_EPS,
) -> torch.Tensor:
    """非对称 Unified Focal loss（论文式(20)-(22)）：L = λ·L_maF + (1-λ)·L_maFT。

    与对称变体的差别：Focal 分量只对背景做 focal 抑制（稀有/前景类去 focal，避免
    压低本就稀少的前景信号）；Focal Tversky 分量只对前景做 focal 增强（背景不做，
    避免增强海量易分背景）。多类时全部前景通道视为稀有类（论文 KiTS19 实验口径）。
    论文中非对称变体在全部 5 个数据集上 DSC 第一。

    参数与返回值同 unified_focal_loss。

    异常：
        ValueError：形状不匹配、target 形式非法或超参越界。
    """
    _check_hparams(gamma, delta, weight)
    prob, log_prob, y = _prepare(pred, target, from_logits, eps)
    fl = _modified_focal_loss(prob, log_prob, y, gamma, delta, asymmetric=True)
    ftl = _modified_focal_tversky_loss(prob, y, gamma, delta, asymmetric=True, eps=eps)
    return weight * fl + (1.0 - weight) * ftl


class UnifiedFocalLoss(nn.Module):
    """对称 Unified Focal loss 的 nn.Module 封装，参数同 unified_focal_loss。

    参数：
        gamma: focal 参数 ∈[0,1]，默认 0.5（论文 3D 推荐/2D 网格中点）。
        delta: 前景类权重 ∈[0,1]，默认 0.6（论文推荐）。
        weight: 分量权重 λ∈[0,1]，默认 0.5（论文推荐，固定后只调 γ）。
        from_logits: forward 收到的 pred 是否为 logit，默认 False（概率）。
        eps: 数值稳定系数，默认 1e-7。
    """

    def __init__(
        self,
        gamma: float = 0.5,
        delta: float = 0.6,
        weight: float = 0.5,
        from_logits: bool = False,
        eps: float = _DEFAULT_EPS,
    ) -> None:
        super().__init__()
        _check_hparams(gamma, delta, weight)
        self.gamma = gamma
        self.delta = delta
        self.weight = weight
        self.from_logits = from_logits
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算对称 Unified Focal loss，参数语义同 unified_focal_loss。"""
        return unified_focal_loss(
            pred,
            target,
            gamma=self.gamma,
            delta=self.delta,
            weight=self.weight,
            from_logits=self.from_logits,
            eps=self.eps,
        )


class AsymmetricUnifiedFocalLoss(nn.Module):
    """非对称 Unified Focal loss 的 nn.Module 封装，参数同 asymmetric_unified_focal_loss。

    参数：
        gamma: focal 参数 ∈[0,1]，默认 0.5（论文推荐 3D 直接取 0.5）。
        delta: 前景类权重 ∈[0,1]，默认 0.6（论文推荐；MONAI 默认 0.7 与论文不同）。
        weight: 分量权重 λ∈[0,1]，默认 0.5（论文推荐）。
        from_logits: forward 收到的 pred 是否为 logit，默认 False（概率）。
        eps: 数值稳定系数，默认 1e-7。
    """

    def __init__(
        self,
        gamma: float = 0.5,
        delta: float = 0.6,
        weight: float = 0.5,
        from_logits: bool = False,
        eps: float = _DEFAULT_EPS,
    ) -> None:
        super().__init__()
        _check_hparams(gamma, delta, weight)
        self.gamma = gamma
        self.delta = delta
        self.weight = weight
        self.from_logits = from_logits
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算非对称 Unified Focal loss，参数语义同 asymmetric_unified_focal_loss。"""
        return asymmetric_unified_focal_loss(
            pred,
            target,
            gamma=self.gamma,
            delta=self.delta,
            weight=self.weight,
            from_logits=self.from_logits,
            eps=self.eps,
        )


# ---------------------------------------------------------------------------
# 朴素双精度参考实现：严格按论文式(16)-(22)手写（指数采用与官方代码/论文正文特例
# 声明一致的约定：mF 指数 γ、mFT 指数 1-γ），不含 clip/eps/γ 捷径等任何稳定化路径，
# 独立于上面的主实现，仅用于自测对拍。
# ---------------------------------------------------------------------------
def _naive_unified_focal_np(
    prob: np.ndarray,
    y: np.ndarray,
    gamma: float,
    delta: float,
    lam: float,
    asymmetric: bool,
) -> float:
    """numpy 朴素版（float64）：prob/y 均为 (B,C,S...) 形式，返回标量。"""
    ce = -y * np.log(prob)
    w = np.full(prob.shape[1], delta, dtype=np.float64)
    w[0] = 1.0 - delta
    w = w.reshape(1, -1, *([1] * (prob.ndim - 2)))
    if gamma != 0.0:
        factor = (1.0 - prob) ** gamma
        if asymmetric:
            factor = np.concatenate([factor[:, :1], np.ones_like(factor[:, 1:])], axis=1)
        ce = ce * factor
    mf = float((w * ce).sum(axis=1).mean())

    axes = tuple(range(2, prob.ndim))
    tp = (prob * y).sum(axes)
    fn = (y * (1.0 - prob)).sum(axes)
    fp = ((1.0 - y) * prob).sum(axes)
    ti = tp / (tp + delta * fn + (1.0 - delta) * fp)  # 无 eps（论文原式）
    base = 1.0 - ti
    if asymmetric:
        base = np.concatenate([base[:, :1], base[:, 1:] ** (1.0 - gamma)], axis=1)
    else:
        base = base ** (1.0 - gamma)
    ftl = float(base.mean())
    return lam * mf + (1.0 - lam) * ftl


def _naive_unified_focal_torch(
    prob: torch.Tensor,
    y: torch.Tensor,
    gamma: float,
    delta: float,
    lam: float,
    asymmetric: bool,
) -> torch.Tensor:
    """torch 朴素版（float64，可微）：公式同 _naive_unified_focal_np，用于梯度对拍。"""
    ce = -y * torch.log(prob)
    w = torch.full_like(prob, delta)
    w[:, 0] = 1.0 - delta
    if gamma != 0.0:
        factor = (1.0 - prob) ** gamma
        if asymmetric:
            factor = torch.cat([factor[:, :1], torch.ones_like(factor[:, 1:])], dim=1)
        ce = ce * factor
    mf = (w * ce).sum(dim=1).mean()

    axes = list(range(2, prob.dim()))
    tp = (prob * y).sum(axes)
    fn = (y * (1.0 - prob)).sum(axes)
    fp = ((1.0 - y) * prob).sum(axes)
    ti = tp / (tp + delta * fn + (1.0 - delta) * fp)
    base = 1.0 - ti
    if asymmetric:
        base = torch.cat([base[:, :1], base[:, 1:] ** (1.0 - gamma)], dim=1)
    else:
        base = base ** (1.0 - gamma)
    ftl = base.mean()
    return lam * mf + (1.0 - lam) * ftl


# ---------------------------------------------------------------------------
# 自测：CPU 可跑、固定随机种子、数秒内完成
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)

    def _rand_prob64(*shape: int) -> torch.Tensor:
        """构造 (0.05,0.95) 内的 float64 概率张量（远离 clip 边界，供 ε=0 对拍）。"""
        return torch.rand(*shape, dtype=torch.float64) * 0.9 + 0.05

    # 1) 与朴素论文公式对拍：二类（C==1 输入，内部扩展两通道）2D+3D × 对称/非对称 × 多组 γ
    p2d = _rand_prob64(2, 1, 16, 16)
    m2d = (torch.rand(2, 1, 16, 16, dtype=torch.float64) > 0.7).double()
    p3d = _rand_prob64(1, 1, 8, 8, 8)
    m3d = (torch.rand(1, 1, 8, 8, 8, dtype=torch.float64) > 0.8).double()
    max_diff = 0.0
    for p, m in ((p2d, m2d), (p3d, m3d)):
        p2 = torch.cat([1.0 - p, p], dim=1).numpy()
        y2 = torch.cat([1.0 - m, m], dim=1).numpy()
        for asym in (False, True):
            for g in (0.0, 0.5, 1.0):
                fn = asymmetric_unified_focal_loss if asym else unified_focal_loss
                got = fn(p, m, gamma=g, delta=0.6, weight=0.5, eps=0.0).item()
                ref = _naive_unified_focal_np(p2, y2, g, 0.6, 0.5, asym)
                max_diff = max(max_diff, abs(got - ref))
    assert max_diff < 1e-10, max_diff
    print(f"PASS 朴素论文公式对拍（二类 2D/3D × 对称/非对称 × γ∈{{0,0.5,1}}，ε=0）：max|Δ|={max_diff:.2e}")

    # 2) 与朴素论文公式对拍：多类（C=4 2D、C=3 3D，one-hot 真值）× 对称/非对称
    p4 = _rand_prob64(2, 4, 12, 12)
    y4 = F.one_hot(torch.randint(0, 4, (2, 12, 12)), 4).permute(0, 3, 1, 2).double()
    p3 = _rand_prob64(1, 3, 8, 8, 8)
    y3 = F.one_hot(torch.randint(0, 3, (1, 8, 8, 8)), 3).permute(0, 4, 1, 2, 3).double()
    max_diff = 0.0
    for p, y in ((p4, y4), (p3, y3)):
        pn, yn = p.numpy(), y.numpy()
        for asym in (False, True):
            for g in (0.0, 0.5):
                fn = asymmetric_unified_focal_loss if asym else unified_focal_loss
                got = fn(p, y, gamma=g, delta=0.6, weight=0.5, eps=0.0).item()
                ref = _naive_unified_focal_np(pn, yn, g, 0.6, 0.5, asym)
                max_diff = max(max_diff, abs(got - ref))
    assert max_diff < 1e-10, max_diff
    print(f"PASS 朴素论文公式对拍（多类 C=4/3，one-hot）：max|Δ|={max_diff:.2e}")

    # 3) 梯度对拍：主实现 autograd 梯度 vs 朴素 torch 参考的 autograd 梯度（float64）
    max_gdiff = 0.0
    for asym in (False, True):
        pp = _rand_prob64(2, 2, 10, 10).requires_grad_(True)
        yy = F.one_hot(torch.randint(0, 2, (2, 10, 10)), 2).permute(0, 3, 1, 2).double()
        fn = asymmetric_unified_focal_loss if asym else unified_focal_loss
        g_mine = torch.autograd.grad(fn(pp, yy, gamma=0.5, delta=0.6, weight=0.5, eps=0.0), pp)[0]
        qq = pp.detach().clone().requires_grad_(True)
        g_ref = torch.autograd.grad(
            _naive_unified_focal_torch(qq, yy, 0.5, 0.6, 0.5, asym), qq
        )[0]
        max_gdiff = max(max_gdiff, (g_mine - g_ref).abs().max().item())
        assert torch.isfinite(g_mine).all()
    assert max_gdiff < 1e-10, max_gdiff
    print(f"PASS 梯度对拍（float64，对称/非对称）：max|Δgrad|={max_gdiff:.2e}")

    # 4) forward+backward 有限性（fp32）：二类/多类 × 概率/logits 入口 × 极端 logits
    for pred, tgt, lg in (
        (torch.randn(2, 1, 16, 16), (torch.rand(2, 1, 16, 16) > 0.8).float(), True),
        (torch.randn(2, 3, 8, 8, 8), torch.randint(0, 3, (2, 1, 8, 8, 8)), True),
        (torch.rand(2, 2, 16, 16), (torch.rand(2, 1, 16, 16) > 0.8).float(), False),
        (torch.rand(2, 3, 8, 8, 8), torch.randint(0, 3, (2, 1, 8, 8, 8)), False),
        (torch.randn(2, 1, 8, 8) * 100.0, (torch.rand(2, 1, 8, 8) > 0.8).float(), True),   # 极端 logits
        (torch.randn(2, 4, 8, 8) * 100.0, torch.randint(0, 4, (2, 1, 8, 8)), True),
    ):
        for fn in (unified_focal_loss, asymmetric_unified_focal_loss):
            x = pred.clone().requires_grad_(True)
            loss = fn(x, tgt, from_logits=lg)
            loss.backward()
            assert torch.isfinite(loss), (fn.__name__, loss)
            assert torch.isfinite(x.grad).all(), (fn.__name__, "grad 含 NaN/Inf")
    print("PASS forward+backward 梯度有限（fp32，概率/logits 入口，极端 logits ±100）")

    # 5) 退化输入：全零前景目标 / 全前景目标 / 单类 one-hot / 常数概率 0.5 / 完美预测
    z = torch.zeros(1, 1, 8, 8)
    o = torch.ones(1, 1, 8, 8)
    pc = torch.full((1, 1, 8, 8), 0.5)
    pm = _rand_prob64(1, 1, 8, 8).float()
    for pred, tgt in ((pc, z), (pc, o), (pm, z), (pm, o), (pm, pm)):
        for fn in (unified_focal_loss, asymmetric_unified_focal_loss):
            for g in (0.0, 0.5, 1.0):
                v = fn(pred, tgt, gamma=g)
                assert torch.isfinite(v), (fn.__name__, g, v)
    y1 = torch.full((1, 1, 8, 8), 1, dtype=torch.long)  # 多类全部同一类
    pm4 = _rand_prob64(1, 3, 8, 8).float()
    for fn in (unified_focal_loss, asymmetric_unified_focal_loss):
        assert torch.isfinite(fn(pm4, y1))
    print("PASS 退化输入有限（全零/全前景目标、单类 one-hot、常数概率、完美预测、γ∈{0,0.5,1}）")

    # 6) 特例退化：γ=0、δ=0.5、λ=1 → 0.5×加权 CE（论文特例关系；0.5 来自 δ=0.5 的类权重）
    p = _rand_prob64(2, 1, 12, 12)
    m = (torch.rand(2, 1, 12, 12, dtype=torch.float64) > 0.7).double()
    got = unified_focal_loss(p, m, gamma=0.0, delta=0.5, weight=1.0, eps=0.0)
    # 二类 BCE：背景体素贡献 -log(1-p)、前景体素贡献 -log(p)
    ce = (-((1.0 - m) * torch.log(1.0 - p) + m * torch.log(p))).mean()
    assert abs(got.item() - 0.5 * ce.item()) < 1e-12, (got.item(), ce.item())
    print(f"PASS 特例 γ=0,δ=0.5,λ=1 ≡ 0.5×CE：got={got.item():.6f} ref={0.5 * ce.item():.6f}")

    # 7) 特例退化：γ=0、δ=0.5、λ=0 → 各类 soft Dice loss 均值（1 - 2TP/(ΣP+ΣG)，闭式）
    got = unified_focal_loss(p, m, gamma=0.0, delta=0.5, weight=0.0, eps=0.0)
    p2 = torch.cat([1.0 - p, p], dim=1).numpy()
    y2 = torch.cat([1.0 - m, m], dim=1).numpy()
    tp = (p2 * y2).sum(axis=(2, 3))
    dice_loss = (1.0 - 2.0 * tp / (p2.sum(axis=(2, 3)) + y2.sum(axis=(2, 3)))).mean()
    assert abs(got.item() - dice_loss) < 1e-12, (got.item(), dice_loss)
    print(f"PASS 特例 γ=0,δ=0.5,λ=0 ≡ 各类平均 soft Dice loss：got={got.item():.6f} ref={dice_loss:.6f}")

    # 8) γ=0 时对称 ≡ 非对称（两组件的 focal 调制均消失，式(20)(21) 退化为式(16)(17)）
    for tgt in (m, 1.0 - m):
        s = unified_focal_loss(p, tgt, gamma=0.0, delta=0.6, weight=0.3, eps=0.0)
        a = asymmetric_unified_focal_loss(p, tgt, gamma=0.0, delta=0.6, weight=0.3, eps=0.0)
        assert abs(s.item() - a.item()) < 1e-12, (s.item(), a.item())
    print("PASS γ=0 时对称 ≡ 非对称（focal 调制消失后退化为同一损失）")

    # 9) δ 方向：欠召回预测（丢掉一半前景）下 δ 越大损失必须越大（δ 加权 FN/前景）
    #    0/1 硬预测含 log(0)，用默认 ε（该参数正是为此存在）
    under = m.clone()
    under[:, :, 6:, :] = 0.0  # 预测丢掉下半空间 → FN 增加
    l_small = unified_focal_loss(under, m, gamma=0.0, delta=0.2, weight=0.5)
    l_big = unified_focal_loss(under, m, gamma=0.0, delta=0.8, weight=0.5)
    assert torch.isfinite(l_small) and torch.isfinite(l_big)
    assert l_big > l_small, (l_small.item(), l_big.item())
    print(f"PASS δ 方向：欠召回下 loss(δ=0.8)={l_big.item():.4f} > loss(δ=0.2)={l_small.item():.4f}")

    # 10) focal 行为：易例（预测接近真值）上 γ>0 压低 Focal 分量（λ=1）；难例上不放大
    easy_p = m * 0.95 + (1.0 - m) * 0.05
    hard_p = m * 0.30 + (1.0 - m) * 0.70
    fl_easy_g0 = unified_focal_loss(easy_p, m, gamma=0.0, delta=0.6, weight=1.0, eps=0.0)
    fl_easy_g5 = unified_focal_loss(easy_p, m, gamma=0.5, delta=0.6, weight=1.0, eps=0.0)
    fl_hard_g0 = unified_focal_loss(hard_p, m, gamma=0.0, delta=0.6, weight=1.0, eps=0.0)
    fl_hard_g5 = unified_focal_loss(hard_p, m, gamma=0.5, delta=0.6, weight=1.0, eps=0.0)
    assert fl_easy_g5 < fl_easy_g0, (fl_easy_g0.item(), fl_easy_g5.item())
    assert fl_hard_g5 <= fl_hard_g0, (fl_hard_g0.item(), fl_hard_g5.item())
    print(
        f"PASS focal 聚焦方向：易例 γ=0.5 压低损失（{fl_easy_g0.item():.4f}→{fl_easy_g5.item():.4f}），"
        f"难例不放大（{fl_hard_g0.item():.4f}→{fl_hard_g5.item():.4f}）"
    )

    # 11) 端到端小模型：2D/3D 网络 + logits 入口 forward+backward，打印参数量
    net2d = nn.Sequential(
        nn.Conv2d(2, 8, 3, padding=1), nn.ReLU(),
        nn.Conv2d(8, 4, 3, padding=1), nn.ReLU(),
        nn.Conv2d(4, 1, 1),
    )
    x2d = torch.randn(2, 2, 32, 32)
    y2d = (torch.rand(2, 1, 32, 32) > 0.9).float()  # 极不平衡：前景约 10%
    loss2d = AsymmetricUnifiedFocalLoss(gamma=0.5, delta=0.6, from_logits=True)(net2d(x2d), y2d)
    loss2d.backward()
    assert torch.isfinite(loss2d) and all(torch.isfinite(q.grad).all() for q in net2d.parameters() if q.grad is not None)
    n2d = sum(q.numel() for q in net2d.parameters())

    net3d = nn.Sequential(
        nn.Conv3d(2, 4, 3, padding=1), nn.ReLU(),
        nn.Conv3d(4, 1, 1),
    )
    x3d = torch.randn(1, 2, 8, 16, 16)
    y3d = (torch.rand(1, 1, 8, 16, 16) > 0.9).float()
    loss3d = UnifiedFocalLoss(gamma=0.5, delta=0.6, from_logits=True)(net3d(x3d), y3d)
    loss3d.backward()
    assert torch.isfinite(loss3d) and all(torch.isfinite(q.grad).all() for q in net3d.parameters() if q.grad is not None)
    n3d = sum(q.numel() for q in net3d.parameters())
    print(
        f"PASS 端到端小模型：2D 参数量 {n2d} loss={loss2d.item():.4f}；"
        f"3D 参数量 {n3d} loss={loss3d.item():.4f}（均 A-UFL/UFL logits 入口）"
    )

    print("PASS 全部自测通过")
