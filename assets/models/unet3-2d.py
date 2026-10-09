"""unet3-2d —— 2D U-Net：编码器-解码器 + 跳跃连接 [model]

论文支撑（统一四问，简体中文）：
- 哪篇论文用了它：Ronneberger、Fischer、Brox（弗莱堡大学），《U-Net: Convolutional
  Networks for Biomedical Image Segmentation》，MICCAI 2015。
  链接：https://arxiv.org/abs/1505.04597（本实现依据 arXiv v1 PDF，2015-05-18，即
  MICCAI 2015 会议版本；另对照 ar5iv HTML 全文核对）。
- 目的：在只有极少量标注（几十张图）的生物医学图像上做像素级分割，替代慢的
  滑窗 CNN（Ciresan et al.）：前者对每个像素单独跑一次网络，冗余且定位精度受限。
- 效果：仅用 30 张 512x512 EM 图训练，ISBI 2012 EM 分割挑战 warping error
  0.000353（当时第一，优于滑窗法 IDSIA 的 0.000420）、rand error 0.0382；
  ISBI 2015 细胞追踪挑战 PhC-U373 IOU 92%（第二名 83%）、DIC-HeLa 77.5%
  （第二名 46%）。Titan 6GB 训练约 10 小时，512x512 推理 <1 秒。
- 结论：编码器-解码器 + 跳跃连接 + 强数据增广（弹性形变）+ 像素加权损失，
  可在极少标注下端到端训练并精确定位；论文自评"适用于更多生物医学分割任务"，
  此后成为医疗 2D 分割事实基线。

实现说明（关键公式与论文式子编号的对应）：
- softmax：p_k(x) = exp(a_k(x)) / Σ_{k'} exp(a_{k'}(x))（论文 §3）。
  数值稳定：torch.log_softmax 内部先减最大激活 a_max 再取 exp，
  exp(a-a_max) <= 1 不会上溢；分母 >= exp(a_k-a_max) 的最大项 1，不会下溢为 0，
  故 log 不产生 -inf，无需额外 eps。
- 加权交叉熵损失：论文式(1) E = Σ_{x∈Ω} w(x)·log(p_ℓ(x)(x))。论文把它写作
  "能量函数"直接最小化；按交叉熵习惯本实现最小化其相反数 -Σ w·log p_ℓ
  （见 weighted_cross_entropy 注释）。
- 权重图：论文式(2) w(x) = w_c(x) + w0·exp(-(d1+d2)^2/(2σ^2))，论文实验取
  w0=10、σ≈5 像素；d1/d2 为到最近/次近细胞"边界"的距离（论文 §3 原文），
  分离边界用形态学操作计算。d1/d2 的逐像素实现论文未规定；本实现按定义
  字面取"到实例边界像素的欧氏距离"：binary_erosion 提取每个实例的 1 像素
  边界，再对边界补集做 EDT（scipy）。对背景像素这等价于"到实例最近像素
  的距离"（实例的最近像素必在边界上）；对实例内部像素则取到自身边界的
  深度（若取"到实例最近像素"则内部恒为 0，与论文字面不符）。
  exp 的指数 <= 0，值域 (0, w0]，无上溢风险，无需 eps。
- He 初始化：论文 §3，权重 ~ N(0, 2/N)，N 为一个神经元的输入节点数
  （例：3x3 卷积、前层 64 通道 → N = 9·64 = 576）。本实现对 Conv2d 取
  N = in_ch·k·k；对 ConvTranspose2d 权重布局为 (in, out, k, k)，按 PyTorch
  kaiming 惯例取 N = out_ch·k·k（论文只针对卷积-ReLU 主干给出 N 的定义，
  转置卷积的 N 论文未规定，此处沿用 nnUNet / PyTorch 一致的约定）。
- 网络结构：论文 §2 + 图 1——收缩路径每级"两个 3x3 无 padding 卷积 + ReLU，
  2x2 max-pool(stride 2) 降采样且通道翻倍"（64→128→256→512，瓶颈 1024）；
  扩张路径每级"2x2 上卷积（通道减半）→ 与收缩路径对应特征图裁剪后拼接 →
  两个 3x3 卷积 + ReLU"；末层 1x1 卷积映射到类别数；共 23 个卷积层。

与开源实现不一致处（以论文为准）：
- 参考了兄弟项目 BC3-seg 在用骨干 /ssd/lx/net_arch/arch_Unet3_2D.py
  （conv2dblock :13-39 用 InstanceNorm2d + LeakyReLU(1e-2) 替代论文的
  无归一化 + ReLU；通道 32 起步且各向异性轴用 1x1 核跳过池化；可选深监督
  变体 Unet3_2DDSV :100）。这些是工程改进，非论文内容：本实现按论文写
  无归一化 + ReLU，norm/激活不加开关，保持参考实现的论文对应性。
- 卷积 padding：论文用无 padding（valid）卷积，输出比输入小一个固定边界
  （572x572 输入 → 388x388 输出，图 1），跳跃特征需"对应裁剪"（copy and
  crop）；mode="valid" 按论文原式实现并复现该尺寸。mode="same"（默认）为
  现代医疗分割框架（含 BC3-seg、nnUNet）普遍采用的 padded 变体，输出与
  输入同尺寸、免裁剪——与论文不一致处，理由是简化分块推理且不改变机制。
- 损失 reduction="sum"（默认 mean）时严格等于论文式(1)的全图求和；
  默认改为按 Σw 归一化的加权平均，是为了让梯度尺度与权重图幅度无关、
  便于换用不同 w0/σ 时沿用同一学习率——论文式(1)本身是求和。

张量/数组布局：模型张量 2D=(B,C,H,W)（torch）；权重图输入/输出为 numpy
数组 (H,W)。依赖：numpy、torch、scipy（EDT）。
"""

from __future__ import annotations

import math
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch import nn

__all__ = [
    "DoubleConvBlock",
    "UNet2D",
    "weighted_cross_entropy",
    "compute_weight_map",
    "he_init_conv_",
]


def he_init_conv_(module: nn.Module, negative_slope: float = 0.0) -> None:
    """按论文 §3 的 He 初始化填充 Conv2d / ConvTranspose2d 权重（原位修改）。

    权重采样自 N(0, std^2)，std = sqrt(2 / ((1 + negative_slope^2) · N))。
    ReLU（negative_slope=0）时退化为论文的 sqrt(2/N)；BC3-seg 家族变体用
    LeakyReLU(1e-2) 时传 negative_slope=1e-2（对应其 InitWeights_He）。
    N = 卷积核面积 × 输入通道数（Conv2d）/ 输出通道数（ConvTranspose2d，
    权重布局 (in, out, k, k)，按 PyTorch kaiming 惯例，见文件头说明）。

    参数：
        module: nn.Module，递归遍历其中的 Conv2d / ConvTranspose2d。
        negative_slope: 激活函数 LeakyReLU 的负斜率；ReLU 填 0。
    """
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            # N = "一个神经元的输入节点数" = 核面积 × 通道数（论文 §3 的 N）
            n = m.weight[0][0].numel() * m.weight.size(1)
            # sqrt(2/(1+a^2)) 因子为 He 初始化对 LeakyReLU 的修正（He et al. 2015）；
            # a=0 时即论文的 sqrt(2/N)。
            std = math.sqrt(2.0 / ((1.0 + negative_slope**2) * n))
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


class DoubleConvBlock(nn.Module):
    """论文 §2 的基本块：两个 3x3 卷积，各接一个 ReLU（论文为无 padding 卷积）。

    张量布局 (B,C,H,W)。padding=0 为论文原式（valid，特征图每层每边缩 1 像素，
    两层共缩 2）；padding=1 为 "same" 变体（尺寸不变）。

    参数：
        in_channels: 输入通道数。
        out_channels: 输出通道数（论文两级相同）。
        padding: 两级卷积的 padding，0（论文）或 1（same 变体）。
    """

    def __init__(self, in_channels: int, out_channels: int, padding: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet2D(nn.Module):
    """2D U-Net（论文图 1 结构）：收缩路径 + 扩张路径 + 跳跃连接，共 23 个卷积层。

    通道按论文每级翻倍：base, 2·base, ..., base·2^depth（论文 base=64、depth=4，
    瓶颈 1024）；扩张路径 2x2 转置卷积（"up-convolution"，通道减半、尺寸翻倍），
    与收缩路径特征拼接后接 DoubleConvBlock；末层 1x1 卷积输出 num_classes 通道。

    张量布局 (B,C,H,W)：
    - mode="same"（默认）：输入输出空间尺寸相同（padded 变体，见文件头不一致说明）。
    - mode="valid"：论文原式，无 padding 卷积；depth=4 时输出比输入每边小 92
      像素（572→388，图 1）。要求每次 max-pool 作用在偶数尺寸上（论文 §2 末句），
      不满足时 forward 抛 ValueError。

    参数：
        in_channels: 输入通道数（灰度 1，RGB 3）。
        num_classes: 输出类别数 K（末层 1x1 卷积的输出通道）。
        base_channels: 第一级通道数，论文取 64。
        depth: max-pool 次数（下采样级数），论文取 4。
        mode: "same"（padded 变体）或 "valid"（论文原式）。
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        base_channels: int = 64,
        depth: int = 4,
        mode: str = "same",
    ) -> None:
        super().__init__()
        if mode not in ("same", "valid"):
            raise ValueError(f"mode 须为 'same' 或 'valid'，收到 {mode!r}")
        if depth < 1:
            raise ValueError(f"depth 至少为 1，收到 {depth}")
        if base_channels < 1:
            raise ValueError(f"base_channels 至少为 1，收到 {base_channels}")
        self.mode = mode
        self.depth = depth

        # 通道序列 base·2^i（i=0..depth）：论文 64→128→256→512→1024（图 1）
        channels = [base_channels * 2**i for i in range(depth + 1)]
        conv_padding = 0 if mode == "valid" else 1

        # 收缩路径：depth 级（每级 DoubleConv + 2x2 max-pool stride 2，论文 §2）
        # 第一级输入为原始图像通道 in_channels；其后各级输入为上一级输出 channels[i-1]
        # （翻倍发生在各级输出通道上，池化不改变通道数）
        self.encoders = nn.ModuleList(
            [
                DoubleConvBlock(
                    in_channels if i == 0 else channels[i - 1],
                    channels[i],
                    padding=conv_padding,
                )
                for i in range(depth)
            ]
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # 瓶颈：最后一次池化后的 DoubleConv，输出 channels[depth]
        self.bottleneck = DoubleConvBlock(channels[depth - 1], channels[depth], padding=conv_padding)

        # 扩张路径：2x2 上卷积（通道减半、尺寸翻倍，论文 §2 的 "up-convolution"）
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(channels[depth - i], channels[depth - 1 - i], kernel_size=2, stride=2)
                for i in range(depth)
            ]
        )
        self.decoders = nn.ModuleList(
            [
                # 拼接后通道 = 上卷积输出 + 跳跃连接，双卷积降到该级通道数
                DoubleConvBlock(channels[depth - i], channels[depth - 1 - i], padding=conv_padding)
                for i in range(depth)
            ]
        )
        # 末层 1x1 卷积把 64 分量特征向量映射到类别数（论文 §2）
        self.final = nn.Conv2d(base_channels, num_classes, kernel_size=1)

        he_init_conv_(self)

    def _center_crop(self, feat: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """中心裁剪到 size=(H,W)（论文图 1 的 copy and crop / §2 的"对应裁剪"）。

        valid 卷积下收缩路径特征图比上采样结果大、多出的正是失去全上下文的
        边界像素，故居中裁掉（572 例：568→392 每边裁 88）。
        """
        diff_h = feat.shape[2] - size[0]
        diff_w = feat.shape[3] - size[1]
        if diff_h < 0 or diff_w < 0:
            raise ValueError(
                f"跳跃特征 {tuple(feat.shape)} 小于目标尺寸 {size}："
                "输入尺寸过小或未满足每次池化前尺寸为偶数（论文 §2）"
            )
        top = diff_h // 2
        left = diff_w // 2
        return feat[:, :, top : top + size[0], left : left + size[1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"输入须为 (B,C,H,W) 四维张量，收到 shape {tuple(x.shape)}")

        # 收缩路径：保留每级池化前的特征图作跳跃连接
        skips = []
        feat = x
        for encoder in self.encoders:
            feat = encoder(feat)
            skips.append(feat)
            if self.mode == "valid" and (feat.shape[2] % 2 or feat.shape[3] % 2):
                # 论文 §2 末句：池化必须作用在偶数尺寸上，否则扩张路径无法对齐
                raise ValueError(
                    f"valid 模式要求池化前尺寸为偶数，得到 {tuple(feat.shape[2:])}；"
                    "请按论文调整输入 tile 尺寸（例：depth=4 时 572）"
                )
            feat = self.pool(feat)

        feat = self.bottleneck(feat)

        # 扩张路径：上采样 → （valid 模式）裁剪跳跃特征 → 拼接 → 双卷积
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            feat = upconv(feat)
            if self.mode == "valid":
                skip = self._center_crop(skip, feat.shape[2:])
            feat = decoder(torch.cat((feat, skip), dim=1))

        return self.final(feat)


def weighted_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """论文式(1)的像素加权交叉熵损失（配合式(2)的权重图）。

    论文 §3：E = Σ_{x∈Ω} w(x)·log(p_ℓ(x)(x))，p_k 为末层特征上的像素级
    softmax，ℓ(x) 为真值标签，w(x) 为权重图。E 即最小化目标；按交叉熵
    习惯本函数返回其相反数（正值），即 -Σ w·(-log p) 的最小化等价形式。

    数值稳定：log p_k = a_k - logsumexp(a)；torch.log_softmax 内部先减最大
    激活再取 exp，指数 <= 0 不上溢，分母含最大项 1 不下溢为 0，log 不产生
    -inf（输入有限时），故无需额外 eps。

    参数与布局：
        logits: (B,K,H,W) 末层未归一化激活 a_k(x)（本文件 UNet2D 的输出）。
        target: (B,H,W) int64 真值标签 ℓ(x)，取值 0..K-1。
        weight: 可选 (B,H,W) float 权重图 w(x)（式(2)；可用 compute_weight_map
                生成）。None 时视为全 1（退化为普通交叉熵）。
        reduction: "mean"（默认，按 Σw 归一化的加权平均，便于沿用同一学习率，
                见文件头不一致说明）或 "sum"（严格等于论文式(1)的全图求和，
                含相反数符号）。
    返回：
        标量损失。
    """
    if logits.dim() != 4:
        raise ValueError(f"logits 须为 (B,K,H,W)，收到 shape {tuple(logits.shape)}")
    expected_target = (logits.shape[0], logits.shape[2], logits.shape[3])
    if target.shape != expected_target:
        raise ValueError(
            f"target 须为 (B,H,W)={expected_target}，收到 {tuple(target.shape)}"
        )
    if target.dtype != torch.int64:
        target = target.long()
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction 须为 'mean' 或 'sum'，收到 {reduction!r}")

    # 论文 §3 的像素级 softmax p_k(x)，取其 log（稳定实现见上注释）
    log_probs = F.log_softmax(logits, dim=1)                      # (B,K,H,W)
    # 取真值类别 ℓ(x) 的 log p_ℓ(x)：交叉熵对每个像素惩罚 p_ℓ(x) 偏离 1 的程度
    nll = -log_probs.gather(1, target.unsqueeze(1)).squeeze(1)    # (B,H,W) = -log p_ℓ

    if weight is not None:
        if weight.shape != nll.shape:
            raise ValueError(f"weight 须为 (B,H,W)={tuple(nll.shape)}，收到 {tuple(weight.shape)}")
        nll = nll * weight.float()

    if reduction == "sum":
        # 论文式(1)：全图求和（取相反数符号，见 docstring）
        return nll.sum()
    # 加权平均：除以 Σw 而非像素数，使梯度尺度与权重图整体幅度无关；
    # 无权重时退化为普通均值。Σw 下限 eps 防零除（全零权重图 + mean 时触发）。
    denom = weight.float().sum() if weight is not None else torch.tensor(
        nll.numel(), dtype=nll.dtype, device=nll.device
    )
    return nll.sum() / denom.clamp_min(1e-12)


def compute_weight_map(
    mask: np.ndarray,
    num_classes: Optional[int] = None,
    w0: float = 10.0,
    sigma: float = 5.0,
) -> np.ndarray:
    """论文式(2)的逐像素损失权重图 w(x) = w_c(x) + w0·exp(-(d1+d2)^2/(2σ^2))。

    - w_c(x)：平衡类别频率的权重。论文只说明其作用、未规定具体形式；此处取
      "类频率倒数" w_c(x) = 1/(K·freq_{ℓ(x)})（全图均值恰为 1），freq 为类
      像素占比，+eps 防零除。num_classes=None 时不加 w_c（即取 1）。
    - d1/d2：到最近/次近细胞"边界"的距离（论文 §3 字面定义）。逐实例实现：
      binary_erosion 提取该实例的 1 像素边界，EDT 到边界补集（见文件头说明）。
      只有一个实例时取 d2 = d1（论文未定义单实例情形，取对称值最保守）。
    - w0、σ：论文实验值 w0=10、σ≈5 像素。

    数值稳定：指数 -(d1+d2)^2/(2σ^2) <= 0，exp 值域 (0, w0]，无上溢；
    d1/d2 很远时结果下溢为 0 属期望行为（远离边界的背景不需要加权）。

    参数与布局（numpy 数组）：
        mask: (H,W) int 标注图；0 为背景，>=1 的不同整数为不同细胞实例。
        num_classes: 类别数 K（含背景），给定时叠加 w_c。
        w0: 式(2)的 w0，论文取 10。
        sigma: 式(2)的 σ（单位：像素），论文约 5。
    返回：
        (H,W) float32 权重图。
    """
    labels = np.asarray(mask)
    if labels.ndim != 2:
        raise ValueError(f"mask 须为 (H,W) 二维数组，收到 shape {labels.shape}")

    instances = np.unique(labels)
    instances = instances[instances != 0]  # 背景不参与细胞间边界加权
    if instances.size == 0:
        border_term = 0.0
    else:
        # 论文 §3 字面定义：d_i = 到"实例 i 边界"的距离。先提取该实例的 1 像素
        # 边界（区域腐蚀一层的差集，即与外部相邻的像素），再 EDT 到边界补集。
        # 背景像素处等价于"到实例最近像素的距离"；实例内部像素得到到自身
        # 边界的深度（若直接 EDT 到实例像素，内部恒为 0，与论文字面不符）。
        # binary_erosion 默认 4 连通结构；1 像素宽的实例腐蚀后为空 → 全域皆边界，
        # 距离仍正确（边界像素自身距离为 0）。
        dist = []
        for lab in instances:
            region = labels == lab
            border = region & ~ndimage.binary_erosion(region)
            dist.append(ndimage.distance_transform_edt(~border))
        dist = np.stack(dist)
        d1 = dist.min(axis=0)
        if instances.size >= 2:
            # 沿实例轴取次小值 = 到次近细胞边界的距离 d2
            d2 = np.partition(dist, 1, axis=0)[1]
        else:
            d2 = d1  # 单实例：论文未定义，取 d2=d1（最保守的对称选择）
        # 式(2)的边界项：指数 <= 0（见上），无上溢风险
        border_term = w0 * np.exp(-((d1 + d2) ** 2) / (2.0 * sigma**2))

    if num_classes is None:
        class_term = np.ones(labels.shape, dtype=np.float64)
    else:
        total = labels.size
        freq = np.bincount(labels.ravel(), minlength=num_classes).astype(np.float64) / total
        # w_c(x) = 1/(K·freq_{ℓ(x)})：全图均值恰为 1；+eps 防某类缺失时零除
        # （缺失类不会被采样到，但显式 eps 使式子对任意输入封闭）
        class_term = 1.0 / (num_classes * freq[labels.ravel()].reshape(labels.shape) + 1e-12)

    weight = class_term + border_term
    return weight.astype(np.float32)


if __name__ == "__main__":
    # 自测：CPU、无数据下载、固定随机种子、几秒内完成
    torch.manual_seed(0)
    np.random.seed(0)

    # ---- 1. 权重图（论文式(2)）：形状、有限性、边界加权行为 ----
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:15, 10:15] = 1  # 实例 1
    mask[10:15, 20:25] = 2  # 实例 2（与实例 1 间隙 5 像素）
    w = compute_weight_map(mask, num_classes=3)
    assert w.shape == (64, 64), f"权重图形状错误: {w.shape}"
    assert w.dtype == np.float32
    assert np.isfinite(w).all(), "权重图含 NaN/Inf"
    assert (w > 0).all(), "权重图应全为正"
    # 间隙中点（约 col 17）d1≈d2≈3 → 边界项 10·exp(-36/50)≈4.87，加权显著
    assert w[12, 17] > 4.0, f"间隙中心应有强边界加权，实际 {w[12, 17]:.3f}"
    # 远离细胞的角落 d1+d2 很大 → 边界项趋 0，只剩类频率项（均值 1 量级）
    assert w[63, 63] < 3.0, f"远离细胞处不应有强边界加权，实际 {w[63, 63]:.3f}"
    # 闭式核对（锁住式(2)与 w_c 的逐值行为）：
    # 背景间隙像素 (12,17)：d1=d2=3 → 10·exp(-36/50) + 1/(3·f0)，f0 为背景占比
    f0 = (64 * 64 - 50) / 64**2
    expect_gap = 10.0 * math.exp(-36.0 / 50.0) + 1.0 / (3.0 * f0)
    assert abs(w[12, 17] - expect_gap) < 1e-4, (
        f"间隙像素应等于闭式值 {expect_gap:.5f}，实际 {w[12, 17]:.5f}"
    )
    # 实例 1 边界像素 (12,14)（属实例 1）：d1=0（自身边界）、d2=6（实例 2 边界）
    # → 4096/75 + 10·exp(-36/50)；实例 1 内部像素 (12,12)：d1=2、d2=8，边界项更小
    expect_edge = 64**2 / 75.0 + 10.0 * math.exp(-36.0 / 50.0)
    assert abs(w[12, 14] - expect_edge) < 1e-4, (
        f"实例边界像素应等于闭式值 {expect_edge:.5f}，实际 {w[12, 14]:.5f}"
    )
    assert w[12, 14] > w[12, 12], "边界带应由外向内衰减（论文图 3d 行为）"
    print(
        f"PASS 权重图（式2）: 间隙中心 w={w[12, 17]:.3f}, 实例边界 w={w[12, 14]:.3f}, "
        f"实例内部 w={w[12, 12]:.3f}, 角落 w={w[63, 63]:.3f}, max={w.max():.3f}"
    )

    # ---- 2. 加权交叉熵（论文式(1)）：与手工计算一致 + 梯度有限 ----
    logits = torch.randn(2, 3, 32, 32, requires_grad=True)
    target = torch.randint(0, 3, (2, 32, 32))
    weight = torch.rand(2, 32, 32) + 0.5  # 任意正权重图
    loss_sum = weighted_cross_entropy(logits, target, weight, reduction="sum")
    manual = (
        -torch.log_softmax(logits, dim=1).gather(1, target.unsqueeze(1)).squeeze(1) * weight
    ).sum()
    assert torch.allclose(loss_sum, manual, atol=1e-5), (
        f"sum 模式与式(1)手工计算不一致: {loss_sum.item()} vs {manual.item()}"
    )
    loss_mean = weighted_cross_entropy(logits, target, weight, reduction="mean")
    assert torch.allclose(loss_mean, manual / weight.sum(), atol=1e-6)
    logits.grad = None  # 清空 sum 断言阶段可能留下的梯度
    weighted_cross_entropy(logits, target, weight, reduction="mean").backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all(), "损失梯度含 NaN/Inf"
    print(f"PASS 加权交叉熵（式1）: sum={loss_sum.item():.4f}, mean={loss_mean.item():.4f}, 梯度有限")

    # ---- 3. 模型 same 模式：前向形状 + 反向梯度 + 参数量 ----
    net = UNet2D(in_channels=1, num_classes=2, base_channels=8, depth=4, mode="same")
    x = torch.randn(1, 1, 64, 64)
    out = net(x)
    assert out.shape == (1, 2, 64, 64), f"same 模式应输出同尺寸，得到 {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "输出含 NaN/Inf"
    tgt = torch.randint(0, 2, (1, 64, 64))
    weighted_cross_entropy(out, tgt).backward()
    n_grad = sum(1 for p in net.parameters() if p.grad is not None)
    assert n_grad == sum(1 for _ in net.parameters()), "存在未收到梯度的参数"
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None), (
        "参数梯度含 NaN/Inf"
    )
    n_small = sum(p.numel() for p in net.parameters())
    print(f"PASS 模型 same 模式: 输出 {tuple(out.shape)}, 参数量 {n_small:,}, 梯度有限")

    # ---- 4. 模型 valid 模式：复现论文图 1 的 572x572 → 388x388，并计时 ----
    net_valid = UNet2D(in_channels=1, num_classes=2, base_channels=8, depth=4, mode="valid")
    t0 = time.perf_counter()
    with torch.no_grad():
        out_valid = net_valid(torch.randn(1, 1, 572, 572))
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert out_valid.shape == (1, 2, 388, 388), (
        f"valid 模式 572 输入应输出 388（论文图 1），得到 {tuple(out_valid.shape)}"
    )
    assert torch.isfinite(out_valid).all()
    print(
        f"PASS 模型 valid 模式: 572x572 → 388x388（论文图 1 尺寸复现），"
        f"CPU 前向 {elapsed_ms:.0f} ms"
    )

    # ---- 5. 论文原始配置（base=64, depth=4）参数量 ----
    net_paper = UNet2D(in_channels=1, num_classes=2, base_channels=64, depth=4, mode="same")
    n_paper = sum(p.numel() for p in net_paper.parameters())
    assert 30_000_000 < n_paper < 32_000_000, f"论文配置参数量异常: {n_paper:,}"
    del net_paper
    print(f"PASS 论文配置参数量: {n_paper:,} ≈ {n_paper / 1e6:.2f}M（2x4090 23GB 预算下宽裕）")

    print("PASS 全部自测通过")
