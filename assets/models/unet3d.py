"""unet3d — 3D U-Net 参考实现（自包含，仅依赖 torch）[model]

论文支撑（统一四问）：
- 哪篇论文用了它：Ö. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox, O. Ronneberger，
  《3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation》，
  MICCAI 2016。链接：https://arxiv.org/abs/1606.06650
- 目的：在只有稀疏标注（每个体积人工标注 19–33 张正交切片，三样本合计 77 张）的
  条件下端到端训练 3D 分割网络：半自动场景把用户的少量标注"致密化"为完整 3D
  分割；全自动场景在稀疏标注训练集上训练后分割新体积。
- 效果：半自动 3 折交叉验证平均 IoU 0.863（带 BN；不带 BN 0.842；同数据 2D U-Net
  仅 0.796，论文表 1）；全自动跨体积测试平均 IoU 0.704（带 BN）/ 0.723（不带
  BN），2D U-Net 仅 0.547（论文表 3）；每个正交方向只标 1 张切片（约 2.5% 体素）
  IoU 0.331–0.483，标 (5,5,3) 张（约 8.9% 体素）三样本 IoU 0.849–0.872（0.856/
  0.849/0.872，论文表 2）。
- 结论：网络可从零端到端训练、无需预训练；把未标注体素的损失权重置 0 即可从稀疏
  标注学习到稠密 3D 分割；3D 上下文相对 2D 提升显著（全自动 0.704 对 0.547）。
  作者预期该架构适用于其他生物医学体积分割任务。

实现说明（论文正文没有编号公式，故按章节号对应）：
- 架构（论文第 2 节与图 2）：分析路每级两个 3×3×3 卷积（各接 ReLU，BN 置于每个
  ReLU 之前）+ 2×2×2、步长 2 的最大池化；合成路每级 2×2×2、步长 2 的
  upconvolution + 两个 3×3×3 卷积（各接 ReLU）；等分辨率 shortcut 拼接进合成路；
  末端 1×1×1 卷积降到类别数；通道数在池化前翻倍（论文引 Szegedy 等的建议以避免
  瓶颈）；分析/合成路各"4 个分辨率级"。
- 关键几何（自推导，并由下方自测第 1 项实际验证）：由论文输入 tile 132×132×116 →
  输出 44×44×28 可反推 valid 卷积几何精确对应 4 级（3 次池化）：
  132→128→池 64→60→池 30→26→池 13→底部两卷积→9；上采样 18/28/48，裁剪 shortcut
  后两卷积→44（z 轴 116→112→56→52→26→22→11→7，上采样 14/20/32→28，同样精确
  吻合）。
- 加权 softmax 交叉熵（论文第 2 节末与第 3.2 节；论文明确称这是"允许在稀疏标注上
  训练的关键"）：逐体素权重 w_{y_i}，未标注体素权重 0，背景降权、目标结构升权；
  loss = Σ_i w_{y_i}·(−log softmax(logits)_{y_i}) / Σ_i w_{y_i}。论文未给出公式，
  "除以权重和"的归一化是本实现的选择（另一种常见做法是对逐体素项求和后不归一或
  按体素数平均）。见 weighted_softmax_cross_entropy 与 class_weight_map。若本仓库
  objectives/ 另收 weighted cross-entropy 专条，训练时以其为准；此处最小实现只为
  让本条目自包含。
- 训练细节（论文第 3.2 节，供复现参考，不属于本文件代码）：Caffe + cuDNN、SGD、
  batch size 1、70,000 次迭代（TitanX 约 3 天）；在线增广：旋转、缩放、灰值变换，
  以及在间距 32 体素的网格上采样 std=4 的正态随机向量、经 B 样条插值得到的平滑
  稠密弹性形变（同时作用于图像与标注）。
- 参考的开源实现：
  * nnU-Net PlainConvUNet（本机 /ssd/lx/totalsegmentator/.venv/lib/python3.11/
    site-packages/dynamic_network_architectures/architectures/unet.py:21），本谱系
    的现代代表（totalsegmentator 使用的 3D 骨干）；
  * /ssd/lx/net_arch/arch_Unet3_2D.py（2D 同谱系变体：InstanceNorm + LeakyReLU +
    深监督，池化数按 spacing/patch 自适应）。
- 与开源实现不一致处（以论文为准）：
  * 卷积 padding：论文为 valid（输出小于输入，132×132×116→44×44×28，整卷靠滑窗拼接），本实现
    默认 padded=False；nnU-Net 等现代实现用 padded 卷积使输出与输入同形，本实现
    以 padded=True 提供该现代行为并注明它不是论文做法。
  * BN 统计：论文因 batch size=1 且样本少，"测试时也用当前批统计效果最好"；
    nn.BatchNorm3d 默认用滑动平均统计。本实现默认
    bn_use_batch_stats_at_eval=True（对应 track_running_stats=False，训练/测试都
    用批统计）以贴近论文；要现代行为置 False。
  * 激活与初始化：论文用 ReLU 且未提及任何初始化方案；nnU-Net 用 LeakyReLU +
    He 初始化，均非论文内容，本实现不做默认（保留 PyTorch 默认初始化）。
- 张量布局：输入/输出为 3D 布局 (B, C, D, H, W)；target/weight 为 (B, D, H, W)。
- 数值稳定：加权交叉熵的分母（权重和）用 eps=1e-12 钳制，仅防"全部权重为 0"的
  0/0；正常训练中权重和远大于 1，eps 不影响数值与梯度。对数部分用
  F.log_softmax（内部 log-sum-exp 稳定），不做裸 log。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["UNet3D", "weighted_softmax_cross_entropy", "class_weight_map"]

# 加权交叉熵分母的最小钳制值：仅防 0/0。正常权重和远大于 1（论文逐体素权重为
# O(1) 量级、体素数百万级），故该值不影响数值。
_LOSS_EPS = 1e-12


class _DoubleConvBlock(nn.Module):
    """3D U-Net 基础块：两个 3×3×3 卷积，每个后面接（可选 BN）+ ReLU。

    对应论文第 2 节："each layer contains two 3×3×3 convolutions each followed by
    a rectified linear unit"，且 BN"before each ReLU"。

    参数：
        in_channels: 输入通道数。
        out_channels: 输出通道数。
        batchnorm: 是否在每个 ReLU 前插入 BatchNorm3d（论文第 2 节为 True）。
        padded: True 时卷积 padding=1（输出与输入同形，现代风格）；False 时
            padding=0（valid 卷积，每维每层收缩 2，论文几何）。
        bn_use_batch_stats_at_eval: True 时 BatchNorm3d 以
            track_running_stats=False 创建（训练/测试都用当前批统计，论文做法，
            因论文 batch size=1）；False 时用常规滑动统计（现代做法）。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        batchnorm: bool,
        padded: bool,
        bn_use_batch_stats_at_eval: bool,
    ) -> None:
        super().__init__()
        padding = 1 if padded else 0
        modules: list[nn.Module] = []
        for c_in in (in_channels, out_channels):
            modules.append(nn.Conv3d(c_in, out_channels, kernel_size=3, padding=padding))
            if batchnorm:
                modules.append(
                    nn.BatchNorm3d(
                        out_channels,
                        track_running_stats=not bn_use_batch_stats_at_eval,
                    )
                )
            modules.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 (B, C, D, H, W) → 输出 (B, out_channels, D', H', W')。

        valid 模式下两个 3×3×3 卷积各收缩 2，共 D'=D−4、H'=H−4、W'=W−4
        （实测 (1,1,10,10,10)→(1,2,6,6,6)）；padded 模式下同形。
        """
        return self.block(x)


def _center_crop_to(x: torch.Tensor, size: torch.Size) -> torch.Tensor:
    """把张量 x 的空间维 (D, H, W) 中心裁剪到 size。

    valid 卷积下 shortcut 特征大于上采样特征，需按经典 U-Net 的中心裁剪对齐后
    再拼接。若某维现尺寸小于目标尺寸则报错（说明输入过小，见 UNet3D 文档）。

    参数：
        x: (B, C, D, H, W) 特征图。
        size: 长度 3 的目标空间尺寸 (D', H', W')。
    返回：
        (B, C, D', H', W') 裁剪后的特征图。
    """
    for i in range(3):
        cur = x.shape[2 + i]
        target = int(size[i])
        if cur == target:
            continue
        if cur < target:
            raise ValueError(
                f"shortcut 特征第 {i} 维尺寸 {cur} 小于目标 {target}：输入体积过小，"
                f"valid 模式 4 级网络要求每维 ≥ 92（建议直接用 132×132×116 或更大）"
            )
        offset = (cur - target) // 2
        x = x.narrow(2 + i, offset, target)
    return x


class UNet3D(nn.Module):
    """3D U-Net（Çiçek et al., MICCAI 2016）的主网络。

    布局：输入 (B, in_channels, D, H, W)；输出 (B, n_classes, D', H', W') 的
    未归一化 logits（softmax/加权交叉熵由调用方做，见 weighted_softmax_cross_
    entropy）。valid 模式（默认，论文几何）下 D' < D 等，需滑窗+拼接推理整卷；
    padded 模式下输出与输入同形。

    参数：
        in_channels: 输入通道数（论文用 3 个荧光通道；CT/MRI 通常为 1）。
        n_classes: 输出类别数（论文为 3：管内/小管/背景；"未标注"不占输出通道，
            而是以权重 0 被损失忽略，见 class_weight_map）。
        channels: 各分辨率级的通道数元组，长度 = 级数（4 级 = 3 次池化，与论文
            几何一致）。论文正文只说明"通道在池化前逐级翻倍"且总参数量
            19,069,955，逐级具体数字仅见图 2（文本层不可验证），常见复现取
            32 起步；故默认 (32, 64, 128, 256)。
        batchnorm: 是否使用 BN（论文第 2 节为 True）。
        padded: False = 论文 valid 卷积（4 级时每维输入 ≥ 92，132×132×116 →
            44×44×28）；True = 现代 padded 卷积（输出同形；非论文做法，见文件头
            说明）。
        bn_use_batch_stats_at_eval: True = 测试时也用当前批统计（论文做法，因
            batch size=1）；False = 常规滑动统计（现代做法）。

    示例：
        >>> model = UNet3D(in_channels=1, n_classes=3)
        >>> x = torch.randn(1, 1, 132, 132, 116)
        >>> logits = model(x)          # (1, 3, 44, 44, 28)
        >>> # 稀疏标注监督：未标注体素(255)权重 0
        >>> target = torch.randint(0, 3, (1, 132, 132, 116))
        >>> w = class_weight_map(target, (0.3, 1.0, 1.0), unlabeled_value=255)
        >>> loss = weighted_softmax_cross_entropy(logits, target, w)
    """

    def __init__(
        self,
        in_channels: int = 1,
        n_classes: int = 2,
        channels: tuple[int, ...] = (32, 64, 128, 256),
        batchnorm: bool = True,
        padded: bool = False,
        bn_use_batch_stats_at_eval: bool = True,
    ) -> None:
        super().__init__()
        if len(channels) < 2:
            raise ValueError("channels 至少需要 2 级（1 次池化），论文为 4 级")
        self.padded = padded
        self.n_levels = len(channels)

        # 分析路：每级两个 3×3×3 卷积；级间 2×2×2 最大池化（论文第 2 节）
        self.enc_blocks = nn.ModuleList()
        c_prev = in_channels
        for c in channels:
            self.enc_blocks.append(
                _DoubleConvBlock(c_prev, c, batchnorm, padded, bn_use_batch_stats_at_eval)
            )
            c_prev = c
        self.pools = nn.ModuleList(
            [nn.MaxPool3d(kernel_size=2, stride=2) for _ in range(self.n_levels - 1)]
        )
        # 合成路：2×2×2、步长 2 的 upconvolution（通道减半）+ 两个 3×3×3 卷积；
        # 卷积输入是"上采样特征 + shortcut"的拼接（2×channels[i]）
        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose3d(channels[i + 1], channels[i], kernel_size=2, stride=2)
                for i in range(self.n_levels - 1)
            ]
        )
        self.dec_blocks = nn.ModuleList(
            [
                _DoubleConvBlock(
                    2 * channels[i], channels[i], batchnorm, padded, bn_use_batch_stats_at_eval
                )
                for i in range(self.n_levels - 1)
            ]
        )
        # 末端 1×1×1 卷积降到类别数（论文第 2 节）；无 BN/ReLU，输出 logits
        self.head = nn.Conv3d(channels[0], n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_channels, D, H, W) → (B, n_classes, D', H', W') logits。"""
        if x.dim() != 5:
            raise ValueError(f"期望 (B, C, D, H, W) 5 维输入，得到 shape={tuple(x.shape)}")
        # 分析路：0..n_levels-2 级各"两卷积 → 池化"，保留池化前特征作 shortcut
        skips: list[torch.Tensor] = []
        h = x
        for i in range(self.n_levels - 1):
            h = self.enc_blocks[i](h)
            skips.append(h)
            h = self.pools[i](h)
        # 底部级：只做两个卷积，不再池化（论文第 2 节）
        h = self.enc_blocks[-1](h)
        # 合成路：上采样 → shortcut 中心裁剪对齐 → 拼接 → 两个卷积
        for i in range(self.n_levels - 2, -1, -1):
            h = self.upconvs[i](h)
            skip = _center_crop_to(skips[i], h.shape[2:])
            h = torch.cat([h, skip], dim=1)
            h = self.dec_blocks[i](h)
        return self.head(h)


def class_weight_map(
    target: torch.Tensor,
    class_weights: tuple[float, ...] | list[float],
    unlabeled_value: int | None = None,
) -> torch.Tensor:
    """按论文第 2 节末/第 3.2 节构造逐体素权重图（稀疏标注训练的关键）。

    规则：第 c 类体素赋权 class_weights[c]；论文中背景降权、目标结构升权以平衡
    二者对损失的贡献；"未标注"体素权重必须为 0（论文 label 3 = "unlabeled"，
    "do not contribute to the loss computation, i.e. have a weight of 0"）。

    参数：
        target: (B, D, H, W) 整型张量，逐体素类别标签。
        class_weights: 长度 n_classes 的权重序列，如背景 0.3、目标 1.0。
        unlabeled_value: 若给定（如 255），target 中等于该值的体素权重置 0
            （对应论文把未标注切片整体记为 label 3 的做法）。
    返回：
        (B, D, H, W) float32 权重图。不属于任何已知类别的体素权重默认 0（安全
        兜底，避免未定义行为）。
    """
    t = target.long()
    weight = torch.zeros(t.shape, dtype=torch.float32)
    for c, w in enumerate(class_weights):
        weight.masked_fill_(t == c, float(w))
    if unlabeled_value is not None:
        weight.masked_fill_(t == int(unlabeled_value), 0.0)
    return weight


def weighted_softmax_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """加权 softmax 交叉熵（论文第 2 节末与第 3.2 节，无编号公式，按文字实现）。

    loss = Σ_i w_{y_i} · ( −log softmax(logits)_{y_i} ) / ( Σ_i w_{y_i} + eps )

    论文未给出公式；逐体素项按"权重和"归一化是本实现的选择（也可求和不归一或按
    体素数平均）。其中 eps=1e-12 仅防"全部权重为 0"时的 0/0（此时返回 0），正常
    训练中权重和远大于 1，不影响数值与梯度；对数用 F.log_softmax（内部
    log-sum-exp，稳定）。

    参数：
        logits: (B, C, D, H, W) 未归一化得分（如 UNet3D 的输出）。
        target: (B, D, H, W) int64 逐体素类别。未标注体素可以留在 target 里
            （如记为 255）：越界索引会被钳制到合法类别范围，仅保证 gather 合法；
            其数值无贡献的前提是对应权重为 0（见 class_weight_map）。
        weight: (B, D, H, W) 浮点逐体素权重（未标注体素为 0，见 class_weight_map）。
    返回：
        标量张量 loss。
    """
    if weight.shape != target.shape:
        raise ValueError(
            f"weight shape {tuple(weight.shape)} 应与 target shape {tuple(target.shape)} 一致"
        )
    n_classes = logits.shape[1]
    # 未标注等"忽略"记号（如 255）钳到合法类别范围：这些体素权重必须为 0
    # （class_weight_map 保证），故钳制不改变损失数值与梯度，只让 gather 合法。
    t = target.clamp(0, n_classes - 1)
    log_prob = F.log_softmax(logits, dim=1)  # (B, C, D, H, W)，数值稳定
    nll = -log_prob.gather(1, t.unsqueeze(1)).squeeze(1)  # (B, D, H, W)
    denom = weight.sum().clamp_min(_LOSS_EPS)
    return (weight * nll).sum() / denom


if __name__ == "__main__":
    torch.manual_seed(0)  # 固定随机种子，保证自测可复现

    # ---- 1) 论文几何自检：valid 模式 4 级，132×132×116 → 44×44×28（论文第 2 节）
    model = UNet3D(in_channels=3, n_classes=3, channels=(2, 4, 8, 16), padded=False)
    x = torch.randn(1, 3, 132, 132, 116)
    model.eval()
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 3, 44, 44, 28), f"输出形状不符: {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "论文几何自检输出含 NaN/Inf"
    print("PASS 1 论文几何: 输入(1,3,132,132,116) → 输出(1,3,44,44,28)，与论文第 2 节一致，输出全部有限")

    # ---- 2) padded 现代模式：输出与输入同形
    model_pad = UNet3D(in_channels=1, n_classes=3, channels=(4, 8, 16), padded=True)
    x2 = torch.randn(1, 1, 32, 32, 24)
    out2 = model_pad(x2)
    assert out2.shape == (1, 3, 32, 32, 24), f"padded 输出形状不符: {tuple(out2.shape)}"
    assert torch.isfinite(out2).all(), "padded 自检输出含 NaN/Inf"
    print("PASS 2 padded 模式: (1,1,32,32,24) → 同形输出，全部有限（现代行为，非论文几何）")

    # ---- 3) 加权 softmax 交叉熵：被忽略体素(权重 0)不贡献损失；梯度有限
    logits = torch.randn(2, 3, 8, 8, 8, requires_grad=True)
    target = torch.randint(0, 3, (2, 8, 8, 8))
    weight_sparse = torch.rand(2, 8, 8, 8)
    ignored = torch.rand(2, 8, 8, 8) < 0.3
    target_ignored = target.clone()
    target_ignored[ignored] = 255  # 未标注记号（合法类别之外）
    weight_sparse[ignored] = 0.0  # 论文：未标注体素权重 0
    loss = weighted_softmax_cross_entropy(logits, target_ignored, weight_sparse)
    assert torch.isfinite(loss), "损失含 NaN/Inf"

    # 手工按公式复算同一权重图下的损失，验证与实现一致
    log_prob = F.log_softmax(logits, dim=1)
    nll = -log_prob.gather(1, target.clamp(0, 2).unsqueeze(1)).squeeze(1)
    loss_manual = ((weight_sparse * nll).sum() / weight_sparse.sum()).detach()
    assert torch.allclose(loss.detach(), loss_manual, atol=1e-6), (
        f"实现与公式不符: {loss.item()} vs {loss_manual.item()}"
    )

    # 未标注体素权重为 0 ⇒ 把这些位置的 logits 任意扰动，损失应严格不变
    pert = torch.where(ignored.unsqueeze(1), 100.0, 0.0)  # (2,1,8,8,8) 广播到 3 通道
    logits2 = logits.detach() + pert
    loss2 = weighted_softmax_cross_entropy(logits2, target_ignored, weight_sparse)
    assert torch.allclose(loss.detach(), loss2, atol=1e-6), (
        f"零权重体素仍影响损失: {loss.item()} vs {loss2.item()}"
    )
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all(), "损失梯度含 NaN/Inf"

    # 全零权重：应返回 0 而非 NaN（eps 防零除）
    loss_zero = weighted_softmax_cross_entropy(
        logits.detach(), target, torch.zeros_like(weight_sparse)
    )
    assert loss_zero.item() == 0.0 and torch.isfinite(loss_zero), (
        f"全零权重应得 0，得到 {loss_zero.item()}"
    )
    print(
        f"PASS 3 加权softmax交叉熵: 与手工公式一致 ({loss.item():.6f})，"
        f"权重 0 的未标注体素不贡献损失，全零权重→0，backward 梯度全部有限"
    )

    # ---- 4) 模型 forward+backward 完整训练步：模拟稀疏标注，断言梯度有限并打印参数量
    model_tr = UNet3D(
        in_channels=1,
        n_classes=3,
        channels=(4, 8, 16),
        padded=True,
        bn_use_batch_stats_at_eval=True,
    )
    x3 = torch.randn(2, 1, 32, 32, 24)
    tgt = torch.randint(0, 3, (2, 32, 32, 24))
    sparse = tgt.clone()
    sparse[torch.rand(2, 32, 32, 24) < 0.3] = 255  # 模拟论文的稀疏标注
    wmap = class_weight_map(sparse, (0.3, 1.0, 1.0), unlabeled_value=255)
    out3 = model_tr(x3)
    assert out3.shape == (2, 3, 32, 32, 24)
    loss3 = weighted_softmax_cross_entropy(out3, sparse, wmap)
    assert torch.isfinite(loss3)
    loss3.backward()
    bad = [
        n
        for n, p in model_tr.named_parameters()
        if p.grad is None or not torch.isfinite(p.grad).all()
    ]
    assert not bad, f"以下参数梯度缺失或含 NaN/Inf: {bad}"
    n_params = sum(p.numel() for p in model_tr.parameters())
    print(
        f"PASS 4 模型 forward+backward: 稀疏标注训练步 loss={loss3.item():.4f}，"
        f"全部梯度有限，参数量={n_params}（小测试配置 channels=(4,8,16)）"
    )

    # ---- 5) 论文风格默认配置的参数量（正文总参数量为 19,069,955；逐级通道数仅见图 2，
    # 本实现取常见复现的 (32,64,128,256)，两者存在出入，见资产卡说明）
    paper_cfg = UNet3D(in_channels=3, n_classes=3)
    n_paper = sum(p.numel() for p in paper_cfg.parameters())
    del paper_cfg
    print(
        f"PASS 5 参数量统计: 论文风格配置（显式 in=3, classes=3, channels=(32,64,128,256), "
        f"valid, BN；构造函数缺省为 in=1, n_classes=2）参数量={n_paper}"
        f"（论文正文给出总数 19069955，逐级通道数仅见图 2）"
    )

    print("PASS 全部自测通过")
