"""vnet [model] — V-Net：端到端 3D 全卷积医学影像分割网络（Milletari et al., 2016）

论文支撑（统一四问）
- 哪篇论文用了它：Milletari、Navab、Ahmadi，《V-Net: Fully Convolutional Neural Networks
  for Volumetric Medical Image Segmentation》（V-Net），2016，3DV。
  链接：https://arxiv.org/abs/1606.04797 。本次阅读版本：arXiv v1 PDF 原文
  （https://arxiv.org/pdf/1606.04797v1，2016-06-15，2026-10-05 下载并用 pdftotext 抽取全文）
  + ar5iv HTML 对照（https://ar5iv.labs.arxiv.org/html/1606.04797）。
- 目的：前列腺 MRI 体数据的端到端 3D 分割；医学体数据中目标解剖常只占极小部分，
  交叉熵训练易偏向背景，论文同时提出基于 Dice 系数的目标函数来缓解前景/背景体素
  强不平衡（该损失的独立条目见 assets/objectives/dice-loss.md，与本条目同源）。
- 效果：PROMISE 2012（训练 50 卷、测试 30 卷）：V-Net + Dice loss 平均 Dice
  0.869±0.033、Hausdorff 距离 5.71±1.20 mm、挑战赛得分 82.39；同一网络换成带样本
  重加权的多项逻辑损失只有 0.739±0.088、10.55±5.38 mm、63.30。
- 结论：全卷积 + 逐 stage 残差的 3D U 形结构配合 Dice 目标可行且快（约 1 秒/卷）；
  Dice 目标"无需对不同类别样本加权即可平衡前景/背景"，适用于前景占比小的二值 3D 分割。

实现说明（关键公式/结构与论文对应）
- 网络结构（论文第 2 节 + Fig. 2 + 官方 Caffe 实现逐层核对）：
  * 左（压缩）路径 5 个 stage，右（解压）路径 4 个 stage + 输出头；每个 stage
    "the input of each stage is (a) used in the convolutional layers ... and (b) added
    to the output of the last convolutional layer of that stage"——即本实现的
    ``_VNetStage``：out = PReLU(stage_in + convs(stage_in))，stage 内每个非末层卷积后
    各接一个 PReLU（PReLU 逐层/逐 stage 位置与官方 prototxt 的 relu_* / outBlock*_RELU 一致）。
  * "The convolutions performed in each stage use volumetric kernels having size
    5×5×5 voxels."——主卷积一律 Conv3d(k=5, padding=2)。
  * 下采样不用 max-pooling，而是 2×2×2、stride 2 的卷积（论文引 [16] 反对 max-pooling；
    对应 prototxt 的 down_pooling_*），通道数随 stage 翻倍；上采样用 2×2×2、stride 2 的
    反卷积（deconv_*）。
  * 跳跃连接：左路径同分辨率特征与右路径 deconv 输出 Concat（官方 prototxt 的
    concat_in*_*；论文 Fig. 2 的 "Fine-grained features forwarding"）。
  * 输出头：5×5×5 卷积 → 2 通道，PReLU，再 1×1×1 卷积 → 2 通道（官方
    conv_in128_chan2_right + conv_in128_chan2_2_right），softmax 后得前景/背景两幅概率图。
    本实现 forward 只返回 logits，softmax 放进 ``DiceLoss``/``predict_binary``（与
    nn.CrossEntropyLoss 同一惯例，数值更稳）。
  * 各 stage 卷积个数与通道（按官方 prototxt 计）：左 [1,1,2,3,3]×(16,32,64,128,256)，
    右 [3,2,1,1]×(256,128,64,32)，与论文正文"每 stage 一到三层卷积、通道逐级加倍"一致。
- 损失（论文第 3 节 "Dice loss layer"，PDF 原文两式均无编号，按出现顺序转写）：
  * 目标函数（最大化 D，等价于最小化 loss = 1 − D）：
        D = 2·Σᵢ pᵢgᵢ / (Σᵢ pᵢ² + Σᵢ gᵢ²)，
    其中 i 遍历全部 N 个体素，pᵢ 为预测概率（softmax 后取前景通道），gᵢ 为真值二值体素。
  * 论文给出的梯度闭式：
        ∂D/∂pⱼ = 2·[ gⱼ(Σᵢpᵢ² + Σᵢgᵢ²) − 2pⱼ·Σᵢpᵢgᵢ ] / (Σᵢpᵢ² + Σᵢgᵢ²)²
    本实现用 autograd 反传；__main__ 自测把 autograd 梯度与上式逐点对拍。
- 与开源/论文不一致处（以论文为准的理由）：
  * 参考的官方实现是 Caffe：github.com/faustomilletari/VNet
    （Prototxt/train_noPooling_ResNet_cinque.prototxt，含 split/concat 残差接线、
    msra 初始化、DiceLoss python 层）。本文件按其逐层配置转写为 PyTorch。
  * 官方 pyLayer.py（DiceLoss 的 Caffe 侧代码，已下载核验）：其 forward 为逐样本
    循环计算 dice 再求和（top=sum）而非逐 batch 合并，backward 才套论文的软梯度
    闭式；本实现损失严格按论文公式、batch 维取"逐样本算 D 再平均"——与官方
    forward 口径一致，且纯论文软式比官方实现更贴论文原文。
  * 逐体素加权推广形式 D_w = 2Σwᵢpᵢgᵢ/(Σwᵢpᵢ² + Σwᵢgᵢ²) 在本次所读的 arXiv v1 中
    **不存在**——v1 原文明确写 "we do not need to assign weights to samples of different
    classes"；``DiceLoss(voxel_weight=...)`` 是本实现提供的可选扩展，默认 None 与论文一致。
  * 论文式子无平滑项；``dice_coefficient`` 在分母加 eps 仅防 0/0（真值与预测全空时
    D 无定义），eps=1e-7 远小于正常体素平方和（N≥1 的量级为 1），对正常样本影响可忽略。
- 布局约定：模型输入/输出 (B, C, D, H, W)；label (B, D, H, W)，取值 {0,1}，1=前景；
  输出通道 0=背景、1=前景（softmax 后取通道 1 为前景概率；官方未注明通道顺序，
  此为常规约定，注明于此）。
- 初始化：官方 prototxt 用 Caffe "msra" filler（variance_norm: 2，按 Caffe
  filler.hpp 的枚举为 AVERAGE，即 fan_in 与 fan_out 方差的平均，与
  kaiming_normal_(mode="fan_out") 接近但不严格相等，后者为本实现采用的近似）
  + constant 0 bias；PReLU 斜率默认 0.25（Caffe/torch 默认一致，
  逐通道，对应 Caffe PReLU 默认 channel-shared=false）。
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["VNet", "DiceLoss", "dice_coefficient", "predict_binary"]

# Dice 分母的平滑项：远小于正常体素平方和量级，仅防 0 除；论文原式无此项。
_DICE_EPS = 1e-7


def dice_coefficient(
    prob: torch.Tensor,
    target: torch.Tensor,
    eps: float = _DICE_EPS,
    voxel_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """逐样本计算论文第 3 节的可导 Dice 系数 D（越大越好，最大 1）。

    布局：``prob`` 与 ``target`` 形状一致，dim 0 为 batch，其余维为体素（3D 时推荐
    (B, D, H, W)）；``prob`` 为前景概率（softmax 后通道 1），``target`` 为 {0,1} 二值
    真值，1=前景。返回形状 (B,) 的张量，第 b 个元素为第 b 个样本的 D_b。

    公式（论文原文，无编号）：D = 2·Σ pᵢgᵢ / (Σ pᵢ² + Σ gᵢ²)。

    参数：
        prob: 预测前景概率，任意浮点张量，数值期望在 [0,1]。
        target: 真值二值张量（{0,1}，1=前景），形状与 ``prob`` 相同。
        eps: 分母平滑项，防真值与预测全空时 0/0（论文原式无平滑，此为数值兜底）。
        voxel_weight: 可选逐体素权重 wᵢ，形状与 ``prob`` 相同；这是本实现的扩展，
            论文 arXiv v1 无此加权形式，默认 None（即 wᵢ≡1，与论文一致）。

    返回：
        形状 (B,) 的逐样本 Dice 系数，数值范围 [0, 1]（eps 兜底时可能略低）。
    """
    prob = prob.reshape(prob.shape[0], -1)
    target = target.reshape(target.shape[0], -1).to(prob.dtype)
    if voxel_weight is not None:
        w = voxel_weight.reshape(voxel_weight.shape[0], -1).to(prob.dtype)
        inter = 2.0 * (w * prob * target).sum(dim=1)
        denom = (w * prob * prob).sum(dim=1) + (w * target * target).sum(dim=1)
    else:
        inter = 2.0 * (prob * target).sum(dim=1)
        denom = (prob * prob).sum(dim=1) + (target * target).sum(dim=1)
    # eps 只加在分母：D = 2Σpg / (Σp² + Σg² + eps)。真值与预测全空时 denom=eps，
    # D→0（保守处理为完全不重叠）；正常样本 Σp²+Σg² ≥ 1，eps 相对影响 < 1e-7。
    return inter / (denom + eps)


class DiceLoss(nn.Module):
    """论文第 3 节的 Dice 目标函数封装：loss = 1 − mean(逐样本 D)。

    与论文的对应：论文在 softmax 之后把前景概率图送入 Dice 目标层优化 1 − D；
    本模块内部先做 softmax（``dim=1``）再取前景通道，等价于论文的
    "softmax + DiceLoss" 两层串联，直接吃网络 logits。

    参数：
        foreground_channel: logits 中前景通道的下标，默认 1（通道 0=背景、1=前景）。
        eps: 分母平滑项（见 ``dice_coefficient``）。
        voxel_weight: 可选逐体素权重张量（扩展功能，论文无；默认 None）。

    forward(logits, target)：
        logits: (B, 2, D, H, W) 网络原始输出；
        target: (B, D, H, W) 二值真值，{0,1}，1=前景；
        返回标量损失 ∈ [0, 1]。
    """

    def __init__(
        self,
        foreground_channel: int = 1,
        eps: float = _DICE_EPS,
        voxel_weight: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.foreground_channel = foreground_channel
        self.eps = eps
        self.voxel_weight = voxel_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.softmax(logits, dim=1)[:, self.foreground_channel]
        dice = dice_coefficient(prob, target, eps=self.eps, voxel_weight=self.voxel_weight)
        return 1.0 - dice.mean()


class _VNetStage(nn.Module):
    """V-Net 的一个 stage：out = PReLU(x + convs(x))（逐 stage 残差，论文第 2 节）。

    stage 内第一个卷积把 ``in_ch`` 映到 ``ch``，其余 ``num_convs-1`` 个卷积保持 ``ch``；
    除最后一层外每个卷积后接逐通道 PReLU（与官方 prototxt 的 relu_conv_* 对应），
    残差相加后再接一个 PReLU（对应 outBlock*_RELU）。

    参数：
        in_ch: stage 输入通道数。
        ch: stage 内各卷积的输出通道数（stage 恒等通道）。
        num_convs: 卷积个数（论文：每 stage 1~3 个）。
    """

    def __init__(self, in_ch: int, ch: int, num_convs: int) -> None:
        super().__init__()
        convs: list[nn.Module] = [nn.Conv3d(in_ch, ch, kernel_size=5, padding=2)]
        convs += [nn.Conv3d(ch, ch, kernel_size=5, padding=2) for _ in range(num_convs - 1)]
        self.convs = nn.ModuleList(convs)
        # 前 num_convs-1 个卷积各配一个 PReLU；残差相加后再来一个（共 num_convs 个）。
        self.prelus = nn.ModuleList([nn.PReLU(num_parameters=ch) for _ in range(num_convs)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for i, conv in enumerate(self.convs):
            y = conv(y)
            if i < len(self.convs) - 1:
                y = self.prelus[i](y)
        # 残差：stage 输入加到末层卷积输出上。in_ch != ch 时（仅首 stage 1→16），
        # 官方 prototxt 用 split×N + concat 显式把输入复制到 ch 通道再相加（tiled1_1），
        # 张量广播加法与其逐元素数值完全等价。
        return self.prelus[-1](y + x)


class _DownBlock(nn.Module):
    """下采样块：2×2×2、stride 2 卷积 + PReLU（官方 down_pooling_*，无 max-pooling）。"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.prelu = nn.PReLU(num_parameters=out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prelu(self.conv(x))


class _UpBlock(nn.Module):
    """上采样块：2×2×2、stride 2 反卷积 + PReLU（官方 deconv_*）。"""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.deconv = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.prelu = nn.PReLU(num_parameters=out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.prelu(self.deconv(x))


class VNet(nn.Module):
    """V-Net 主网络（论文 Fig. 2 / 官方 prototxt 的 PyTorch 转写）。

    布局：输入 (B, in_channels, D, H, W)，要求 D、H、W 均能被 16 整除
    （压缩路径含 4 次 stride-2 下采样，论文用 128×128×64）；输出 (B, 2, D, H, W)
    logits（未过 softmax），通道 0=背景、1=前景。

    参数：
        in_channels: 输入通道数，论文为 1（单模态 MRI）。
        base_channels: 第一 stage 通道数，论文/官方为 16；各 stage 通道按
            16→32→64→128→256 逐级加倍，改此参数即整体缩放网络规模。

    说明：所有主卷积为 5×5×5（padding=2，尺寸不变），下/上采样均为 2×2×2、stride 2，
    与论文正文与官方实现一致；输入尺寸可变（全卷积），但需被 16 整除。
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        c1, c2, c3, c4, c5 = (base_channels * 2**i for i in range(5))  # 16,32,64,128,256

        # 压缩（左）路径：卷积个数 [1,1,2,3,3]（官方 prototxt 实测）。
        self.stage1 = _VNetStage(in_channels, c1, 1)
        self.down1 = _DownBlock(c1, c2)
        self.stage2 = _VNetStage(c2, c2, 1)
        self.down2 = _DownBlock(c2, c3)
        self.stage3 = _VNetStage(c3, c3, 2)
        self.down3 = _DownBlock(c3, c4)
        self.stage4 = _VNetStage(c4, c4, 3)
        self.down4 = _DownBlock(c4, c5)
        self.stage5 = _VNetStage(c5, c5, 3)  # 瓶颈 stage（官方 conv_in8_chan256*）

        # 解压（右）路径：deconv 通道减半，与左路径同分辨率特征 Concat 后进 stage；
        # 卷积个数 [3,2,1,1]（官方 prototxt 实测）。
        self.up4 = _UpBlock(c5, c4)
        self.rstage4 = _VNetStage(2 * c4, c5, 3)  # concat(c4, c4)=2*c4 -> c5（官方 256 出）
        self.up3 = _UpBlock(c5, c3)
        self.rstage3 = _VNetStage(2 * c3, c4, 2)  # concat(c3, c3)=2*c3 -> c4
        self.up2 = _UpBlock(c4, c2)
        self.rstage2 = _VNetStage(2 * c2, c3, 1)  # concat(c2, c2)=2*c2 -> c3
        self.up1 = _UpBlock(c3, c1)
        self.rstage1 = _VNetStage(2 * c1, c2, 1)  # concat(c1, c1)=2*c1 -> c2

        # 输出头：5×5×5 卷积 -> 2 通道 + PReLU，再 1×1×1 卷积 -> 2 通道。
        self.head_conv = nn.Conv3d(c2, 2, kernel_size=5, padding=2)
        self.head_prelu = nn.PReLU(num_parameters=2)
        self.head_1x1 = nn.Conv3d(2, 2, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        """按官方 prototxt 初始化：msra(He, fan_out) 权重 + 零偏置。"""
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 压缩路径
        s1 = self.stage1(x)          # (B, c1, D, H, W)
        s2 = self.stage2(self.down1(s1))  # (B, c2, D/2, H/2, W/2)
        s3 = self.stage3(self.down2(s2))  # (B, c3, D/4, ...)
        s4 = self.stage4(self.down3(s3))  # (B, c4, D/8, ...)
        b = self.stage5(self.down4(s4))   # (B, c5, D/16, ...)

        # 解压路径：上采样 -> 与左路径跳跃特征 Concat -> 残差 stage
        y4 = self.rstage4(torch.cat((self.up4(b), s4), dim=1))
        y3 = self.rstage3(torch.cat((self.up3(y4), s3), dim=1))
        y2 = self.rstage2(torch.cat((self.up2(y3), s2), dim=1))
        y1 = self.rstage1(torch.cat((self.up1(y2), s1), dim=1))

        h = self.head_prelu(self.head_conv(y1))  # (B, 2, D, H, W)
        return self.head_1x1(h)  # logits；softmax 交给 DiceLoss / predict_binary


def predict_binary(
    logits: torch.Tensor,
    threshold: float = 0.5,
    foreground_channel: int = 1,
) -> torch.Tensor:
    """按论文第 3.2 节的测试规则把网络输出转成二值分割。

    论文："The voxels having higher probability (>0.5) to belong to the foreground
    than to the background are considered part of the anatomy."——softmax 后前景概率
    大于阈值（默认 0.5，等价于前景概率大于背景概率）的体素判为前景。

    布局：logits (B, 2, D, H, W)；返回 (B, D, H, W) 的 uint8 张量，取值 {0,1}，1=前景。

    参数：
        logits: 网络原始输出（未经 softmax）。
        threshold: 前景概率阈值，论文取 0.5。
        foreground_channel: 前景通道下标，默认 1。
    """
    prob = torch.softmax(logits, dim=1)[:, foreground_channel]
    return (prob > threshold).to(torch.uint8)


if __name__ == "__main__":
    # 自测：CPU 可跑、无外部数据；覆盖损失 forward/backward、论文梯度闭式对拍、
    # 网络 forward/backward 与参数量、二值化输出。
    import time

    torch.manual_seed(42)

    # ---- 1) Dice 损失：完美预测 / 全空真值 / 随机有限性 ----
    shape = (2, 6, 6, 6)
    target = (torch.rand(shape) < 0.3).float()
    perfect_logits = torch.stack([(1.0 - target) * 20.0, target * 20.0], dim=1)
    loss = DiceLoss()(perfect_logits, target)
    assert 0.0 <= loss.item() < 1e-5, f"完美预测的 Dice loss 应≈0，实际 {loss.item()}"
    empty_target = torch.zeros(shape)
    loss_empty = DiceLoss()(torch.randn(2, 2, 6, 6, 6), empty_target)
    # 真值全空：denom=eps 兜底，D≈0，loss≈1 且有限（保守处理，不产生 NaN）。
    assert torch.isfinite(loss_empty) and 0.99 <= loss_empty.item() <= 1.0
    logits = torch.randn(2, 2, 6, 6, 6, requires_grad=True)
    loss = DiceLoss()(logits, target)
    assert 0.0 <= loss.item() <= 1.0 and torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    print(f"PASS Dice 损失 forward/backward（完美≈0={DiceLoss()(perfect_logits, target).item():.2e}，"
          f"空真值={loss_empty.item():.4f}，随机={1.0 - dice_coefficient(torch.softmax(logits, 1)[:, 1], target).mean().item():.4f}）")

    # ---- 2) 论文梯度闭式对拍：autograd vs ∂D/∂pⱼ = 2[gⱼ(Σp²+Σg²) − 2pⱼΣpg]/(Σp²+Σg²)² ----
    p = torch.rand(1, 4, 4, 4, dtype=torch.float64, requires_grad=True)
    g = (torch.rand(1, 4, 4, 4, dtype=torch.float64) < 0.5).double()
    d = dice_coefficient(p, g, eps=0.0)  # 对拍论文原式：关闭平滑项
    d.sum().backward()
    pf, gf = p.detach()[0], g[0]
    inter, denom = (pf * gf).sum(), (pf * pf).sum() + (gf * gf).sum()
    closed = 2.0 * (gf * denom - 2.0 * pf * inter) / denom**2  # 论文梯度式，逐体素
    err = (p.grad[0] - closed).abs().max().item()
    assert err < 1e-10, f"autograd 与论文梯度闭式偏差过大：{err}"
    print(f"PASS 论文梯度闭式对拍（max|Δ|={err:.2e}，N=64 体素）")

    # ---- 3) VNet forward + backward + 参数量（输入 32×32×16，可被 16 整除；CPU 数秒） ----
    t0 = time.time()
    model = VNet(in_channels=1)
    n_params = sum(q.numel() for q in model.parameters())
    x = torch.randn(1, 1, 32, 32, 16)
    out = model(x)
    # 输出 (B, 2, D, H, W)：通道 2（背景/前景），空间尺寸与输入一致。
    assert out.shape == (1, 2, 32, 32, 16), f"输出形状应为 (1,2,32,32,16)，实际 {tuple(out.shape)}"
    # 球形前景作为伪真值，走完整 loss->backward。
    grid = torch.stack(torch.meshgrid(*[torch.arange(s, dtype=torch.float32) for s in (32, 32, 16)], indexing="ij"), dim=0)
    sphere = ((grid - torch.tensor([16.0, 16.0, 8.0]).view(3, 1, 1, 1)) ** 2).sum(0) < 25.0
    loss = DiceLoss()(out, sphere.unsqueeze(0).float())
    assert torch.isfinite(loss) and 0.0 <= loss.item() <= 1.0
    loss.backward()
    bad = [n for n, q in model.named_parameters() if q.grad is not None and not torch.isfinite(q.grad).all()]
    assert not bad, f"以下参数梯度含 NaN/Inf: {bad}"
    assert all(q.grad is not None for q in model.parameters())
    print(f"PASS VNet forward/backward（输入 1×1×32×32×16，输出 {tuple(out.shape)}，"
          f"参数量 {n_params:,}，loss={loss.item():.4f}，耗时 {time.time() - t0:.1f}s）")

    # ---- 4) 论文 3.2 节测试规则：二值化输出 ----
    seg = predict_binary(out.detach())
    assert seg.shape == (1, 32, 32, 16) and seg.dtype == torch.uint8
    assert set(seg.unique().tolist()) <= {0, 1}
    print(f"PASS predict_binary（形状 {tuple(seg.shape)}，取值 {sorted(seg.unique().tolist())}，"
          f"前景体素占比 {seg.float().mean().item():.3f}）")

    print(f"参数量（论文未报告，本实现实测）：{n_params:,} = {n_params / 1e6:.1f} M")
    print("PASS 全部自测通过")
