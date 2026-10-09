"""attention-unet —— 注意力门控 U-Net（Attention U-Net，2D/3D 通用）[model]

论文支撑统一四问：
- 哪篇论文用了它：Ozan Oktay, Jo Schlemper 等（Imperial College London / Babylon Health），
  《Attention U-Net: Learning Where to Look for the Pancreas》，MIDL 2018，
  https://arxiv.org/abs/1804.03999
- 目的：在 U-Net 跳跃连接上插入可微的加性注意力门控（AG），用粗尺度上下文自动抑制
  无关背景区域、突出小目标（胰腺），从而省去级联 CNN 的外部器官定位模型。
- 效果：CT-150 多类腹部分割，胰腺 DSC 0.814 -> 0.840（召回 0.806 -> 0.841，p=.005），
  S2S 距离 2.358 -> 1.920 mm；参数仅 5.88M -> 6.40M（约 +8%），推理 0.167 -> 0.179 s。
  训练集缩到 30 例 DSC 0.741 -> 0.767（p=.01）；TCIA CT-82 从头训练 0.815 -> 0.821。
- 结论：AG 以近零开销稳定提升小目标分割灵敏度；高对比度大器官（肾）几乎无增益，
  适用边界为细小、形状变异大、低对比度的目标。

实现说明——关键公式与论文对应：
- 式(1)：q_att = psi^T( sigma1( W_x^T x_i + W_g^T g_i + b_g ) ) + b_psi
  （见 AttentionGate2D/3D.forward 中 theta/phi/psi 三层 1x1 卷积与 ReLU 的组合；
  论文 2.2 节明确 W_x、W_g 为无空间支撑的 1x1(1x1x1) 卷积，sigma1(x)=max(0,x) 即 ReLU）
- 式(2)：alpha_i = sigmoid(q_att)，输出 x_hat_{i,c} = x_{i,c} * alpha_i（论文 2.2 节）；
  论文图 2 说明：注意力系数用 trilinear 插值重采样回 x 的分辨率（2D 对应 bilinear）
- 式(3)：d(x_hat)/d(Phi^{l-1}) = alpha * d(f)/d(Phi) + d(alpha)/d(Phi) * x ——
  这是 autograd 自动完成的反向传播分析，本实现无需手写。
- sigmoid 而非 softmax：论文 2.2 节明确 softmax 使激活过稀、sigmoid 训练收敛更好。

参考的开源实现：
- 官方 https://github.com/ozan-oktay/Attention-Gated-Networks
  - models/layers/grid_attention_layer.py 的 _GridAttentionBlockND._concatenation（门控主干）
  - models/networks/unet_CT_single_att_dsv_3D.py（AG 挂在第 2/3/4 层跳跃连接、第 1 层不挂；
    门控信号取自解码路径更粗尺度特征）
- 本地 /ssd/lx/net_arch/arch_Unet3_2D.py 为普通 2D U-Net，无注意力实现，未采用。

与官方实现不一致处（以论文为准的理由已逐条注明在代码内）：
1) 官方在 x*alpha 之后又接 1x1 conv+BN 的输出变换 W_y；论文式(2) 的输出就是
   x_hat = x * alpha，并无输出变换，故本实现默认不加（output_transform=False，可选开启）。
2) 官方 theta 用 kernel=stride=sub_sample_factor 的 strided conv 把 x 下采样到门控分辨率；
   论文式(1) 的 W_x 是 1x1 卷积，"下采样到门控分辨率"是 "Attention Gates in U-Net
   Model" 小节第三段的独立描述（不属于式(1) 所在段落）。本实现两种都支持：
   sub_sample_factor 全 1 时严格等于式(1) 的纯 1x1 形式；>1 时取官方 strided conv 路径
   （参数更省，等价于"先 1x1 再下采样"的融合实现）。
3) psi 初始化：官方默认 kaiming（其 TORR 变体的 sigmoid 模式用 psi.bias=3.0）；论文原文写明
   "Gating parameters are initialised so that attention gates pass through feature vectors
   at all spatial locations"，故本实现取 psi 权重置零 + bias=3.0（sigmoid(3)≈0.953，
   起始近乎全通，训练初期等价普通 U-Net）。
4) 论文训练另用深监督（DSV）与多门变体，属训练技巧/扩展，本参考实现只含单门主干。

数值稳定说明：
- 唯一的 exp 出现在 torch.sigmoid 内部（式 2 的 sigma2），torch 对其做了饱和截断，
  数值稳定，无需手工 eps；本实现无除法/log/幂运算，BatchNorm 自带 eps=1e-5。

布局约定：2D 张量 (B, C, H, W)，3D 张量 (B, C, D, H, W)；B 批、C 通道、其余为空间维。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "AttentionGate2D",
    "AttentionGate3D",
    "AttentionUNet2D",
    "AttentionUNet3D",
]


def _make_conv(dim: int, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1,
               padding: int = 0, bias: bool = True) -> nn.Module:
    """按维度构造 nn.Conv2d / nn.Conv3d。

    参数：dim=2 或 3；in_ch/out_ch 输入输出通道；kernel_size、stride、padding 为各维相同的 int；
    bias 是否带偏置。
    """
    return (nn.Conv2d if dim == 2 else nn.Conv3d)(
        in_channels=in_ch, out_channels=out_ch,
        kernel_size=kernel_size, stride=stride, padding=padding, bias=bias,
    )


def _conv_bn_relu(dim: int, in_ch: int, out_ch: int) -> nn.Module:
    """构造 Conv(k=3,p=1)-BatchNorm-ReLU 基本块（论文实现细节使用 batch-normalisation）。"""
    return nn.Sequential(
        _make_conv(dim, in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch) if dim == 2 else nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )


class _AttentionGateND(nn.Module):
    """加性注意力门控（论文式(1)(2)）的 2D/3D 公共实现。

    参数：
    - in_channels: 输入特征 x 的通道数 F_l
    - gating_channels: 门控信号 g 的通道数 F_g
    - inter_channels: 中间层维度 F_int；None 时取 max(1, F_l//2)（官方块默认）
    - dim: 2 或 3
    - sub_sample_factor: 每维下采样倍数（int 或 tuple）；全 1 时为论文式(1) 的纯 1x1 形式，
      >1 时与官方一致用 strided conv 把 x 下采样到门控分辨率（"下采样到门控分辨率"
      见论文 "Attention Gates in U-Net Model" 小节第三段）
    - output_transform: 是否在 x*alpha 后接官方的 1x1 conv+BN 输出变换（论文无此层，默认 False）

    张量布局：2D x=(B,F_l,H,W)、g=(B,F_g,H/s,W/s)；3D 同理多一维 D；s 为 sub_sample_factor。
    forward 返回 (x_hat, alpha)：x_hat 与 x 同形，alpha=(B,1,x 的空间形状)，取值 (0,1)。
    """

    def __init__(self, in_channels: int, gating_channels: int, inter_channels: int | None = None,
                 dim: int = 2, sub_sample_factor: int | tuple[int, ...] = 1,
                 output_transform: bool = False) -> None:
        super().__init__()
        assert dim in (2, 3)
        self.dim = dim
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = max(1, in_channels // 2) if inter_channels is None else inter_channels
        ss = (sub_sample_factor,) * dim if isinstance(sub_sample_factor, int) else tuple(sub_sample_factor)
        self.sub_sample_factor = ss

        # 式(1) 的三个线性变换：
        # theta 对应 W_x（bias=False，论文式(1) 的 x 支路无独立偏置项）；kernel=stride=ss、
        # padding=0（官方同款）时同时完成"Attention Gates in U-Net Model" 小节第三段
        # 所述的"输入特征图下采样到门控分辨率"
        self.theta = (nn.Conv2d if dim == 2 else nn.Conv3d)(
            in_channels, self.inter_channels, kernel_size=ss, stride=ss, padding=0, bias=False)
        # phi 对应 W_g 与 b_g（论文式(1) 的门控支路带偏置 b_g）
        self.phi = _make_conv(dim, gating_channels, self.inter_channels, 1, stride=1, padding=0, bias=True)
        # psi 对应 psi^T 与 b_psi（式(1) 最后一层 1x1 卷积，输出单通道标量场）
        self.psi = _make_conv(dim, self.inter_channels, 1, 1, stride=1, padding=0, bias=True)

        # 按论文初始化门控：psi 权重置零 + 偏置 3.0，sigmoid(3)≈0.953，起始在所有空间位置
        # 近乎全通（与官方 TORR 变体 sigmoid 模式的 constant bias=3.0 一致；官方默认 kaiming，
        # 不满足论文"初始化使门控直通"的原句，故以论文为准）
        nn.init.zeros_(self.psi.weight)
        nn.init.constant_(self.psi.bias, 3.0)

        # 可选的官方输出变换 W_y（1x1 conv+BN）；论文式(2) 无此层，默认关闭（见文件头不一致处 1)
        self.output_transform: nn.Module | None = None
        if output_transform:
            self.output_transform = nn.Sequential(
                _make_conv(dim, in_channels, in_channels, 1),
                nn.BatchNorm2d(in_channels) if dim == 2 else nn.BatchNorm3d(in_channels),
            )

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: (B,F_l,空间)，g: (B,F_g,更粗的空间)；返回 (x_hat, alpha)，形状见类 docstring。"""
        theta_x = self.theta(x)  # (B,F_int,空间/s) —— 式(1) 的 W_x^T x
        phi_g = self.phi(g)  # (B,F_int,门控空间) —— 式(1) 的 W_g^T g + b_g
        if phi_g.shape[2:] != theta_x.shape[2:]:
            # 门控分辨率与下采样后的 x 不一致时对齐（官方同款处理；3D trilinear、2D bilinear，
            # 对应论文图 2"grid resampling ... trilinear interpolation"）
            phi_g = F.interpolate(phi_g, size=theta_x.shape[2:], mode="trilinear" if self.dim == 3 else "bilinear")
        f = F.relu(theta_x + phi_g)  # sigma1 —— 式(1)
        q = self.psi(f)  # psi^T(·)+b_psi —— 式(1)
        alpha = torch.sigmoid(q)  # sigma2 —— 式(2)；torch.sigmoid 内部饱和截断，数值稳定
        if alpha.shape[2:] != x.shape[2:]:
            # 注意力系数重采样回 x 的分辨率（论文图 2）
            alpha = F.interpolate(alpha, size=x.shape[2:], mode="trilinear" if self.dim == 3 else "bilinear")
        x_hat = x * alpha  # x_hat_{i,c} = x_{i,c} * alpha_i —— 论文 2.2 节输出定义
        if self.output_transform is not None:
            x_hat = self.output_transform(x_hat)
        return x_hat, alpha


class AttentionGate2D(_AttentionGateND):
    """2D 加性注意力门控（论文式(1)(2)）。

    x 布局 (B,F_l,H,W)，g 布局 (B,F_g,H/s,W/s)；返回 (x_hat, alpha)。
    参数含义见 _AttentionGateND。
    """

    def __init__(self, in_channels: int, gating_channels: int, inter_channels: int | None = None,
                 sub_sample_factor: int | tuple[int, ...] = 1, output_transform: bool = False) -> None:
        super().__init__(in_channels, gating_channels, inter_channels, dim=2,
                         sub_sample_factor=sub_sample_factor, output_transform=output_transform)


class AttentionGate3D(_AttentionGateND):
    """3D 加性注意力门控（论文式(1)(2)）。

    x 布局 (B,F_l,D,H,W)，g 布局 (B,F_g,D/s,H/s,W/s)；返回 (x_hat, alpha)。
    参数含义见 _AttentionGateND。
    """

    def __init__(self, in_channels: int, gating_channels: int, inter_channels: int | None = None,
                 sub_sample_factor: int | tuple[int, ...] = 1, output_transform: bool = False) -> None:
        super().__init__(in_channels, gating_channels, inter_channels, dim=3,
                         sub_sample_factor=sub_sample_factor, output_transform=output_transform)


class _AttentionUNetND(nn.Module):
    """标准 U-Net + 跳跃连接注意力门控（论文图 1 的 Attention U-Net 主干）。

    参数：
    - in_channels: 输入模态通道数；n_classes: 输出类别数（logits，未过 softmax）
    - base_filters: 第 1 层通道数，第 l 层为 base_filters * 2**(l-1)
    - levels: 编码/解码层数（>=2）；levels=1 时唯一的跳跃连接是第 1 条（论文规定
      第 1 条不挂门控），网络将不含任何注意力门，故要求 levels>=2
    - dim: 2 或 3
    - sub_sample_factor: 传给各注意力门（见 _AttentionGateND）

    默认宽度说明：默认 base_filters=32、levels=4 不对应论文表 1 规模——论文官方
    代码 feature_scale=4（通道 [16,32,64,256] 递进）；本实现默认 3D 实测 23.442M
    参数，base_filters=16 时 5.863M 接近论文 6.40M。

    结构对应论文图 1：编码器逐级 2 倍下采样；第 l(>=2) 条跳跃连接先过 AG（门控信号取自
    解码路径上一级更粗特征）再与上采样特征拼接；第 1 条跳跃连接不加门控——论文
    "Attention Gates in U-Net Model" 小节第三段："low-level feature-maps, i.e.
    the first skip connections, are not used in the gating function"（官方网络同样不挂）。
    """

    def __init__(self, in_channels: int, n_classes: int, base_filters: int = 32, levels: int = 4,
                 dim: int = 2, sub_sample_factor: int | tuple[int, ...] = 2) -> None:
        super().__init__()
        assert levels >= 2
        self.dim = dim
        self.levels = levels
        filt = [base_filters * 2 ** l for l in range(levels)]
        pool = nn.MaxPool2d(2) if dim == 2 else nn.MaxPool3d(2)

        self.enc_convs = nn.ModuleList()
        self.pools = nn.ModuleList()
        ch = in_channels
        for f in filt:
            self.enc_convs.append(nn.Sequential(_conv_bn_relu(dim, ch, f), _conv_bn_relu(dim, f, f)))
            self.pools.append(pool)
            ch = f
        self.bottleneck = nn.Sequential(_conv_bn_relu(dim, filt[-1], filt[-1] * 2),
                                        _conv_bn_relu(dim, filt[-1] * 2, filt[-1] * 2))

        # 各级跳跃连接的注意力门（跳过第 1 层，levels-1 个）。门控通道对应论文"门控信号取自
        # 更粗尺度"（官方同款）：skip l 的门控信号 = 解码侧 level l+1 的特征，通道数
        # l<levels-1 时为 filt[l+1]；最深层（l=levels-1）为瓶颈输出 filt[-1]*2。
        self.gates = nn.ModuleList(
            _AttentionGateND(filt[k + 1],
                             filt[k + 2] if k < levels - 2 else filt[k + 1] * 2,
                             inter_channels=filt[k + 1], dim=dim,
                             sub_sample_factor=sub_sample_factor)
            for k in range(levels - 1))

        self.ups = nn.ModuleList()  # 上采样：ConvTranspose(k=2,s=2) + 2 个 Conv-BN-ReLU
        for l in range(levels - 1, -1, -1):
            skip_ch = filt[l]
            up_ch = filt[l] * 2 if l == levels - 1 else filt[l + 1]  # 输入=上采样前的解码特征
            upconv = (nn.ConvTranspose2d(up_ch, skip_ch, 2, stride=2) if dim == 2
                      else nn.ConvTranspose3d(up_ch, skip_ch, 2, stride=2))
            self.ups.append(nn.ModuleDict({
                "up": upconv,
                "conv": nn.Sequential(_conv_bn_relu(dim, skip_ch * 2, skip_ch), _conv_bn_relu(dim, skip_ch, skip_ch)),
            }))

        self.final = _make_conv(dim, filt[0], n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """输入 2D (B,in_channels,H,W) / 3D (B,in_channels,D,H,W)；返回 logits，空间形状不变。"""
        skips = []
        h = x
        for i, conv in enumerate(self.enc_convs):
            h = conv(h)
            skips.append(h)
            h = self.pools[i](h)
        dec = self.bottleneck(h)  # 最粗尺度上下文，作为最深层跳跃连接的门控信号来源
        gate_idx = len(self.gates) - 1  # 从最深层跳跃连接往浅层走
        for l, up in enumerate(self.ups):
            skip = skips[self.levels - 1 - l]  # ups 按最深→最浅构建，skips 是浅→深序
            if gate_idx >= 0:
                # 论文：门控信号取自更粗尺度（此处为上采样前的解码侧特征 dec，官方同款）；
                # 第 1 层（最浅）跳跃连接不挂门控——门耗尽后 gate_idx<0 自然跳过
                # （论文 "Attention Gates in U-Net Model" 小节第三段）
                skip, _ = self.gates[gate_idx](skip, dec)
                gate_idx -= 1
            dec = up["conv"](torch.cat([up["up"](dec), skip], dim=1))
        return self.final(dec)


class AttentionUNet2D(_AttentionUNetND):
    """2D Attention U-Net（论文图 1）。

    输入 (B,in_channels,H,W)，输出 logits (B,n_classes,H,W)；空间形状需能被 2**levels 整除。
    """

    def __init__(self, in_channels: int, n_classes: int, base_filters: int = 32, levels: int = 4,
                 sub_sample_factor: int | tuple[int, ...] = 2) -> None:
        super().__init__(in_channels, n_classes, base_filters, levels, dim=2, sub_sample_factor=sub_sample_factor)


class AttentionUNet3D(_AttentionUNetND):
    """3D Attention U-Net（论文主实验所用形态）。

    输入 (B,in_channels,D,H,W)，输出 logits (B,n_classes,D,H,W)；空间形状需能被 2**levels 整除。
    """

    def __init__(self, in_channels: int, n_classes: int, base_filters: int = 32, levels: int = 4,
                 sub_sample_factor: int | tuple[int, ...] = 2) -> None:
        super().__init__(in_channels, n_classes, base_filters, levels, dim=3, sub_sample_factor=sub_sample_factor)


if __name__ == "__main__":
    # 自测：仅 CPU，固定随机种子，断言形状/数值范围/梯度有限，几秒内完成
    torch.manual_seed(42)

    # 1) 2D 门控：形状、alpha 范围、初始化直通（论文"初始化使门控在所有位置直通"）
    x2 = torch.randn(2, 8, 32, 32)
    g2 = torch.randn(2, 4, 16, 16)
    gate2 = AttentionGate2D(in_channels=8, gating_channels=4, inter_channels=8, sub_sample_factor=2)
    y2, a2 = gate2(x2, g2)
    assert y2.shape == x2.shape, f"2D 门控输出形状错误: {y2.shape}"
    assert a2.shape == (2, 1, 32, 32), f"2D alpha 形状错误: {a2.shape}"
    assert torch.isfinite(y2).all(), "2D 门控输出出现 NaN/Inf"
    assert 0.0 < float(a2.detach().min()) and float(a2.detach().max()) < 1.0, "alpha 未落在 (0,1)"
    assert float(a2.detach().min()) > 0.9, f"初始门控应近乎全通(sigmoid(3)≈0.953)，实测 min={float(a2.detach().min()):.4f}"
    print(f"PASS 2D 门控前向：形状 {tuple(y2.shape)}，alpha∈({float(a2.detach().min()):.4f},{float(a2.detach().max()):.4f})，初始化直通")

    # 2) 3D 门控：同上 + 反向
    x3 = torch.randn(2, 8, 16, 16, 16)
    g3 = torch.randn(2, 4, 8, 8, 8)
    gate3 = AttentionGate3D(in_channels=8, gating_channels=4, inter_channels=8, sub_sample_factor=2)
    y3, a3 = gate3(x3, g3)
    assert y3.shape == x3.shape and a3.shape == (2, 1, 16, 16, 16), "3D 门控形状错误"
    assert torch.isfinite(y3).all() and float(a3.detach().min()) > 0.9, "3D 门控数值异常"
    (y3.square().mean()).backward()
    assert all(torch.isfinite(p.grad).all() for p in gate3.parameters() if p.grad is not None), "3D 门控梯度 NaN/Inf"
    print("PASS 3D 门控前向+反向：形状/范围/梯度有限")

    # 3) 2D Attention U-Net：forward+backward、参数量
    net2 = AttentionUNet2D(in_channels=1, n_classes=3, base_filters=8, levels=3, sub_sample_factor=2)
    out2 = net2(torch.randn(2, 1, 64, 64))
    assert out2.shape == (2, 3, 64, 64), f"2D 网络输出形状错误: {out2.shape}"
    assert torch.isfinite(out2).all(), "2D 网络输出 NaN/Inf"
    out2.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in net2.parameters() if p.grad is not None), "2D 网络梯度 NaN/Inf"
    n2 = sum(p.numel() for p in net2.parameters())
    print(f"PASS 2D Attention U-Net forward+backward：输出 {tuple(out2.shape)}，参数量 {n2/1e6:.3f} M")

    # 4) 3D Attention U-Net：forward+backward、参数量
    net3 = AttentionUNet3D(in_channels=1, n_classes=3, base_filters=8, levels=3, sub_sample_factor=2)
    out3 = net3(torch.randn(1, 1, 32, 32, 32))
    assert out3.shape == (1, 3, 32, 32, 32), f"3D 网络输出形状错误: {out3.shape}"
    assert torch.isfinite(out3).all(), "3D 网络输出 NaN/Inf"
    out3.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in net3.parameters() if p.grad is not None), "3D 网络梯度 NaN/Inf"
    n3 = sum(p.numel() for p in net3.parameters())
    print(f"PASS 3D Attention U-Net forward+backward：输出 {tuple(out3.shape)}，参数量 {n3/1e6:.3f} M")

    print("PASS 全部自测通过")
