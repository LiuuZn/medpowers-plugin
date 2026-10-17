"""nnunet-resenc — nnU-Net ResEnc 预设：3D 残差编码器 U-Net 参考实现 [model]

资产卡：assets/models/nnunet-resenc.md

论文支撑（统一四问）：
- 哪篇论文用了它：F. Isensee、T. Wald、C. Ulrich、M. Rokuss、T. Wald 等（德国癌症研究中心
  DKFZ），《nnU-Net Revisited: A Call for Rigorous Validation in 3D Medical Image
  Segmentation》，MICCAI 2024。链接：https://arxiv.org/abs/2404.09556。所读版本：2026-10-09
  核对 arXiv HTML 全文基准数字；预设使用口径另行核对官方文档
  https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md（main 分支，
  2026-10-09 抓取），以及本机 nnunetv2 2.7.0 的 residual_encoder_unet_planners.py 与
  dynamic_network_architectures 包源码（/ssd/lx/totalsegmentator/.venv）。
- 目的：论文一面指出 2022–2024 年"超越 nnU-Net"文献中的系统性验证缺陷（弱基线、不公平
  训练配方、小数据集、忽略算力开销），一面把 nnU-Net 自身推向现代 GPU 的算力上限——
  编码器换成残差块，并按显存预算给出 ResEnc-M/L/XL 三个预设。
- 效果：5 折交叉验证 DSC（对同时训练的原版 nnU-Net，数字自官方文档核对）：KiTS
  86.04→86.79(M)/88.17(L)/88.67(XL)；AMOS 88.64→88.77/89.41/89.68；LiTS
  80.09→80.75/81.60/81.19——大数据集增益明显，小数据集（BTCV/ACDC/BraTS）增益≈0。
  代价（A100 40GB 单卡，官方文档口径）：M 9.1GB/约 12h、L 22.7GB/约 35h、XL 36.6GB/约 66h。
- 结论：验证严谨的前提下，配置得当的 CNN U-Net 仍优于被复核的多数新架构；官方文档明确
  "We recommend nnU-Net ResEnc L as the new default nnU-Net configuration"，并要求新方法与
  同显存/算力预算的 ResEnc 变体对比。

实现说明（论文为"预设/配方"而非单一固定网络，本文件实现其网络部分）：
- 网络对应官方 dynamic_network_architectures 的 ResidualEncoderUNet =
  ResidualEncoder + UNetDecoder（本机路径：
  /ssd/lx/totalsegmentator/.venv/lib/python3.11/site-packages/dynamic_network_architectures/
  architectures/unet.py:111、building_blocks/residual_encoders.py:13、
  building_blocks/residual.py:13（BasicBlockD）、building_blocks/unet_decoder.py:14）。
- 编码器：stem（1 个 3×3×3 卷积 + IN + LeakyReLU，步长 1）+ n_stages 个残差级；每级
  n_blocks_per_stage 个 ResNet-D 式残差块（论文称 residual encoder；块数口径——构造
  默认 2 是 dynamic_network_architectures 的 ResidualEncoder 默认值，而 nnU-Net ResEnc
  预设实际用**固定常数序列 (1,3,4,6,6,6,…)**（M/L/XL 共用，nnunetv2
  resencUNet_planner.py 的 UNet_blocks_per_stage_encoder），复现预设时应显式传入该
  序列——同 features 下参数量约为每级 2 块的 2.4 倍，见资产卡"量级"）；
  ResNet-D 出自 He et al., "Bag of Tricks for Image Classification with Convolutional
  Neural Networks", CVPR 2019——官方代码注释注明遵循该设计）：主路 conv1（步长 s）→IN→
  LeakyReLU→conv2→IN，短路路 avgpool(s)→1×1×1 卷积→IN（仅在需要降采样/升通道时），
  相加后 LeakyReLU。
- 解码器：普通卷积解码器（不是残差解码器，官方即如此）：每级转置卷积（kernel=stride）
  上采样 → 与同级 shortcut 拼接 → n_conv_per_stage_decoder（nnU-Net 默认 1）个
  conv→IN→LeakyReLU 块 → 1×1×1 分割头（bias=True）。
- 卷积全部 same padding（奇数核 padding=(k−1)/2，输出与输入同形；nnU-Net 规划只产生
  奇数核），这是现代 nnU-Net 谱系与经典 valid 卷积 U-Net 的关键区别之一。
- 预设的本质（重要）：ResEnc-M/L/XL 不是固定拓扑，而是把"显存目标"（M=8GB、L=24GB、
  XL=40GB，规划值；实测见 RES_ENC_PRESET_INFO）交给 nnU-Net 的 ResEncUNetPlanner，按
  数据集 spacing 自动算 patch/池化（可各向异性）/通道/批大小，并取消"单批不超过数据集
  0.05"的旧上限（官方文档）。本文件的 TYPICAL_3D_FULLRES_CONFIG 是 planner 文档字符串里
  给出的典型 3D fullres 规划结果（6 级、(32,64,128,256,320,320)、每级 2 块、步长
  (1,2,2,2,2,2)、解码每级 1 卷积），仅作示例（块数 2 是官方文档示例的取值，非预设常数，
  见上）；完整增益依赖 nnU-Net 框架（预处理、增广、
  训练配方、滑窗推理），不要用单网络复刻冒充 ResEnc-L 基线。
- 初始化：He 正态（kaiming_normal_，a=1e-2，LeakyReLU 斜率匹配；偏置置 0）+ 残差相加前
  最后一个 norm 的仿射权重置 0（对应官方 ResidualEncoderUNet.initialize 的
  init_last_bn_before_add_to_0）——初始时残差块为恒等映射，训练更稳。
- 与官方实现不一致处（有意偏离，逐条）：
  1. 深监督输出的多尺度上采样由调用方在损失里做——与官方 nnU-Net 训练器一致，本文件不
     内置上采样；
  2. 官方 planner 按 spacing 自动规划各向异性步长，本实现把 strides 作为显式参数传入；
  3. 未实现 dropout（nnU-Net 3D fullres 默认 p=0）与梯度检查点（框架层职责）。
- 张量布局：输入/输出 3D 布局 (B, C, D, H, W)；logits 未过 softmax。deep_supervision=True
  时 forward 返回 logits 列表 [全分辨率, 1/2, 1/4, ...]（由粗到细反序后，全分辨率在前）。
- 数值稳定：InstanceNorm3d 内部对 var+eps 开方（eps=1e-5，PyTorch 默认），无裸 log/exp；
  转置卷积 kernel=stride 保证输出尺寸精确为输入的 stride 倍，无 padding 歧义。自测断言
  输出与梯度无 NaN/Inf。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "BasicResidualBlock3D",
    "ResidualEncoder3D",
    "PlainConvDecoder3D",
    "ResidualEncoderUNet3D",
    "RES_ENC_PRESET_INFO",
    "TYPICAL_3D_FULLRES_CONFIG",
]

# ResEnc 预设事实（官方文档 resenc_presets.md 口径，2026-10-09 核对）：gpu_memory_target_gb
# 是传给 planner 的规划目标，measured_vram_gb 是官方在 A100 40GB 上实训的显存占用。
RES_ENC_PRESET_INFO: dict[str, dict[str, object]] = {
    "M": {
        "gpu_memory_target_gb": 8,
        "measured_vram_gb": 9.1,
        "train_hours_a100": 12,
        "gpu_recommendation": "9–11GB 显存即可；2×4090 轻松",
    },
    "L": {
        "gpu_memory_target_gb": 24,
        "measured_vram_gb": 22.7,
        "train_hours_a100": 35,
        "gpu_recommendation": "官方要求 24GB 卡（RTX 4090 标称 24GB 可用，紧张时框架自动缩 patch）",
    },
    "XL": {
        "gpu_memory_target_gb": 40,
        "measured_vram_gb": 36.6,
        "train_hours_a100": 66,
        "gpu_recommendation": "需 40GB 级（A100）；2×4090 单卡装不下",
    },
}

# 典型 3D fullres 规划结果（取自本机 nnunetv2 2.7.0 residual_encoder_unet_planners.py 的
# 模块级示例，第 299–302 行）：仅示例——真实规划随数据集 spacing 变化。
TYPICAL_3D_FULLRES_CONFIG: dict[str, object] = {
    "n_stages": 6,
    "features_per_stage": (32, 64, 128, 256, 320, 320),
    "kernel_sizes": 3,
    "strides": (1, 2, 2, 2, 2, 2),
    "n_blocks_per_stage": 2,
    "n_conv_per_stage_decoder": 1,
}

# LeakyReLU 负斜率与 He 初始化的 a 参数（nnU-Net 默认 1e-2，保持一致）
_LEAKY_SLOPE = 1e-2


class _ConvNormAct(nn.Module):
    """conv3d → InstanceNorm3d(affine) →（可选）LeakyReLU，nnU-Net 风格的卷积单元。

    参数：
        in_channels / out_channels: 输入/输出通道数。
        kernel_size: 卷积核大小（int 或三元组）。
        stride: 步长（int 或三元组），默认 1。
        bias: 卷积是否带偏置；nnU-Net 默认 False（后面有 affine 归一化，偏置冗余）。
        use_act: False 时只做 conv+norm、不带激活（对应官方残差块第二卷积与短路投影卷积
            ——它们在相加后才过激活）。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        bias: bool = False,
        use_act: bool = True,
    ) -> None:
        super().__init__()
        ks = kernel_size if isinstance(kernel_size, (tuple, list)) else (kernel_size, kernel_size, kernel_size)
        if any(k % 2 == 0 for k in ks):
            raise ValueError(f"nnU-Net 规划只产生奇数卷积核（same padding 无歧义），得到 {kernel_size}")
        # same padding：奇数核 (k-1)/2 使输出与输入同形（nnU-Net 全 padded 卷积，非经典 U-Net 的 valid）
        padding = tuple((k - 1) // 2 for k in ks)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=bias)
        self.norm = nn.InstanceNorm3d(out_channels, affine=True, track_running_stats=False)
        self.act = nn.LeakyReLU(_LEAKY_SLOPE, inplace=True) if use_act else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_channels, D, H, W) → (B, out_channels, D', H', W')。"""
        x = self.norm(self.conv(x))
        if self.act is not None:
            x = self.act(x)
        return x


class BasicResidualBlock3D(nn.Module):
    """ResNet-D 式 3D 残差块（对应官方 dynamic_network_architectures 的 BasicBlockD）。

    主路：conv1（步长 stride）→IN→LeakyReLU → conv2（步长 1）→IN。
    短路：需要降采样时先 AvgPool3d(kernel=stride, stride=stride)，需要变通道时再接
    1×1×1 卷积+IN（ResNet-D 的"池化代替步长卷积"技巧，官方注释明确）。
    输出：LeakyReLU(主路 + 短路)。

    参数：
        in_channels / out_channels: 输入/输出通道数。
        kernel_size: 主路两个卷积的核大小（官方 nnU-Net 规划为 3）。
        stride: 步长，只作用于 conv1 与短路（int 或三元组；nnU-Net 编码器各级为 1 或 2）。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int] = 3,
        stride: int | tuple[int, int, int] = 1,
    ) -> None:
        super().__init__()
        self.conv1 = _ConvNormAct(in_channels, out_channels, kernel_size, stride, use_act=True)
        self.conv2 = _ConvNormAct(out_channels, out_channels, kernel_size, 1, use_act=False)
        self.out_act = nn.LeakyReLU(_LEAKY_SLOPE, inplace=True)

        stride_t = stride if isinstance(stride, (tuple, list)) else (stride, stride, stride)
        has_stride = any(s != 1 for s in stride_t)
        needs_proj = in_channels != out_channels
        if has_stride or needs_proj:
            ops: list[nn.Module] = []
            if has_stride:
                ops.append(nn.AvgPool3d(kernel_size=stride_t, stride=stride_t))
            if needs_proj:
                ops.append(_ConvNormAct(in_channels, out_channels, 1, 1, use_act=False))
            self.skip: nn.Module = nn.Sequential(*ops)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_channels, D, H, W) → (B, out_channels, D/stride, H/stride, W/stride)。"""
        return self.out_act(self.conv2(self.conv1(x)) + self.skip(x))


def _normalize_per_stage(
    value: int | tuple[int, ...] | list[int],
    n_stages: int,
    name: str,
) -> list[int | tuple[int, ...]]:
    """把标量配置广播成每级一份；序列配置则校验长度。"""
    if isinstance(value, int):
        return [value] * n_stages
    value = tuple(value)
    if len(value) != n_stages:
        raise ValueError(f"{name} 需要标量或长度 {n_stages} 的序列，得到 {value}")
    return list(value)


class ResidualEncoder3D(nn.Module):
    """残差编码器：stem + n_stages 个残差级（对应官方 ResidualEncoder，pool_type='conv'
    即降采样进块内卷积步长，不用池化算子）。

    forward 返回各级输出（shortcut）列表 [s0, s1, ..., s_{n-1}]，末级为瓶颈特征。

    参数：
        input_channels: 输入通道数（CT/MRI 通常 1）。
        n_stages: 分辨率级数。
        features_per_stage: 每级通道数（长度 n_stages；nnU-Net 规划为翻倍后在 320/512 封顶）。
        kernel_sizes: 每级卷积核大小，标量广播或长度 n_stages。
        strides: 每级步长（降采样倍率），标量广播或长度 n_stages；第一级应为 1（降采样
            放在级间，与 nnU-Net 规划一致）。
        n_blocks_per_stage: 每级残差块数（默认 2 = 官方库默认值；nnU-Net ResEnc 预设
            固定序列 (1,3,4,6,6,6)，复现预设时按级显式传入），标量广播或长度 n_stages。
    """

    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: tuple[int, ...] | list[int],
        kernel_sizes: int | tuple[int, ...] | list[int] = 3,
        strides: int | tuple[int, ...] | list[int] = 1,
        n_blocks_per_stage: int | tuple[int, ...] | list[int] = 2,
    ) -> None:
        super().__init__()
        if len(features_per_stage) != n_stages:
            raise ValueError(f"features_per_stage 需要长度 {n_stages}，得到 {len(features_per_stage)}")
        kernels = _normalize_per_stage(kernel_sizes, n_stages, "kernel_sizes")
        strides_l = _normalize_per_stage(strides, n_stages, "strides")
        n_blocks = _normalize_per_stage(n_blocks_per_stage, n_stages, "n_blocks_per_stage")
        for s in strides_l:
            st = s if isinstance(s, (tuple, list)) else (s, s, s)
            if any(t < 1 for t in st):
                raise ValueError(f"strides 不允许 <1，得到 {s}")
        first = strides_l[0]
        first_t = first if isinstance(first, (tuple, list)) else (first, first, first)
        if any(t != 1 for t in first_t):
            raise ValueError("strides[0] 必须为 1（stem 步长 1，降采样从级间开始，与 nnU-Net 规划一致）")

        self.strides: list[int | tuple[int, ...]] = strides_l
        self.output_channels: list[int] = list(features_per_stage)
        self.kernel_sizes: list[int | tuple[int, ...]] = kernels

        self.stem = _ConvNormAct(input_channels, features_per_stage[0], kernels[0], 1)
        stages: list[nn.Module] = []
        c_in = features_per_stage[0]
        for s in range(n_stages):
            blocks: list[nn.Module] = [
                BasicResidualBlock3D(c_in, int(features_per_stage[s]), kernels[s], strides_l[s])
            ]
            for _ in range(int(n_blocks[s]) - 1):
                blocks.append(BasicResidualBlock3D(int(features_per_stage[s]), int(features_per_stage[s]), kernels[s], 1))
            stages.append(nn.Sequential(*blocks))
            c_in = int(features_per_stage[s])
        self.stages = nn.ModuleList(stages)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """(B, input_channels, D, H, W) → [n_stages 个 (B, features[i], D/sᵢ, H/sᵢ, W/sᵢ)]。"""
        x = self.stem(x)
        skips: list[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            skips.append(x)
        return skips


class PlainConvDecoder3D(nn.Module):
    """普通卷积解码器（对应官方 UNetDecoder）：转置卷积上采样 → 拼接 shortcut →
    卷积块 → 1×1×1 分割头。

    参数：
        encoder: 已建好的 ResidualEncoder3D（借用其通道/核/步长配置）。
        num_classes: 输出类别数。
        n_conv_per_stage_decoder: 每个解码级的卷积块数（nnU-Net 默认 1），标量或长度
            n_stages-1 的序列。
        deep_supervision: True 时 forward 返回各尺度 logits 列表（全分辨率在前）；False 时
            只返回全分辨率 logits。分割头无条件全部构建（与官方一致，便于加载
            deep_supervision 训练的权重做纯推理）。
    """

    def __init__(
        self,
        encoder: ResidualEncoder3D,
        num_classes: int,
        n_conv_per_stage_decoder: int | tuple[int, ...] | list[int] = 1,
        deep_supervision: bool = False,
    ) -> None:
        super().__init__()
        self.deep_supervision = deep_supervision
        n_stages = len(encoder.output_channels)
        n_dec = n_stages - 1
        n_conv = _normalize_per_stage(n_conv_per_stage_decoder, n_dec, "n_conv_per_stage_decoder")

        transpconvs: list[nn.Module] = []
        stages: list[nn.Module] = []
        seg_layers: list[nn.Module] = []
        for s in range(1, n_stages):
            feats_below = encoder.output_channels[-s]      # 更粗一级（来源）通道
            feats_here = encoder.output_channels[-(s + 1)]  # 本级（更细）通道
            stride = encoder.strides[-s]
            stride_t = stride if isinstance(stride, (tuple, list)) else (stride, stride, stride)
            # kernel=stride 的转置卷积使输出尺寸精确为输入 × stride（官方同款做法）
            transpconvs.append(
                nn.ConvTranspose3d(feats_below, feats_here, kernel_size=stride_t, stride=stride_t, bias=False)
            )
            kernel = encoder.kernel_sizes[-(s + 1)]
            blocks: list[nn.Module] = [
                _ConvNormAct(2 * feats_here, feats_here, kernel, 1),
            ]
            for _ in range(int(n_conv[s - 1]) - 1):
                blocks.append(_ConvNormAct(feats_here, feats_here, kernel, 1))
            stages.append(nn.Sequential(*blocks))
            seg_layers.append(nn.Conv3d(feats_here, num_classes, kernel_size=1, bias=True))

        self.transpconvs = nn.ModuleList(transpconvs)
        self.stages = nn.ModuleList(stages)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips: list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor]:
        """skips 为编码器各级输出（瓶颈在末位）。

        返回：deep_supervision=False 时单个 (B, num_classes, D, H, W) logits；
        True 时 [全分辨率, 1/2, 1/4, ...] 的 logits 列表（各尺度不做上采样，由调用方处理）。
        """
        h = skips[-1]
        seg_outputs: list[torch.Tensor] = []
        for s in range(len(self.stages)):
            h = self.transpconvs[s](h)
            h = torch.cat((h, skips[-(s + 2)]), dim=1)
            h = self.stages[s](h)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](h))
            elif s == len(self.stages) - 1:
                seg_outputs.append(self.seg_layers[-1](h))
        seg_outputs = seg_outputs[::-1]  # 反转成全分辨率在前
        if not self.deep_supervision:
            return seg_outputs[0]
        return seg_outputs


class ResidualEncoderUNet3D(nn.Module):
    """nnU-Net ResEnc 预设的网络本体：残差编码器 + 普通解码器（3D）。

    布局：输入 (B, input_channels, D, H, W)，输出 (B, num_classes, D, H, W) 的未归一化
    logits（deep_supervision=True 时见 PlainConvDecoder3D.forward）。输入每维必须能被
    总降采样倍率整除（如 5 次步长 2 池化需被 32 整除）。

    注意：本类只是 ResEnc 预设的"网络部分"参考实现；ResEnc-M/L/XL 的完整收益来自
    nnU-Net 框架的自动规划 + 训练配方 + 推理流水，实际训练请用 nnU-Net v2
    （-pl nnUNetPlannerResEncM/L/XL，-p nnUNetResEncUNet{M,L,XL}Plans），见资产卡。

    参数：
        input_channels: 输入通道数。
        n_stages: 分辨率级数。
        features_per_stage: 每级通道数（长度 n_stages）。
        num_classes: 输出类别数。
        kernel_sizes / strides / n_blocks_per_stage / n_conv_per_stage_decoder: 同编码器/
            解码器说明；典型 3D 规划见 TYPICAL_3D_FULLRES_CONFIG。
        deep_supervision: 是否输出多尺度深监督 logits（训练用；nnU-Net/MedNeXt 论文均用）。
        initialize_he: True 时在构造末尾执行 he_initialize（He 初始化 + 残差支路末级 norm
            仿射权重置 0，官方配方）。
    """

    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: tuple[int, ...] | list[int],
        num_classes: int,
        kernel_sizes: int | tuple[int, ...] | list[int] = 3,
        strides: int | tuple[int, ...] | list[int] = 1,
        n_blocks_per_stage: int | tuple[int, ...] | list[int] = 2,
        n_conv_per_stage_decoder: int | tuple[int, ...] | list[int] = 1,
        deep_supervision: bool = False,
        initialize_he: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = ResidualEncoder3D(
            input_channels, n_stages, features_per_stage, kernel_sizes, strides, n_blocks_per_stage
        )
        self.decoder = PlainConvDecoder3D(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)
        if initialize_he:
            he_initialize(self)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | list[torch.Tensor]:
        """(B, input_channels, D, H, W) → logits（形状含义见类 docstring）。"""
        if x.dim() != 5:
            raise ValueError(f"期望 (B, C, D, H, W) 5 维输入，得到 shape={tuple(x.shape)}")
        # 每维必须能被该轴累积步长整除，否则转置卷积无法精确还原尺寸
        for axis in range(3):
            factor = 1
            for s in self.encoder.strides:
                st = s if isinstance(s, (tuple, list)) else (s, s, s)
                factor *= st[axis]
            if x.shape[2 + axis] % factor != 0:
                raise ValueError(
                    f"输入第 {axis} 维 {x.shape[2 + axis]} 不能被累积步长 {factor} 整除；"
                    f"请 pad 到整除（nnU-Net 预处理保证这一点）"
                )
        skips = self.encoder(x)
        return self.decoder(skips)


def he_initialize(module: nn.Module) -> None:
    """nnU-Net 官方初始化：所有卷积权重 kaiming_normal_（a=1e-2，匹配 LeakyReLU 斜率）、
    偏置置 0；并把每个残差块主路第二个卷积（相加前最后一级）的 InstanceNorm 仿射权重
    置 0——初始时残差支路输出全 0，块为恒等映射（对应官方 init_last_bn_before_add_to_0，
    作用是把"残差网络从恒等起步"推广到带归一化的实现）。
    """
    for m in module.modules():
        if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
            nn.init.kaiming_normal_(m.weight, a=_LEAKY_SLOPE, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    for m in module.modules():
        if isinstance(m, BasicResidualBlock3D):
            nn.init.zeros_(m.conv2.norm.weight)


if __name__ == "__main__":
    torch.manual_seed(0)  # 固定随机种子，保证自测可复现

    # ---- 1) 小配置端到端 forward（各向同性步长）：输出与输入同形、深监督关闭
    net = ResidualEncoderUNet3D(
        input_channels=1,
        n_stages=4,
        features_per_stage=(4, 8, 16, 32),
        num_classes=3,
        strides=(1, 2, 2, 2),
        n_blocks_per_stage=2,
        n_conv_per_stage_decoder=1,
        deep_supervision=False,
    )
    net.eval()
    with torch.no_grad():
        out = net(torch.randn(1, 1, 32, 32, 32))
    assert isinstance(out, torch.Tensor) and out.shape == (1, 3, 32, 32, 32), f"形状不符: {tuple(out.shape)}"
    assert torch.isfinite(out).all(), "自检 1 输出含 NaN/Inf"
    print("PASS 1 端到端 forward: (1,1,32,32,32) → (1,3,32,32,32) 同形输出，全部有限")

    # ---- 2) 深监督：返回多尺度 logits 列表，全分辨率在前
    net_ds = ResidualEncoderUNet3D(
        input_channels=1,
        n_stages=4,
        features_per_stage=(4, 8, 16, 32),
        num_classes=3,
        strides=(1, 2, 2, 2),
        deep_supervision=True,
    )
    with torch.no_grad():
        outs = net_ds(torch.randn(1, 1, 32, 32, 32))
    assert isinstance(outs, list) and len(outs) == 3, f"深监督应有 3 个输出，得到 {len(outs)}"
    for scale, o in zip((1, 2, 4), outs):
        expect = (1, 3, 32 // scale, 32 // scale, 32 // scale)
        assert o.shape == expect, f"深监督尺度 {scale} 形状不符: {tuple(o.shape)}"
        assert torch.isfinite(o).all(), f"深监督尺度 {scale} 输出含 NaN/Inf"
    print("PASS 2 深监督: 3 个多尺度 logits（1, 1/2, 1/4），全分辨率在前，全部有限")

    # ---- 3) 各向异性步长（CT z 向层厚大的常见规划）：形状精确还原
    net_aniso = ResidualEncoderUNet3D(
        input_channels=1,
        n_stages=3,
        features_per_stage=(4, 8, 16),
        num_classes=2,
        strides=((1, 1, 1), (1, 1, 2), (2, 2, 2)),
        n_blocks_per_stage=2,
    )
    with torch.no_grad():
        out_aniso = net_aniso(torch.randn(1, 1, 8, 8, 16))
    assert out_aniso.shape == (1, 2, 8, 8, 16), f"各向异性形状不符: {tuple(out_aniso.shape)}"
    assert torch.isfinite(out_aniso).all(), "各向异性输出含 NaN/Inf"
    print("PASS 3 各向异性步长: (1,1,8,8,16)（strides 含 (1,1,2)/(2,2,2)）→ 同形输出，全部有限")

    # ---- 4) forward+backward：所有参数梯度有限；官方初始化生效
    net_tr = ResidualEncoderUNet3D(
        input_channels=1,
        n_stages=4,
        features_per_stage=(4, 8, 16, 32),
        num_classes=3,
        strides=(1, 2, 2, 2),
        deep_supervision=True,
    )
    x = torch.randn(1, 1, 32, 32, 32)
    outs = net_tr(x)
    assert isinstance(outs, list)
    # nnU-Net 深监督损失：各尺度上采样后求和（此处用 F.interp 模拟训练器的上采样项）
    loss = sum(
        F.interpolate(o, size=x.shape[2:], mode="nearest").square().mean() for o in outs
    )
    loss.backward()
    bad = [n for n, p in net_tr.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not bad, f"以下参数梯度缺失或含 NaN/Inf: {bad}"
    for blk in net_tr.encoder.modules():
        if isinstance(blk, BasicResidualBlock3D):
            assert (blk.conv2.norm.weight == 0).all(), "残差支路末级 norm 仿射权重应被初始化为 0"
    print("PASS 4 forward+backward: 深监督总损失反传，全部梯度有限；残差支路末级 norm 权重=0（官方初始化）")

    # ---- 5) 典型 3D fullres 规划配置的参数量（CPU 只实例化不做前向）
    typical = ResidualEncoderUNet3D(
        input_channels=1,
        num_classes=2,
        **TYPICAL_3D_FULLRES_CONFIG,  # type: ignore[arg-type]
    )
    n_params = sum(p.numel() for p in typical.parameters())
    del typical
    print(
        f"PASS 5 典型 3D fullres 规划配置参数量={n_params}（n_stages=6, features=(32,64,128,256,320,320), "
        f"每级 2 残差块；解码每级 1 卷积；来源 nnunetv2 2.7.0 planner 示例配置）"
    )

    # ---- 6) 预设信息表完整性
    assert set(RES_ENC_PRESET_INFO) == {"M", "L", "XL"}
    assert RES_ENC_PRESET_INFO["L"]["measured_vram_gb"] == 22.7
    print("PASS 6 预设信息表: M/L/XL 三档（9.1/22.7/36.6 GB 实测显存，官方文档口径）")

    print("PASS 全部自测通过")
