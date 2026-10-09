"""mednext-v2 — MedNeXt-v2：ConvNeXt 式 3D U-Net + 3D GRN 参考实现 [model]

资产卡：assets/models/mednext-v2.md

论文支撑（统一四问）：
- 哪篇论文用了它：S. Roy、Y. Kirchhoff、C. Ulrich、M. Rokuss、T. Wald、F. Isensee、
  K. Maier-Hein（德国癌症研究中心 DKFZ），《MedNeXt-v2: Scaling 3D ConvNeXts for
  Large-Scale Supervised Representation Learning in Medical Image Segmentation》，
  arXiv:2512.17774（2025-12-19 提交）。所读版本：2026-10-09 核对 arXiv 摘要页（效果数字为
  摘要口径，未逐表核对全文）；架构代码逐行核对官方仓库 MIC-DKFZ/MedNeXt main 分支
  nnunet_mednext/network_architecture/mednextv1/{blocks,MedNextV1,create_mednext_v1}.py
  （2026-10-09 抓取）。架构基础 = 同团队 MedNeXt（MICCAI 2023，
  https://arxiv.org/abs/2303.09975），其基准数字经 nnU-Net Revisited（arXiv:2404.09556）
  交叉核对。
- 目的：v2 论文研究"大规模监督预训练中，骨干本身是否是有效的表征学习器"：在 MedNeXt
  上加入 3D Global Response Normalization（GRN），按深度/宽度/上下文复合缩放，在约 1.8
  万卷 CT 上预训练，面向六个 CT/MR 基准（144 结构）的微调评测。v1 论文的目的：把
  ConvNeXt 块的复合缩放引入 3D 医学分割，在小标注数据上超过 nnU-Net。
- 效果：v2（摘要口径）：六个 CT/MR 基准（144 结构）微调后达到 SOTA，胜过七个公开的
  预训练模型；表征缩放对病理分割的收益尤其大；预训练模态特异性在全量微调后收益可忽略。
  v1（交叉核对口径）：在 nnU-Net Revisited 基准上唯一在全部 6 个数据集同时 ≥ 原版
  nnU-Net 的方法（MedNeXt L k3：BTCV 84.70 / ACDC 92.65 / LiTS 82.14 / BraTS 91.35 /
  KiTS 88.25 / AMOS 89.62，5 折 DSC）；代价是训练时长——L k3 17.3GB / 约 68h，k5 约
  233h（A100 口径）。
- 结论：v2 论文发现"从零训练更强 ⇒ 预训练后下游更强"可可靠预测，骨干缩放本身值得投入；
  代码与预训练权重经官方 nnUNet 仓库发布。v1 的适用边界：等训练预算下只追平 nnU-Net
  （架构不免费），增益要靠更大 kernel（UpKern）、深监督、残差与更长训练换。

实现说明（对齐官方仓库当前代码，grn=True 即 v2 块）：
- MedNeXtBlock（官方 blocks.py，forward 顺序）：逐通道深度卷积 k×k×k（groups=in，步长 1，
  零填充 k//2）→ GroupNorm(num_groups=in, num_channels=in)（每通道一组，等价 ConvNeXt 的
  通道序 LayerNorm）→ 1×1×1 卷积扩张 ×exp_r → GELU → [GRN] → 1×1×1 卷积压缩到 out →
  与输入残差相加（do_res）。
- 3D GRN（官方 blocks.py 内嵌实现，同 ConvNeXt V2 的 GRN 推广到 3D）：gx = 空间维 L2 范数
  （逐通道，keepdim）；nx = gx / (gx 在通道维的均值 + 1e-6)；输出 = gamma·(x·nx) + beta +
  x。gamma/beta 零初始化 ⇒ 初始时 GRN 为恒等映射，训练中逐渐学出通道级全局重标定。
- 下/上采样：MedNeXtDownBlock 用步长 2 的深度卷积、MedNeXtUpBlock 用步长 2 的深度转置
  卷积（后接空间维左侧 pad 1 补齐尺寸）；二者可带 1×1×1（转置）卷积残差路
  （do_res_up_down）。跳跃连接是**相加**（同通道），不是 U-Net 的拼接——MedNeXt 特征。
- 网络组装（官方 MedNextV1.py）：stem 1×1×1 卷积 → 4 组「编码块 + 下采样」→ 瓶颈块 →
  4 组「上采样（+skip 相加）+ 解码块」→ 输出头；deep_supervision 在瓶颈与 3 个解码级各
  挂 1×1×1 输出头，forward 返回 [全分辨率, 1/2, 1/4, 1/8, 1/16] 共 5 个 logits。
- 变体（官方 create_mednext_v1.py 原样）：S：exp_r=2、每级 2 块；B：exp_r=
  [2,3,4,4,4,4,4,3,2]、每级 2 块；M：同 B 的 exp_r、块数 [3,4,4,4,4,4,4,4,3]；L：
  exp_r=[3,4,8,8,8,8,8,4,3]、块数 [3,4,8,8,8,8,8,4,3]。基宽均为 n_channels=32。官方
  README 参数量表：S k3=5.6M、S k5=5.9M（v2 的 GRN 参数增量可忽略）。
- 与官方实现不一致处（有意偏离，逐条编号，以论文为准）：
  1. 官方 MedNextV1.py 中 down_0 漏传 grn 参数（其余 6 个重采样块都传了）——疑为笔误；
     本实现对全部重采样块统一传 grn，以 v2 论文"加入 3D GRN 模块"的整体表述为准；
  2. 仅实现 3D（官方另支持 2d 分支，本仓库服务体数据任务，不再复制）；
  3. 未内置梯度检查点（官方 checkpoint_style='outside_block' 用于大模型省显存；本实现为
     自包含参考，需要时用 torch.utils.checkpoint 包裹各级块序列即可）；
  4. Up/Down 块的尺寸补齐只对奇数核成立（官方同样隐含此假设），本实现显式断言。
- 张量布局：输入/输出 3D 布局 (B, C, D, H, W)；logits 未过 softmax；输入每维须被 16 整除
  （4 次步长 2 下采样）。deep_supervision=True 时返回 5 个尺度的 logits 列表（全分辨率在
  前，不做上采样，上采样由调用方在损失内完成——nnU-Net 系训练器的标准做法）。
- 数值稳定：GRN 分母的 1e-6 防全零特征图 0/0（正常激活的通道均值远大于 1e-6，不影响
  数值与梯度）；GroupNorm/InstanceNorm 内部对 var+eps 开方（eps=1e-5，PyTorch 默认）；
  无裸 log/exp。自测断言输出与梯度无 NaN/Inf。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "MedNeXtBlockV2",
    "MedNeXtDownBlockV2",
    "MedNeXtUpBlockV2",
    "GlobalResponseNorm3d",
    "MedNeXtV2",
    "MEDNEXT_V2_VARIANTS",
    "create_mednext_v2",
]


class GlobalResponseNorm3d(nn.Module):
    """3D Global Response Normalization（ConvNeXt-V2 的 GRN 推广到体数据）。

    逐通道计算空间维 L2 范数 gx，除以所有通道范数的均值（加 1e-6 防全零图零除）得到
    归一化系数 nx，输出 gamma·(x·nx) + beta + x。gamma/beta 零初始化：初始为恒等映射。

    参数：
        channels: 通道数（GRN 参数逐通道：gamma、beta 各 1×C×1×1×1）。
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, D, H, W) → (B, C, D, H, W)。"""
        gx = x.norm(p=2, dim=(2, 3, 4), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)  # 1e-6：防全零特征图 0/0，官方取值
        return self.gamma * (x * nx) + self.beta + x


class MedNeXtBlockV2(nn.Module):
    """MedNeXt-v2 基础块：深度卷积 → GroupNorm → 扩张 → GELU → GRN → 压缩 →（残差）。

    布局：(B, in_channels, D, H, W) → (B, out_channels, D, H, W)，空间尺寸不变。

    参数：
        in_channels / out_channels: 输入/输出通道数（U-Net 内部两者相等，残差才有意义）。
        exp_r: 扩张比（扩张层通道 = exp_r × in_channels，继承自 Swin 的记法）。
        kernel_size: 深度卷积核大小（奇数；论文 3/5，UpKern 从 3 长到 5）。
        do_res: 是否残差相加（论文训练时为 True）。
        grn: 是否启用 3D GRN（v2 为 True；GRN 作用于扩张后的隐层通道）。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 3,
        do_res: bool = True,
        grn: bool = True,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size 必须为奇数（零填充 k//2 + 上采样补齐的尺寸推导），得到 {kernel_size}")
        self.do_res = do_res
        hidden = exp_r * in_channels
        self.conv_dw = nn.Conv3d(in_channels, in_channels, kernel_size, padding=kernel_size // 2, groups=in_channels)
        self.norm = nn.GroupNorm(num_groups=in_channels, num_channels=in_channels)
        self.conv_expand = nn.Conv3d(in_channels, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.grn = GlobalResponseNorm3d(hidden) if grn else None
        self.conv_compress = nn.Conv3d(hidden, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_channels, D, H, W) → (B, out_channels, D, H, W)。"""
        h = self.conv_dw(x)
        h = self.norm(h)
        h = self.act(self.conv_expand(h))
        if self.grn is not None:
            h = self.grn(h)
        h = self.conv_compress(h)
        if self.do_res:
            h = h + x
        return h


class MedNeXtDownBlockV2(MedNeXtBlockV2):
    """步长 2 下采样块：深度卷积带步长 2；可选 1×1×1 步长 2 卷积残差路。

    参数：
        in_channels / out_channels: 输入/输出通道数（输出通常为输入 2 倍）。
        其余同 MedNeXtBlockV2；do_res 在此控制的是重采样残差路（块内主路与官方一致不带
        残差——官方 Down/Up 块强制内层 do_res=False）。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 3,
        do_res: bool = True,
        grn: bool = True,
    ) -> None:
        # 内层块不带残差（官方 Down/Up 块强制内层 do_res=False）；重采样残差路用
        # 独立属性 resample_do_res，避免与父类 forward 的块内残差逻辑冲突
        super().__init__(in_channels, out_channels, exp_r, kernel_size, do_res=False, grn=grn)
        self.resample_do_res = do_res
        # 覆盖父类的深度卷积：带步长 2（同官方 MedNeXtDownBlock）
        self.conv_dw = nn.Conv3d(in_channels, in_channels, kernel_size, stride=2, padding=kernel_size // 2, groups=in_channels)
        self.res_conv: nn.Conv3d | None = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=2) if do_res else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C_in, D, H, W) → (B, C_out, ⌈D/2⌉, ⌈H/2⌉, ⌈W/2⌉)（输入偶数维时即 D/2）。"""
        h = super().forward(x)
        if self.resample_do_res and self.res_conv is not None:
            h = h + self.res_conv(x)
        return h


class MedNeXtUpBlockV2(MedNeXtBlockV2):
    """步长 2 上采样块：深度转置卷积 + 空间维左侧 pad 1 补齐；可选转置卷积残差路。

    尺寸推导：转置卷积输出 = (D−1)×2 − 2·⌊k/2⌋ + k = 2D−1（奇数 k），左侧 pad 1 后 = 2D，
    与跳跃连接精确对齐。
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        exp_r: int = 4,
        kernel_size: int = 3,
        do_res: bool = True,
        grn: bool = True,
    ) -> None:
        # 同 DownBlock：内层块不带残差，重采样残差路用独立属性 resample_do_res
        super().__init__(in_channels, out_channels, exp_r, kernel_size, do_res=False, grn=grn)
        self.resample_do_res = do_res
        # 覆盖父类的深度卷积：步长 2 的分组转置卷积（同官方 MedNeXtUpBlock）
        self.conv_dw = nn.ConvTranspose3d(
            in_channels, in_channels, kernel_size, stride=2, padding=kernel_size // 2, groups=in_channels
        )
        self.res_conv: nn.ConvTranspose3d | None = (
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=1, stride=2) if do_res else None
        )

    @staticmethod
    def _pad_left(x: torch.Tensor) -> torch.Tensor:
        """每个空间维的起点侧 pad 1：(1,0,1,0,1,0) 对应 W/H/D 三轴（torch F.pad 轴序）。"""
        return F.pad(x, (1, 0, 1, 0, 1, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C_in, D, H, W) → (B, C_out, 2D, 2H, 2W)。"""
        h = self._pad_left(super().forward(x))
        if self.resample_do_res and self.res_conv is not None:
            h = h + self._pad_left(self.res_conv(x))
        return h


class _OutBlockV2(nn.Module):
    """输出头：1×1×1 转置卷积到类别数（官方用 ConvTranspose3d(k=1)，函数上等价卷积）。"""

    def __init__(self, in_channels: int, n_classes: int) -> None:
        super().__init__()
        self.conv_out = nn.ConvTranspose3d(in_channels, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_channels, D, H, W) → (B, n_classes, D, H, W) logits。"""
        return self.conv_out(x)


class MedNeXtV2(nn.Module):
    """MedNeXt-v2 主网络（官方 MedNextV1.py 组装 + grn=True）。

    布局：输入 (B, in_channels, D, H, W)（每维须被 16 整除）；deep_supervision=True 时返回
    5 个 logits 的列表 [全分辨率, 1/2, 1/4, 1/8, 1/16]（全分辨率在前，未上采样），False 时
    返回单个全分辨率 logits 张量。跳跃连接为逐元素相加（同通道），非拼接。

    参数：
        in_channels: 输入通道数（CT/MRI 通常 1）。
        n_classes: 输出类别数。
        n_channels: 基宽（官方全部变体为 32；各级通道 = n_channels × 2^级数，瓶颈 16 倍）。
        exp_r: 扩张比，标量广播到 9 个位置，或长度 9 的序列（位置顺序：编码 0–3、瓶颈、
            解码 3–0，与官方一致）。
        kernel_size: 深度卷积核（奇数；3 或 5）。
        block_counts: 长度 9 的每级块数序列（位置顺序同 exp_r）。
        deep_supervision / do_res / do_res_up_down / grn: 同块级说明（论文训练配置全 True）。
    """

    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        n_channels: int = 32,
        exp_r: int | list[int] = 2,
        kernel_size: int = 3,
        block_counts: list[int] | None = None,
        deep_supervision: bool = True,
        do_res: bool = True,
        do_res_up_down: bool = True,
        grn: bool = True,
    ) -> None:
        super().__init__()
        block_counts = [2] * 9 if block_counts is None else block_counts
        if len(block_counts) != 9:
            raise ValueError(f"block_counts 需要长度 9（编码 0–3、瓶颈、解码 3–0），得到 {len(block_counts)}")
        if isinstance(exp_r, int):
            exp_r_list = [exp_r] * 9
        else:
            if len(exp_r) != 9:
                raise ValueError(f"exp_r 序列需要长度 9，得到 {len(exp_r)}")
            exp_r_list = list(exp_r)
        self.do_ds = deep_supervision
        c = n_channels

        self.stem = nn.Conv3d(in_channels, c, kernel_size=1)

        def enc_stage(i: int, ch: int) -> nn.Sequential:
            return nn.Sequential(
                *[MedNeXtBlockV2(ch, ch, exp_r_list[i], kernel_size, do_res, grn) for _ in range(block_counts[i])]
            )

        self.enc_block_0 = enc_stage(0, c)
        self.down_0 = MedNeXtDownBlockV2(c, 2 * c, exp_r_list[1], kernel_size, do_res_up_down, grn)
        self.enc_block_1 = enc_stage(1, 2 * c)
        self.down_1 = MedNeXtDownBlockV2(2 * c, 4 * c, exp_r_list[2], kernel_size, do_res_up_down, grn)
        self.enc_block_2 = enc_stage(2, 4 * c)
        self.down_2 = MedNeXtDownBlockV2(4 * c, 8 * c, exp_r_list[3], kernel_size, do_res_up_down, grn)
        self.enc_block_3 = enc_stage(3, 8 * c)
        self.down_3 = MedNeXtDownBlockV2(8 * c, 16 * c, exp_r_list[4], kernel_size, do_res_up_down, grn)

        self.bottleneck = nn.Sequential(
            *[MedNeXtBlockV2(16 * c, 16 * c, exp_r_list[4], kernel_size, do_res, grn) for _ in range(block_counts[4])]
        )

        self.up_3 = MedNeXtUpBlockV2(16 * c, 8 * c, exp_r_list[5], kernel_size, do_res_up_down, grn)
        self.dec_block_3 = enc_stage(5, 8 * c)
        self.up_2 = MedNeXtUpBlockV2(8 * c, 4 * c, exp_r_list[6], kernel_size, do_res_up_down, grn)
        self.dec_block_2 = enc_stage(6, 4 * c)
        self.up_1 = MedNeXtUpBlockV2(4 * c, 2 * c, exp_r_list[7], kernel_size, do_res_up_down, grn)
        self.dec_block_1 = enc_stage(7, 2 * c)
        self.up_0 = MedNeXtUpBlockV2(2 * c, c, exp_r_list[8], kernel_size, do_res_up_down, grn)
        self.dec_block_0 = enc_stage(8, c)

        self.out_0 = _OutBlockV2(c, n_classes)
        if deep_supervision:
            self.out_1 = _OutBlockV2(2 * c, n_classes)
            self.out_2 = _OutBlockV2(4 * c, n_classes)
            self.out_3 = _OutBlockV2(8 * c, n_classes)
            self.out_4 = _OutBlockV2(16 * c, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        """(B, in_channels, D, H, W) → logits（单张量或 5 尺度列表，见类 docstring）。"""
        if x.dim() != 5:
            raise ValueError(f"期望 (B, C, D, H, W) 5 维输入，得到 shape={tuple(x.shape)}")

        x_res_0 = self.enc_block_0(self.stem(x))
        x = self.down_0(x_res_0)
        x_res_1 = self.enc_block_1(x)
        x = self.down_1(x_res_1)
        x_res_2 = self.enc_block_2(x)
        x = self.down_2(x_res_2)
        x_res_3 = self.enc_block_3(x)
        x = self.down_3(x_res_3)

        x = self.bottleneck(x)
        if self.do_ds:
            x_ds_4 = self.out_4(x)

        x = self.dec_block_3(self.up_3(x) + x_res_3)  # 跳跃连接：相加（非拼接）
        if self.do_ds:
            x_ds_3 = self.out_3(x)
        x = self.dec_block_2(self.up_2(x) + x_res_2)
        if self.do_ds:
            x_ds_2 = self.out_2(x)
        x = self.dec_block_1(self.up_1(x) + x_res_1)
        if self.do_ds:
            x_ds_1 = self.out_1(x)
        x = self.dec_block_0(self.up_0(x) + x_res_0)
        x = self.out_0(x)

        if self.do_ds:
            return [x, x_ds_1, x_ds_2, x_ds_3, x_ds_4]
        return x


# 官方变体配置（create_mednext_v1.py 原样；v2 = 同配置 + grn=True）
MEDNEXT_V2_VARIANTS: dict[str, dict[str, object]] = {
    "S": {"exp_r": 2, "block_counts": [2, 2, 2, 2, 2, 2, 2, 2, 2]},
    "B": {"exp_r": [2, 3, 4, 4, 4, 4, 4, 3, 2], "block_counts": [2, 2, 2, 2, 2, 2, 2, 2, 2]},
    "M": {"exp_r": [2, 3, 4, 4, 4, 4, 4, 3, 2], "block_counts": [3, 4, 4, 4, 4, 4, 4, 4, 3]},
    "L": {"exp_r": [3, 4, 8, 8, 8, 8, 8, 4, 3], "block_counts": [3, 4, 8, 8, 8, 8, 8, 4, 3]},
}


def create_mednext_v2(
    model_id: str,
    in_channels: int,
    n_classes: int,
    kernel_size: int = 3,
    deep_supervision: bool = True,
) -> MedNeXtV2:
    """按官方变体表构建 MedNeXt-v2（S/B/M/L；基宽 32，grn=True）。

    参数：
        model_id: "S" / "B" / "M" / "L"。
        in_channels / n_classes / kernel_size / deep_supervision: 同 MedNeXtV2。
    返回：
        MedNeXtV2 实例（He 无特殊初始化——官方 MedNeXt 未做自定义初始化，保留 PyTorch 默认）。
    """
    if model_id not in MEDNEXT_V2_VARIANTS:
        raise ValueError(f"model_id 须为 {sorted(MEDNEXT_V2_VARIANTS)}，得到 {model_id!r}")
    cfg = MEDNEXT_V2_VARIANTS[model_id]
    return MedNeXtV2(
        in_channels=in_channels,
        n_classes=n_classes,
        n_channels=32,
        exp_r=cfg["exp_r"],  # type: ignore[arg-type]
        kernel_size=kernel_size,
        block_counts=cfg["block_counts"],  # type: ignore[arg-type]
        deep_supervision=deep_supervision,
        do_res=True,
        do_res_up_down=True,
        grn=True,
    )


if __name__ == "__main__":
    torch.manual_seed(0)  # 固定随机种子，保证自测可复现

    # ---- 1) GRN：零初始化 ⇒ 恒等映射；学出非零 gamma/beta 后输出改变且有限
    grn = GlobalResponseNorm3d(4)
    x = torch.randn(2, 4, 6, 6, 6)
    with torch.no_grad():
        identity = grn(x)
    assert torch.allclose(identity, x, atol=1e-6), "零初始化的 GRN 应为恒等映射"
    with torch.no_grad():
        grn.gamma.fill_(0.5)
        grn.beta.fill_(0.1)
    out = grn(x)
    assert not torch.allclose(out, x), "非零参数后 GRN 应改变输出"
    assert torch.isfinite(out).all(), "GRN 输出含 NaN/Inf"
    x_zero = torch.zeros(1, 4, 4, 4, 4)  # 全零退化输入：防零除路径
    assert torch.isfinite(grn(x_zero)).all(), "全零输入下 GRN 应有限（1e-6 防零除）"
    print("PASS 1 GRN: 零初始化=恒等映射；非零参数改变输出；全零退化输入仍有限（eps=1e-6）")

    # ---- 2) 小网络端到端 forward（深监督开）：5 个尺度、形状正确
    net = MedNeXtV2(
        in_channels=1,
        n_classes=3,
        n_channels=4,
        exp_r=2,
        kernel_size=3,
        block_counts=[1] * 9,
        deep_supervision=True,
        do_res=True,
        do_res_up_down=True,
        grn=True,
    )
    x32 = torch.randn(1, 1, 32, 32, 32)  # 注意：16³ 会让瓶颈落到 1³，训练模式下 GroupNorm 无法估方差
    outs = net(x32)
    assert isinstance(outs, list) and len(outs) == 5, f"深监督应有 5 个输出，得到 {len(outs)}"
    for scale, o in zip((1, 2, 4, 8, 16), outs):
        expect = (1, 3, 32 // scale, 32 // scale, 32 // scale)
        assert o.shape == expect, f"尺度 1/{scale} 形状不符: {tuple(o.shape)}"
        assert torch.isfinite(o).all(), f"尺度 1/{scale} 输出含 NaN/Inf"
    print("PASS 2 端到端 forward: 5 个尺度 logits（1 ~ 1/16），全分辨率在前，全部有限")

    # ---- 3) 深监督关 + 残差关：返回单张量、路径仍有限
    net_plain = MedNeXtV2(
        in_channels=1, n_classes=2, n_channels=4, exp_r=2, kernel_size=3,
        block_counts=[1] * 9, deep_supervision=False, do_res=False, do_res_up_down=False, grn=True,
    )
    with torch.no_grad():
        single = net_plain(torch.randn(1, 1, 32, 32, 32))
    assert isinstance(single, torch.Tensor) and single.shape == (1, 2, 32, 32, 32)
    assert torch.isfinite(single).all(), "无残差路径输出含 NaN/Inf"
    print("PASS 3 深监督关/残差关: 返回单张量 (1,2,16,16,16)，全部有限")

    # ---- 4) forward+backward：全部参数梯度有限（含 GRN 的 gamma/beta）
    loss = sum(o.square().mean() for o in outs)
    loss.backward()
    bad = [n for n, p in net.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not bad, f"以下参数梯度缺失或含 NaN/Inf: {bad}"
    n_tiny = sum(p.numel() for p in net.parameters())
    print(f"PASS 4 forward+backward: 深监督总损失反传，全部梯度有限（含 GRN 参数），小网参数量={n_tiny}")

    # ---- 5) 官方变体参数量核对：S 应≈5.6M（官方 README 表），L 一并打印
    for mid, expect_m in (("S", 5.6), ("L", None)):
        m = create_mednext_v2(mid, in_channels=1, n_classes=2, kernel_size=3, deep_supervision=False)
        n = sum(p.numel() for p in m.parameters())
        del m
        if expect_m is not None:
            assert abs(n / 1e6 - expect_m) < 0.15, f"{mid} 参数量 {n/1e6:.2f}M 偏离官方表 {expect_m}M 过多"
            print(f"PASS 5 变体参数量: {mid} k3 = {n}（≈{n/1e6:.2f}M，与官方 README 表 {expect_m}M 一致，GRN 增量可忽略）")
        else:
            print(f"PASS 5 变体参数量: {mid} k3 = {n}（≈{n/1e6:.2f}M，供资产卡量级参考）")

    print("PASS 全部自测通过")
