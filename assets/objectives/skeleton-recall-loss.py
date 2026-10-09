"""skeleton-recall-loss —— 细管状结构连通性损失（骨架召回） [loss]（资产卡：assets/objectives/skeleton-recall-loss.md）

论文支撑（统一四问，依据论文 ECCV 2024 正式版对应的 arXiv v2，链接见下）：
- 哪篇论文用了它：Yannick Kirchhoff、Maximilian R. Rokuss、Saikat Roy、Constantin
  Ulrich、Fabian Isensee、Klaus Maier-Hein 等（德国癌症研究中心 DKFZ 团队），
  "Skeleton Recall loss for Connectivity Conserving and Resource Efficient
  Segmentation of Thin Tubular Structures"，ECCV 2024，
  https://arxiv.org/abs/2404.03010（所读版本：arXiv v2，2024-07-17）。
- 目的：血管、神经、气道等细管状结构的分割中，Dice/CE 只优化体素重叠、不约束
  拓扑连通性，管子断成几截会毁掉下游血流计算与导航；已有拓扑感知损失（clDice 等）
  要在 GPU 上对预测反复做可微骨架化，3D 与多类场景开销爆炸（clDice 在 A100 40GB
  上 13 类直接 OOM）。
- 效果：ToothFairy（3D 牙槽神经管）Dice 71.80→74.42、clDice 指标 89.16→92.05；
  TopCoW 13 类多类血管 Dice 85.36→86.59、clDice 93.68→94.35；2D Roads/DRIVE/
  Cracks 多数指标最优。开销仅训练时长 +8%、显存 +2%（clDice 为 +88% / +52%），
  总开销降低 90% 以上。
- 结论：把骨架化挪到 CPU 数据加载/离线阶段、只对 GT 骨架管道算预测的 soft
  recall，即可在资源开销降低一个数量级的同时取得细管结构连通性的 SOTA；是首个
  支持多类设定的细管结构连通性损失。骨架质量依赖预处理，建议与 Dice+CE 组合
  （连接性权重 w ∈ {0.1, 1.0}）。

实现说明
- 关键公式与论文式子编号的对应（所读版本：arXiv v2，2024-07-17）：
  * 论文 Section 3.2 式(1)：L_SkelRecall = -(1/|C|) Σ_{c∈C} [ Σ_i Y_skel,i,c ·
    Ŷ_i,c / Σ_i Y_skel,i,c ] —— 网络预测概率 Ŷ 在 GT 骨架（管道）Y_skel 内的
    soft recall 取负、按类平均。skeleton_recall_loss 取 smooth=0 即此式
    （smooth>0 的默认口径见下方"与官方实现不一致处"第 (3) 条）；
  * 论文式(2)：L = L_generic + w · L_connectivity —— 官方 nnU-Net 集成为
    CE + Soft Dice + 1 · SkelRecall（weight_ce = weight_dice = weight_srec = 1，
    见 nnUNetTrainerSkeletonRecall）。本条目只实现连接性项 L_connectivity 本身，
    与 Dice+CE 的组合按式(2)在任务仓库侧加权。
- 参考的开源实现（MIC-DKFZ/Skeleton-Recall，本地克隆
  /tmp/mtskills-ref-skeleton-recall）：
  * skeleton_recall/training/loss/dice.py::SoftSkeletonRecallLoss —— soft recall
    的分子分母、smooth 与聚合口径（对 (B, C_fg) 每样本每通道各算一个 recall 再
    mean，返回其负值）；
  * skeleton_recall/training/data_augmentation/skeletonization.py::
    SkeletonTransform —— 骨架提取（skimage skeletonize）+ 管道膨胀（footprint =
    ndi.iterate_structure(ndi.generate_binary_structure(ndim, 1), 2)，十字形
    连通性 1 结构元迭代 2 次 = 曼哈顿半径 2）+ 乘 GT 裁回（seg_skel = skel·seg，
    兼做按类着色）；
  * skeleton_recall/training/loss/compound_losses.py::DC_SkelREC_and_CE_loss 与
    skeleton_recall/training/nnUNetTrainer/nnUNetTrainerSkeletonRecall.py ——
    组合权重（三者均 1）与 smooth=1e-5。
- 与官方实现不一致处（逐条编号）：
  * (1) 骨架提取不进本条目运行时依赖：官方用 skimage.morphology.skeletonize
    （2D Zhang-Suen / 3D Lee 薄化，论文 Section 3.2），本仓库代码规范
    （assets/CODE_STYLE.md）禁用 skimage 且 scipy 无薄化算法，故把预计算好的
    骨架（管道）作为损失输入参数——与官方"数据加载阶段 CPU 预计算"的用法一致；
    实际任务中骨架提取用 skimage.morphology.skeletonize（2D/3D）等工具，属任务
    仓库预处理步骤。dilate_skeleton_to_tube 提供官方同款 scipy 管道膨胀。
  * (2) 输入布局收紧为 C ≥ 2 的 softmax 布局（通道 0 = 背景）：官方 scatter 分支
    恒丢通道 0（背景类），C==1 时官方 y[:, 1:] 会取空通道（其单通道二值场景在
    nnU-Net 中实际仍是 C=2 布局），本实现显式拒绝 C==1 以免语义歧义；二值任务
    请用 (背景, 前景) 双通道。
  * (3) 数值稳定口径：论文式(1)无平滑项；官方实现在分子分母各加 smooth=1e-5
    并对分母 clamp(min=1e-8)。本实现默认对齐官方（smooth=1e-5、clamp_min=1e-8，
    保证逐数值等价，见自测第 2 项），同时保留 smooth=0 得到严格论文式(1)
    （自测第 1 项按此与朴素双精度参考对拍）。注意坑：空骨架通道在 smooth>0
    官方口径下 rec = smooth/clip(smooth) = 1 → 该通道损失贡献 -1。
  * (4) 管道膨胀用 scipy.ndimage.binary_dilation（border_value=0）替代官方
    skimage.morphology.dilation（灰度膨胀、边界 reflect）：对 0/1 输入与二值
    footprint 二者等价，仅图像边界处理约定不同——骨架不贴边时逐像素一致
    （自测第 7 项在 skimage 可用时对拍，缺席则 SKIP）；合成用例均刻意不贴边。
  * (5) 聚合口径：官方默认 batch_dice=False（每 (样本, 通道) 一个 recall 再
    mean），batch_dice=True 的跨卡/跨样本求和聚合是 nnU-Net DDP 训练细节，
    本条目不实现（任务需要时在调用侧对 reduction="none" 结果自行求和）。
  * (6) 官方支持 loss_mask（ignore label 场景），本实现同样提供该参数且语义
    一致：骨架统计与召回统计同乘掩码。
  * (7) 分母 dtype 提升修复：官方分母是 int64 的 sum_gt 直接加 Python 标量
    smooth，torch 类型提升使其落入 float32（分子则跟随输入 dtype）——float64
    输入时分子分母精度劈叉（空骨架通道 recall 会算成 1.00000003 而非 1）。本
    实现显式把 sum_gt 转到分子的累加 dtype；float32 输入（官方训练场景）下与
    官方逐位一致，float64 输入下整体保持 float64。
- 数值稳定：smooth=1e-5 同时加在分子与分母（官方 trainer 值；远小于任何非空
  骨架的体素软计数量级 ≥1，fp32 下对非空通道不改变损失值，防全空骨架 0/0）；
  分母再 clamp_min=1e-8 兜底（与官方 torch.clip(sum_gt + smooth, 1e-8) 一致）；
  概率来自 softmax/sigmoid 天然落在 (0,1)，无 log/exp/幂运算，无需额外钳制。

布局约定：2D=(B,C,H,W)，3D=(B,C,D,H,W)（softmax 布局，通道 0=背景；from_logits=
True 时 pred 为未过 softmax 的 logits）；骨架 skeleton 为 (B,1,...) 整数标签图
（值=类别索引，0=无骨架，官方 SkeletonTransform 追加通道的形式）或 (B,C,...)
one-hot/多热图；dilate_skeleton_to_tube / skeleton_tube_footprint 用 numpy 空间
布局 (D,H,W)（3D）/(H,W)（2D）。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

__all__ = [
    "skeleton_recall_loss",
    "SkeletonRecallLoss",
    "skeleton_tube_footprint",
    "dilate_skeleton_to_tube",
]

# smooth 与分母下限：与官方 nnUNetTrainerSkeletonRecall / SoftSkeletonRecallLoss
# 完全一致（见文件头"数值稳定"注释）。smooth=0 时退化为严格论文式(1)。
_DEFAULT_SMOOTH = 1e-5
_DEFAULT_CLAMP_MIN = 1e-8


def _to_prob(pred: torch.Tensor, from_logits: bool) -> torch.Tensor:
    """把网络原始输出转成每通道概率。

    参数：
        pred: torch.Tensor，2D=(B,C,H,W) 或 3D=(B,C,D,H,W)，C ≥ 2。
        from_logits: True 表示 pred 是未过 softmax 的 logit（沿通道维做
            softmax）；False 表示 pred 已是概率。
    返回：
        与 pred 同形状的概率张量，每通道值域 [0,1]，通道和为 1。
    """
    if not from_logits:
        return pred
    return F.softmax(pred, dim=1)


def _skeleton_to_fg_onehot(skeleton: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
    """把骨架输入统一成 (B, C-1, ...) 的前景 one-hot（丢掉背景通道）。

    两种合法输入（语义见文件头"布局约定"）：
        (a) 与 prob 同形的 (B,C,...) one-hot/多热图——官方 equal-shape 分支，
            取 y[:, 1:]；
        (b) (B,1,...) 整数标签图（值=类别索引，0=无骨架）——官方 scatter 分支，
            scatter 成 one-hot 后丢通道 0。
    """
    if skeleton.dim() != prob.dim():
        raise ValueError(
            f"skeleton 维度 {skeleton.dim()} 与 pred 维度 {prob.dim()} 不一致；"
            "应为 (B,1,...) 标签图或 (B,C,...) one-hot"
        )
    if skeleton.shape == prob.shape:
        return skeleton.to(dtype=prob.dtype)[:, 1:]
    if skeleton.shape[1] == 1:
        num_chans = prob.shape[1]
        smin = int(skeleton.min())
        smax = int(skeleton.max())
        if smin < 0 or smax >= num_chans:
            raise ValueError(
                f"skeleton 标签取值 [{smin}, {smax}] 越界，应为 [0, {num_chans - 1}]"
                f"（通道数 C={num_chans}，0 表示无骨架）"
            )
        label = skeleton.long()
        onehot = torch.zeros(prob.shape, device=prob.device, dtype=label.dtype)
        onehot.scatter_(1, label, 1)  # 官方同款 scatter
        return onehot[:, 1:]
    raise ValueError(
        f"skeleton 形状 {tuple(skeleton.shape)} 非法：应为 (B,1,...) 标签图或"
        f" (B,C,...) one-hot {tuple(prob.shape)}"
    )


def skeleton_recall_loss(
    pred: torch.Tensor,
    skeleton: torch.Tensor,
    from_logits: bool = False,
    smooth: float = _DEFAULT_SMOOTH,
    clamp_min: float = _DEFAULT_CLAMP_MIN,
    loss_mask: torch.Tensor | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Skeleton Recall 损失（论文 Section 3.2 式(1)）：预测在 GT 骨架管道内的
    soft recall 取负。

    损失越小表示骨架管道被预测覆盖得越完整（perfect recall 时 loss=-1，
    smooth>0 官方口径下损失 ∈ [-1, 0]）。骨架（管道）必须离线预计算：骨架
    提取（如 skimage.morphology.skeletonize）属任务仓库预处理，本函数只吃
    现成的骨架输入；原始骨架可用 dilate_skeleton_to_tube 膨胀成官方同款
    半径 2 管道（并乘 GT 裁回/按类着色）。

    参数：
        pred: 2D=(B,C,H,W) 或 3D=(B,C,D,H,W)。softmax 布局，通道 0=背景、
            C ≥ 2；from_logits=True 时为未过 softmax 的 logits。
        skeleton: 骨架（管道）输入，二选一——
            (a) (B,1,...) 整数标签图，值=类别索引（1..C-1），0=无骨架
                （官方 SkeletonTransform 追加通道 + skel·seg 的形式）；
            (b) (B,C,...) one-hot/多热图（与 pred 同形，官方 equal-shape 分支）。
        from_logits: pred 是否为 logit（沿通道 softmax），默认 False。
        smooth: 分子分母各加的平滑项，默认 1e-5（官方口径；取值依据见文件头）。
            设 0 得到严格论文式(1)（此时空骨架通道返回 0 而非 -1）。
        clamp_min: 分母下限，默认 1e-8（官方 torch.clip(..., 1e-8) 兜底防零除）。
        loss_mask: 可选 (B,1,...) 掩码（ignore label 场景），与官方一致地同乘
            骨架统计与召回统计。
        reduction: "mean"（标量，官方口径）/ "sum" / "none"（返回 (B,C_fg)）。

    返回：
        torch.Tensor，reduction 决定标量或 (B,C_fg)；smooth>0 时标量 ∈ [-1, 0]。

    异常：
        ValueError：C < 2、skeleton 形状/取值非法或 reduction 非法。
    """
    if pred.dim() not in (4, 5):
        raise ValueError(
            f"pred 应为 2D (B,C,H,W) 或 3D (B,C,D,H,W)，收到 {tuple(pred.shape)}"
        )
    if pred.shape[1] < 2:
        raise ValueError(
            "skeleton recall 需要显式背景通道（softmax 布局，通道 0=背景，C≥2）；"
            "二值任务请输出 (背景, 前景) 双通道（见文件头不一致处第 (2) 条）"
        )
    prob = _to_prob(pred, from_logits)
    skel_fg = _skeleton_to_fg_onehot(skeleton, prob)
    prob = prob[:, 1:]  # 官方 x = x[:, 1:]：损失只用前景概率，背景通道不参与

    # 空间维聚合出每 (B, 前景通道) 的软计数：分子 = 骨架管道内被预测覆盖的概率
    # 质量（式(1) 的 Σ_i Y_skel·Ŷ），分母 = 管道体素数（Σ_i Y_skel）
    axes = list(range(2, prob.dim()))
    if loss_mask is not None:
        inter = (prob * skel_fg * loss_mask).sum(axes)
        sum_gt = (skel_fg * loss_mask).sum(axes)
    else:
        inter = (prob * skel_fg).sum(axes)
        sum_gt = skel_fg.sum(axes)

    # 官方口径：rec = (inter + smooth) / clip(sum_gt + smooth, clamp_min)；
    # smooth=0、管道非空时即论文式(1) 的逐类 recall。注意官方对 int64 的 sum_gt
    # 直接加 Python 标量 smooth，torch 类型提升会把分母落成 float32（分子跟随
    # 输入 dtype）——本实现显式把 sum_gt 转到分子累加 dtype：float32 输入（官方
    # 训练场景）时与官方逐位一致，float64 输入时避免分子分母精度劈叉（见文件头
    # 不一致处第 (7) 条）。
    sum_gt = sum_gt.to(dtype=inter.dtype)
    rec = (inter + smooth) / (sum_gt + smooth).clamp_min(clamp_min)
    loss = -rec

    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"reduction 只支持 mean/sum/none，收到 {reduction!r}")


class SkeletonRecallLoss(torch.nn.Module):
    """Skeleton Recall 损失的 nn.Module 封装，参数同 skeleton_recall_loss。

    参数：
        from_logits: forward 收到的 pred 是否为 logit，默认 False。
        smooth: 分子分母平滑项，默认 1e-5（官方口径）。
        clamp_min: 分母下限，默认 1e-8。
        reduction: "mean" / "sum" / "none"，默认 "mean"。
    """

    def __init__(
        self,
        from_logits: bool = False,
        smooth: float = _DEFAULT_SMOOTH,
        clamp_min: float = _DEFAULT_CLAMP_MIN,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.from_logits = from_logits
        self.smooth = smooth
        self.clamp_min = clamp_min
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        skeleton: torch.Tensor,
        loss_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """计算 Skeleton Recall 损失，参数语义同 skeleton_recall_loss。"""
        return skeleton_recall_loss(
            pred,
            skeleton,
            from_logits=self.from_logits,
            smooth=self.smooth,
            clamp_min=self.clamp_min,
            loss_mask=loss_mask,
            reduction=self.reduction,
        )


def skeleton_tube_footprint(ndim: int) -> np.ndarray:
    """官方同款管道膨胀 footprint：十字形（连通性 1）结构元迭代 2 次。

    结果是曼哈顿距离 ≤ 2 的菱形/十字球（2D 为 5×5 菱形，3D 为 5×5×5 十字球），
    对应论文 Section 3.2 "diamond kernel of radius 2" 与官方 SkeletonTransform
    的 get_tube_footprint（ndi.iterate_structure(ndi.generate_binary_structure(
    ndim, 1), 2)）。

    参数：
        ndim: 空间维数，2 或 3。

    返回：
        numpy.ndarray，bool 数组，形状 (5,)*ndim。

    异常：
        ValueError：ndim 不为 2 或 3。
    """
    if ndim not in (2, 3):
        raise ValueError(f"ndim 只支持 2 或 3，收到 {ndim}")
    return ndi.iterate_structure(ndi.generate_binary_structure(ndim, 1), 2)


def dilate_skeleton_to_tube(
    skeleton: np.ndarray,
    tube: bool = True,
    clip_to: np.ndarray | None = None,
) -> np.ndarray:
    """把二值骨架膨胀成管道软标签（numpy，scipy 实现；官方膨胀算子的等价替换）。

    对应官方 SkeletonTransform 的膨胀与"乘 GT 裁回/按类着色"两步（骨架提取本身
    用 skimage.morphology.skeletonize 等工具在任务仓库预处理完成，见文件头不一致
    处第 (1)(4) 条）。

    参数：
        skeleton: numpy.ndarray，二值骨架，非零为骨架点。布局 (D,H,W)（3D）/
            (H,W)（2D）。骨架点应距图像边界 ≥ 2（官方 skimage 膨胀与 scipy 在
            边界处理上约定不同，贴边时结果可能差 1-2 圈，见不一致处第 (4) 条）。
        tube: True（默认）时用官方同款半径 2 footprint 膨胀成管道；False 时
            原样返回布尔化骨架。
        clip_to: 可选，与 skeleton 同形的 GT 标签图或掩码；管道与其逐元素相乘，
            把管道裁回 GT 并（标签图场景）按 GT 类别着色——官方
            seg_skel = skel * seg 的步骤，其输出可直接作为损失的 skeleton 标签图。

    返回：
        numpy.ndarray：clip_to 为 None 时返回 bool 管道掩码；给定时返回
        bool * clip_to 的广播提升结果（整型标签图输入 → 整型标签图）。

    异常：
        ValueError：skeleton 维度不是 2 或 3，或与 clip_to 形状不匹配。
    """
    if skeleton.ndim not in (2, 3):
        raise ValueError(f"skeleton 应为 (H,W) 或 (D,H,W)，收到形状 {skeleton.shape}")
    if clip_to is not None and clip_to.shape != skeleton.shape:
        raise ValueError(
            f"clip_to 形状 {clip_to.shape} 与 skeleton {skeleton.shape} 不匹配"
        )
    binary = skeleton != 0
    if tube:
        binary = ndi.binary_dilation(binary, structure=skeleton_tube_footprint(skeleton.ndim))
    if clip_to is None:
        return binary
    return binary * clip_to


# ---------------------------------------------------------------------------
# 自测：CPU 可跑、固定随机种子、数秒内完成
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    # ---- 合成数据：细管线状骨架，同一条线膨胀出骨架管道与 GT 管道（同源构造，
    # ---- 骨架严格落在 GT 内）；曲线与管道均刻意不贴边（边界膨胀约定见不一致处 (4)）
    def _curve_2d(h: int, w: int, y_lo: int, y_hi: int) -> np.ndarray:
        img = np.zeros((h, w), dtype=bool)
        y = int(rng.integers(y_lo, y_hi + 1))
        for x in range(3, w - 3):
            img[y, x] = True
            y = int(np.clip(y + rng.integers(-1, 2), y_lo, y_hi))
        return img

    def _curve_3d(d: int, h: int, w: int, z_lo: int, z_hi: int) -> np.ndarray:
        vol = np.zeros((d, h, w), dtype=bool)
        y = h // 2
        z = int(rng.integers(z_lo, z_hi + 1))
        for x in range(3, w - 3):
            vol[z, y, x] = True
            z = int(np.clip(z + rng.integers(-1, 2), z_lo, z_hi))
        return vol

    def _tube(skel_bool: np.ndarray) -> np.ndarray:
        return ndi.binary_dilation(skel_bool, structure=skeleton_tube_footprint(skel_bool.ndim))

    def _fg_onehot(label_b1: torch.Tensor, c: int) -> torch.Tensor:
        """(B,1,...) 标签图 → (B,C-1,...) 前景 one-hot（丢背景通道），2D/3D 通吃。"""
        lbl = label_b1[:, 0].long()
        if lbl.dim() == 3:  # (B,H,W)
            return F.one_hot(lbl, c).permute(0, 3, 1, 2)[:, 1:]
        return F.one_hot(lbl, c).permute(0, 4, 1, 2, 3)[:, 1:]  # (B,D,H,W)

    def _make_case_2d() -> tuple[torch.Tensor, torch.Tensor]:
        """2D 多类用例：C=4，GT 只含类 1/2 的细管（类 3 骨架通道为空）。

        返回 (gt 标签图 (B,1,H,W) long, skeleton 标签图 (B,1,H,W) long)。
        """
        b, c, h, w = 2, 4, 32, 32
        seg = np.zeros((b, h, w), dtype=np.int64)
        skel = np.zeros((b, h, w), dtype=np.int64)
        for i in range(b):
            for cls, (lo, hi) in enumerate([(4, 12), (20, 28)], start=1):
                tube = _tube(_curve_2d(h, w, lo, hi))
                seg[i][tube] = cls
                skel[i] += tube * cls  # 骨架管道裁回 GT 并按类着色（官方 skel·seg）
        return (torch.from_numpy(seg).view(b, 1, h, w),
                torch.from_numpy(skel).view(b, 1, h, w))

    def _make_case_3d() -> tuple[torch.Tensor, torch.Tensor]:
        """3D 多类用例：C=3，GT 只含类 1/2 的细管（两管 z 区间分离、不重叠）。"""
        b, c, d, h, w = 2, 3, 14, 12, 16
        seg = np.zeros((b, d, h, w), dtype=np.int64)
        skel = np.zeros((b, d, h, w), dtype=np.int64)
        for i in range(b):
            for cls, (lo, hi) in enumerate([(2, 3), (8, 9)], start=1):
                tube = _tube(_curve_3d(d, h, w, lo, hi))
                seg[i][tube] = cls
                skel[i] += tube * cls
        return (torch.from_numpy(seg).view(b, 1, d, h, w),
                torch.from_numpy(skel).view(b, 1, d, h, w))

    def _naive_paper_ref(prob_fg64: np.ndarray, skel_fg64: np.ndarray) -> tuple[float, np.ndarray]:
        """论文式(1) 朴素双精度参考：无任何平滑项，空骨架通道经 valid 掩码剔除。

        入参均为前景口径：prob_fg64/skel_fg64 形状 (B, C-1, ...)。
        """
        axes = tuple(range(2, prob_fg64.ndim))
        inter = (prob_fg64 * skel_fg64).sum(axes)
        sgt = skel_fg64.sum(axes)
        valid = sgt > 0
        return float(-(inter[valid] / sgt[valid]).mean()), valid

    def _naive_official_ref(prob_fg64: np.ndarray, skel_fg64: np.ndarray, smooth: float,
                            clamp_min: float) -> float:
        """官方公式朴素双精度参考：(inter+smooth)/clip(sum_gt+smooth, clamp_min) 取负平均。"""
        axes = tuple(range(2, prob_fg64.ndim))
        inter = (prob_fg64 * skel_fg64).sum(axes)
        sgt = skel_fg64.sum(axes)
        return float(-((inter + smooth) / np.clip(sgt + smooth, clamp_min, None)).mean())

    gt2, sk2 = _make_case_2d()
    gt3, sk3 = _make_case_3d()
    onehot2 = F.one_hot(gt2[:, 0].long(), 4).permute(0, 3, 1, 2)   # (B,4,H,W)
    onehot3 = F.one_hot(gt3[:, 0].long(), 3).permute(0, 4, 1, 2, 3)  # (B,3,D,H,W)

    # 1) 论文式(1)对拍：smooth=0 与朴素 float64 参考（无稳定项）对拍，空骨架
    #    通道剔除后取平均；2D（C=4，类 3 天然为空通道）与 3D
    for (onehot, sk, name) in [(onehot2, sk2, "2D"), (onehot3, sk3, "3D")]:
        prob64 = onehot.double()  # 完美预测 = GT one-hot（float64）
        skel_fg64 = _fg_onehot(sk, prob64.shape[1]).double()
        got_none = skeleton_recall_loss(prob64, sk, smooth=0.0, reduction="none")
        ref, valid = _naive_paper_ref(prob64[:, 1:].numpy(), skel_fg64.numpy())
        assert torch.isfinite(got_none).all()
        got_valid = float(got_none.numpy()[valid].mean())
        assert abs(got_valid - ref) < 1e-10, (got_valid, ref)
        print(f"PASS 论文式(1)对拍（{name}，smooth=0 vs 朴素 float64）："
              f"max|Δ|={abs(got_valid - ref):.2e}")

    # 1b) 空骨架通道剔除口径：抹掉半个 batch 的 GT/骨架后，空通道与非空通道混合，
    #     剔除空通道后仍与朴素参考一致
    gt3e, sk3e = gt3.clone(), sk3.clone()
    gt3e[0] = 0
    sk3e[0] = 0
    prob64 = F.one_hot(gt3e[:, 0].long(), 3).permute(0, 4, 1, 2, 3).double()
    skel_fg64 = _fg_onehot(sk3e, 3).double()
    got_none = skeleton_recall_loss(prob64, sk3e, smooth=0.0, reduction="none")
    ref, valid = _naive_paper_ref(prob64[:, 1:].numpy(), skel_fg64.numpy())
    assert not valid.all() and valid.any(), "应恰有部分 (样本,通道) 为空骨架"
    assert abs(float(got_none.numpy()[valid].mean()) - ref) < 1e-10
    print("PASS 论文式(1)对拍（3D 空骨架通道剔除口径）：max|Δ|="
          f"{abs(float(got_none.numpy()[valid].mean()) - ref):.2e}")

    # 2) 官方口径对拍：默认 smooth=1e-5 + clamp_min=1e-8 与官方公式 float64 参考
    #    全量对拍（含空骨架通道 → rec=1 → 该通道贡献 -1 的官方语义），2D/3D
    for (onehot, sk, name) in [(onehot2, sk2, "2D"), (onehot3, sk3, "3D")]:
        prob64 = onehot.double()
        skel_fg64 = _fg_onehot(sk, prob64.shape[1]).double()
        got = skeleton_recall_loss(prob64, sk)
        ref = _naive_official_ref(prob64[:, 1:].numpy(), skel_fg64.numpy(), 1e-5, 1e-8)
        assert abs(got.item() - ref) < 1e-12, (got.item(), ref)
        print(f"PASS 官方口径对拍（{name}，smooth=1e-5+clamp，含空通道=-1）："
              f"|Δ|={abs(got.item() - ref):.2e}")

    # 3) 输入形式等价：(B,1,..) 标签图 与 (B,C,..) one-hot 两种骨架输入必须给出
    #    完全相同的损失（官方两个分支的等价性）
    oh_skel2 = F.one_hot(sk2[:, 0].long(), 4).permute(0, 3, 1, 2).float()
    l_label = skeleton_recall_loss(onehot2.float(), sk2, reduction="none")
    l_onehot = skeleton_recall_loss(onehot2.float(), oh_skel2, reduction="none")
    assert torch.equal(l_label, l_onehot), (l_label, l_onehot)
    print("PASS 骨架输入形式等价：标签图 scatter ≡ one-hot 直传（逐元素相等）")

    # 4) 方向性：预测在骨架处断裂（一条横贯带的概率整条归给背景）的损失必须显著
    #    大于预测完整覆盖骨架的情形（覆盖 → rec=1 → loss=-1）
    def _broken_probs(onehot: torch.Tensor) -> torch.Tensor:
        p = onehot.float().clone()
        if p.dim() == 4:      # 2D：中间 6 列划给背景，拦腰截断管道
            p[..., :, 14:20] = 0.0
            p[:, 0, :, 14:20] = 1.0
        else:                 # 3D：中间 3 层划给背景
            p[..., 6:9, :, :] = 0.0
            p[:, 0, 6:9, :, :] = 1.0
        return p

    l_cover2 = skeleton_recall_loss(onehot2.float(), sk2)
    l_broken2 = skeleton_recall_loss(_broken_probs(onehot2), sk2)
    assert l_broken2 > l_cover2 + 0.1, (l_broken2.item(), l_cover2.item())
    assert abs(l_cover2.item() + 1.0) < 1e-4, l_cover2.item()
    l_cover3 = skeleton_recall_loss(onehot3.float(), sk3)
    l_broken3 = skeleton_recall_loss(_broken_probs(onehot3), sk3)
    assert l_broken3 > l_cover3 + 0.1, (l_broken3.item(), l_cover3.item())
    print(f"PASS 骨架断裂方向性：2D {l_broken2.item():.4f} > 覆盖 {l_cover2.item():.4f}；"
          f"3D {l_broken3.item():.4f} > 覆盖 {l_cover3.item():.4f}")

    # 5) forward + backward：logits 入口（from_logits=True）多类 2D/3D，损失与
    #    梯度均有限；骨架/GT 侧（numpy/scipy 预计算结果）不参与梯度
    for (onehot, sk, name) in [(onehot2, sk2, "2D"), (onehot3, sk3, "3D")]:
        logits = (onehot.float() * 4.0 - 2.0) + torch.randn_like(onehot.float()) * 2.0
        logits.requires_grad_(True)
        assert not sk.requires_grad
        loss = SkeletonRecallLoss(from_logits=True)(logits, sk)
        loss.backward()
        assert torch.isfinite(loss), loss
        assert torch.isfinite(logits.grad).all(), "logits.grad 含 NaN/Inf"
        assert sk.grad is None
        assert -1.0 - 1e-6 <= loss.item() <= 1e-6, loss.item()  # 官方口径 ∈ [-1, 0]
        print(f"PASS forward+backward（{name} 多类 logits）：loss={loss.item():.4f}，"
              f"梯度有限，损失 ∈ [-1,0]")

    # 6) footprint 形状与半径：2D (5,5)、3D (5,5,5)，恰好覆盖曼哈顿距离 ≤ 2 的点集
    fp2 = skeleton_tube_footprint(2)
    fp3 = skeleton_tube_footprint(3)
    assert fp2.shape == (5, 5) and fp3.shape == (5, 5, 5)
    for fp, ndim in [(fp2, 2), (fp3, 3)]:
        ctr = tuple(s // 2 for s in fp.shape)  # 5×(5...) 足迹的中心在 size//2
        for idx in np.ndindex(*fp.shape):
            manh = sum(abs(i - c) for i, c in zip(idx, ctr))
            assert bool(fp[idx]) == (manh <= 2), (idx, manh)
    try:
        skeleton_tube_footprint(4)
        raise AssertionError("ndim=4 应抛 ValueError")
    except ValueError:
        pass
    print("PASS 管道膨胀 footprint：2D (5,5)/3D (5,5,5)，曼哈顿半径恰为 2，非法 ndim 拒绝")

    # 7) dilate_skeleton_to_tube：管道 ⊇ 骨架、不含曼哈顿距离 > 2 的点、裁回/着色
    #    正确；skimage 可用时与官方 skimage 膨胀逐像素对拍（缺席则 SKIP）
    curve = _curve_2d(32, 32, 6, 26)
    tube = dilate_skeleton_to_tube(curve)
    assert tube.dtype == np.bool_ and (tube >= curve).all(), "管道必须包含骨架本身"
    far = ndi.binary_dilation(curve, structure=ndi.iterate_structure(
        ndi.generate_binary_structure(2, 1), 3))
    assert not (tube & ~far).any(), "管道不得包含曼哈顿距离 > 2 的点"
    seg = tube.astype(np.int64) * 3  # 造一个值=3 的标签图验证 clip_to 裁回/着色
    clipped = dilate_skeleton_to_tube(curve, clip_to=seg)
    assert (clipped == seg * tube).all() and np.issubdtype(clipped.dtype, np.integer)
    try:
        from skimage.morphology import dilation as _sk_dilation
        skel_int = (curve > 0).astype(np.int16)
        ref_tube = _sk_dilation(skel_int, footprint=skeleton_tube_footprint(2)) > 0
        assert (tube == ref_tube).all(), "与官方 skimage 膨胀不一致"
        print("PASS 管道膨胀 helper：半径性质 + 裁回着色 + 与官方 skimage 膨胀逐像素一致")
    except ImportError:
        print("SKIP skimage 缺席：官方膨胀对拍跳过（半径性质与裁回已验证）")

    # 8) 退化路径：C==1 拒绝；全空骨架（官方口径 rec=1 → loss=-1，有限非 NaN）；
    #    随机噪声概率下损失有限且 ∈ [-1, 0]
    try:
        skeleton_recall_loss(torch.rand(1, 1, 8, 8), torch.zeros(1, 1, 8, 8, dtype=torch.long))
        raise AssertionError("C==1 应抛 ValueError")
    except ValueError:
        pass
    l_empty = skeleton_recall_loss(torch.rand(2, 3, 8, 8),
                                   torch.zeros(2, 1, 8, 8, dtype=torch.long))
    assert torch.isfinite(l_empty) and abs(l_empty.item() + 1.0) < 1e-6, l_empty.item()
    skel_rand = torch.from_numpy(
        _tube(_curve_2d(16, 16, 4, 11)).astype(np.int64)).view(1, 1, 16, 16)
    l_noise = skeleton_recall_loss(torch.randn(2, 3, 16, 16), skel_rand.repeat(2, 1, 1, 1),
                                   from_logits=True)
    assert torch.isfinite(l_noise) and -1.0 - 1e-6 <= l_noise.item() <= 1e-6
    print(f"PASS 退化路径：C==1 拒绝；空骨架 loss={l_empty.item():.4f}（官方口径 -1）；"
          f"噪声概率 loss={l_noise.item():.4f} ∈ [-1,0]，均有限")

    # 9) loss_mask：与官方语义一致——骨架统计与召回统计同乘掩码，float64 对拍
    mask = torch.zeros(1, 1, 32, 32)
    mask[:, :, :, :20] = 1.0
    prob64 = onehot2[:1].double()
    skel_fg64 = _fg_onehot(sk2[:1], 4).double()
    got = skeleton_recall_loss(prob64, sk2[:1], loss_mask=mask)
    inter = (prob64[:, 1:] * skel_fg64 * mask.double()).sum((2, 3))
    sgt = (skel_fg64 * mask.double()).sum((2, 3))
    ref = float((-(inter + 1e-5) / np.clip(sgt.numpy() + 1e-5, 1e-8, None)).mean())
    assert abs(got.item() - ref) < 1e-12, (got.item(), ref)
    print(f"PASS loss_mask（ignore 语义）：|Δ|={abs(got.item() - ref):.2e}")

    print("PASS 全部自测通过")
