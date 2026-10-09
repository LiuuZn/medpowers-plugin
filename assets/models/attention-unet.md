# attention-unet — 注意力门控 U-Net（Attention U-Net）

> 论文原文所读版本：arXiv:1804.03999（v3，2018-05-20 修订，MIDL'18 已接收版），经 ar5iv HTML 全文阅读（2026-10-05 抓取）；式(1)(2)(3) 编号与表 1–4 数字均出自该版本。官方代码：https://github.com/ozan-oktay/Attention-Gated-Networks

## 论文支撑（统一四问）

- **哪篇论文用了它**：Oktay、Schlemper 等（Imperial College London / Babylon Health 团队），《Attention U-Net: Learning Where to Look for the Pancreas》，MIDL 2018（<https://arxiv.org/abs/1804.03999>）。
- **目的**：在 U-Net 的跳跃连接上插入可微的加性注意力门控（AG），用粗尺度的门控信号自动抑制无关背景、突出小目标（胰腺），从而去掉级联 CNN 里额外的外部器官定位模型，单模型端到端训练。
- **效果**：CT-150 多类腹部分割（120 训练/30 测试）上胰腺 DSC 0.814±0.116 → 0.840±0.087，召回 0.806 → 0.841（p=.005），表面 S2S 距离 2.358 → 1.920 mm；训练集缩到 30 例时 DSC 0.741 → 0.767（p=.01）。代价仅参数 5.88M → 6.40M（约 +8%）、推理 0.167 → 0.179 s（160×160×96 输入前向）。同等容量对照（表 2）显示加 AG 优于把容量均匀加大：10.40M 的大 U-Net 胰腺 DSC 只有 0.825，低于 6.40M 的 Attention U-Net 0.840（p=.007）。TCIA CT-82 上从头训练 DSC 0.815 → 0.821，5 折交叉验证 81.48±6.23 DSC。
- **结论**：AG 以近零开销显著提高小器官分割的灵敏度（收益主要来自召回），单模型即可接近多模型级联的精度；对高对比度大器官（肾 DSC 0.963 → 0.964）几乎无增益——适用边界是"细小、形状变异大、对比度低的目标"。

## 核心机制

一句话：在每条跳跃连接上，把来自更粗尺度的上下文特征 g 与该级特征 x 分别经 1×1 卷积映射到中间维度后相加、ReLU（式 1），再经 1×1 卷积 + sigmoid 得到逐位置注意力系数 α∈[0,1]（式 2），输出 x̂=x·α——特征与梯度都只在任务相关区域按 α 加权通过（式 3：反向传播时背景区梯度被 α 下加权），相当于网络内部自动做软 ROI 选择。

## 量级

- 门控块自身参数 ≈ (F_l + F_g)·F_int + 2·F_int + 1（theta 无偏置；phi、psi 各带偏置，分别为 b_g∈R^F_int 与 b∈R^1），很小；论文整网仅 +8%（5.88M → 6.40M），前向时延 +7%（0.167 → 0.179 s）。
- 默认宽度说明：本参考实现默认 base_filters=32、levels=4（3D 实测 23.442M 参数），不对应论文表 1 规模——论文官方代码 feature_scale=4，通道 [16,32,64,256] 递进；base_filters=16 时实测 5.863M，接近论文 6.40M。上文"5.88M→6.40M、+8%"为论文事实，按原样保留。
- 2×4090（每卡约 23 GB）预算下完全可行：2D 全分辨率输入或 3D patch 训练都放得下，建议照常开 AMP；门控计算主要多在两次插值与几层 1×1 卷积，显存增量可忽略。

## 适用

- 数据量级：小/中数据尤其划算——论文 30 例训练仍有 +2.6% DSC 显著提升；不需要多阶段训练，标准端到端即可。
- 维度：2D/3D 通用（门控全部用 1×1/1×1×1 卷积；注意力图重采样 2D 用 bilinear、3D 用 trilinear）。
- 目标尺度：细小结构、类内形状变异大的目标（论文以胰腺为证）；目标大而对比度高时接近无增益。

## 短板与不适用场景

- sigmoid 软注意力只是"压低"不"清零"，极端前景/背景不平衡下背景假阳抑制有限；论文也承认残余/稠密连接、CRF 后处理等可与 AG 叠加。
- 第 1 层（最高分辨率）跳跃连接不挂门控——低级特征语义太弱，门控无意义；因此对"主要靠低级纹理/边缘"的分割任务收益有限。
- 门控信号来自粗尺度：若目标小到在瓶颈分辨率上已无响应，门控可能失效（此时应减小下采样深度或用带 DSV 的变体）。
- 论文实验在 2 mm 各向同性重采样后进行，原始分辨率极细小结构上的表现论文未直接验证。

## 经验备注

- 初始化（论文原文 + 官方 TORR 变体）：ψ 置零权重 + 偏置 3.0，sigmoid(3)≈0.95，起始近乎全通（对应论文"Gating parameters are initialised so that attention gates pass through feature vectors at all spatial locations"），训练初期等价普通 U-Net、随后逐渐聚焦。
- σ₂ 选 sigmoid 不选 softmax：论文明确 softmax 会让激活过稀、sigmoid 收敛更好（跨位置独立系数，不做跨位置归一化）。
- 变体：多门（每门关注器官子集，官方 `MultiAttentionBlock`）、深监督 DSV 版（官方 `unet_CT_single_att_dsv_3D.py`）；论文训练用 Adam、batch 2–4、BN、深监督、强度标准化 N(0,1)、Sørensen-Dice loss。
- 参考实现：`assets/models/attention-unet.py`（2D/3D 门控 + 完整 Attention U-Net，CPU 自测可跑）。
- 外部参考：官方仓库 `models/layers/grid_attention_layer.py`；本地兄弟项目 `/ssd/lx/net_arch/arch_Unet3_2D.py` 为普通 2D U-Net（无注意力），可作挂门控前的基线。

## 实验反链

- 暂无（`experiments/` 目前为空）。
