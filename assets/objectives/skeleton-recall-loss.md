# skeleton-recall-loss —— Skeleton Recall 损失（细管状结构连通性）

## 论文支撑（统一四问）

- **哪篇论文用了它**：Yannick Kirchhoff、Maximilian R. Rokuss、Saikat Roy、Balint Kovacs、Constantin Ulrich、Tassilo Wald、Maximilian Zenk、Philipp Vollmuth、Jens Kleesiek、Fabian Isensee、Klaus Maier-Hein（德国癌症研究中心 DKFZ 团队），《Skeleton Recall Loss for Connectivity Conserving and Resource Efficient Segmentation of Thin Tubular Structures》，ECCV 2024，https://arxiv.org/abs/2404.03010 。本卡依据 arXiv v2（2024-07-17，ECCV 正式版）摘要页与 HTML 全文整理，并对照官方代码 MIC-DKFZ/Skeleton-Recall（nnU-Net 集成）核对实现口径。
- **目的**：血管、神经、气道等细管状结构分割中，Dice/CE 只优化体素重叠、不约束拓扑连通性，管子断成几截会毁掉下游血流计算与导航；已有拓扑感知损失（clDice 等）要对**预测**做可微骨架化，GPU 开销随 3D/类别数爆炸（clDice 在 A100 40GB 上 13 类直接 OOM）。论文把骨架化挪到 CPU 数据加载/离线阶段、只对 GT 骨架管道算预测的 soft recall。
- **效果**：ToothFairy（3D 牙槽神经管）Dice 71.80→74.42、clDice 指标 89.16→92.05（+clDice loss 仅 72.36/89.67）；TopCoW 13 类多类血管 Dice 85.36→86.59、clDice 93.68→94.35（clDice loss OOM 无结果）；TopCoW 二类 Dice 93.55→93.72、clDice 98.25→98.48；2D Roads/DRIVE/Cracks 多数指标最优。开销仅训练时长 +8%、显存 +2%（clDice 为 +88% / +52%），总开销降低 90% 以上。后续被 RSNA 2025 颅内动脉瘤检测挑战赛第一名方案采用（官方仓库 readme）。
- **结论**：资源高效（开销降一个数量级）与连通性 SOTA 可以兼得，是首个支持多类设定的细管结构连通性损失；骨架质量依赖预处理，与 CE+Dice 组合即可（连接性权重 w ∈ {0.1, 1.0} 两档足够，作者刻意避免大量调参）。适用边界：细管状结构；粗大器官或对管径本身有要求的场景不是它的目标。

## 类别标签

`[loss]`（可导损失；对预测概率可微、梯度自动微分，骨架/GT 侧是预计算常数不参与梯度）

## 形式

**管道构造**（CPU 离线/数据加载阶段，官方 SkeletonTransform）：GT 标签图二值化 → 骨架化（2D Zhang-Suen / 3D Lee 薄化）→ 用十字形（连通性 1）结构元迭代 2 次的 footprint（曼哈顿半径 2 菱形，论文称 "diamond kernel of radius 2"）膨胀成管道 → 乘回 GT 标签图，把管道裁回 GT 内并按类别着色，作为追加通道。

**损失**（论文 Section 3.2 式(1)）：

```
L_SkelRecall = −(1/|C|) Σ_{c∈C} [ Σ_i Y_skel,i,c · Ŷ_i,c / Σ_i Y_skel,i,c ]
```

Ŷ 为 softmax 预测概率，Y_skel 为（管道化、按类着色的）GT 骨架，分子是骨架管道内被预测覆盖的概率质量、分母是管道体素数——即**预测在骨架管道内的 soft recall 取负、按类平均**。训练时与通用损失组合（论文式(2)）：

```
L = L_generic + w · L_connectivity    （官方 nnU-Net：CE + Soft Dice + 1·SkelRecall）
```

官方实现为数值稳定在分子分母各加 smooth=1e-5 并对分母 clamp(min=1e-8)（论文式(1)本身无平滑项）；平滑后损失值域为 [-1, 0]，完整覆盖骨架管道时 = -1。

## 适用

- 2D/3D、二类/多类分割；目标是血管、神经、气道、牙槽神经管、脑血管（Willis 环）等**细管状结构**，也可用于道路、裂缝（论文 2D 实验）。
- 3D 多类场景是相对 clDice 的决定性优势（clDice 开销随类数线性增长、13 类即 OOM，本方法显存仅 +2%）。
- 断裂代价高的临床场景：管状结构连通性影响血流/导航/计数等下游任务。
- 骨架需**离线预计算**（或数据加载阶段 CPU 计算）——骨架提取不在训练图内；本仓库参考实现亦把骨架作为输入参数，骨架提取（如 `skimage.morphology.skeletonize`，2D/3D 均可）属任务仓库预处理步骤。
- 2×RTX 4090（23 GB/卡）预算下无显存压力（+2%），AMP 训练照常。

## 行为特征

- 损失 = -recall ∈ [-1, 0]（官方 smooth 口径）：预测完整盖住骨架管道 → -1，骨架处断裂立即推高（参考实现自测：2D 完美 -1.0 → 断裂 -0.86）；梯度只流向骨架管道内的预测概率。
- **只约束召回、不约束精确率**：管道外假阳不受罚（由 Dice/CE 负责），所以它教会网络"别断"，不教"别粗"——必须与 Dice+CE 组合使用，单独使用会过度膨胀。
- 对细结构断裂极敏感（管道体素少，断一段 recall 掉一大截），这正是设计意图；对整体形状/管径不敏感。
- 观察指标：与 clDice 指标（拓扑连通性）和 Dice 配合看；训练中 clDice 指标应系统性高于纯 nnU-Net 基线（论文 +2.89 / +0.67 / +0.10 不等），Dice 不应明显回落。
- 收敛动力学与基线 nnU-Net 几乎一致（开销 +8% 时长），无 early loss 崩塌类风险；骨架为常数项，不影响数值稳定性。

## 调参经验

- **组合权重**：CE + Soft Dice + w·SkelRecall，官方默认 `weight_ce = weight_dice = weight_srec = 1`；论文消融只扫 w ∈ {0.1, 1.0} 两档，先用 1.0，连通性过拟合/假阳失控再降 0.1。
- **膨胀半径**：官方固定为十字形结构元迭代 2 次（曼哈顿半径 2）。半径是"断裂容差"而非管径参数：预测只要盖住管道（骨架 ±2 体素）即拿满 recall。极细管（1-2 体素直径）无需调整；粗管场景若想放宽容差，可增大 `ndi.iterate_structure` 的迭代数（footprint 见参考实现 `skeleton_tube_footprint`），但论文未消融此参数，改动需自证。
- **smooth**：跟官方 1e-5 即可（分子分母各加）；设 0 得到严格论文式(1)，但空骨架通道行为改变（见"坑"）。
- **工程**：骨架化放数据加载阶段的 transform（官方做法，追加一个通道，随 nnU-Net 增广自动同步翻转/缩放）；deep supervision 场景官方提供 `skeleton_deep_supervision` 开关（默认关，关时骨架只在全分辨率监督）。骨架化本身约 0.05s/样本（论文 CPU 实测口径），可离线缓存。
- 论文 5 个数据集均未调膨胀半径/骨架算法，"两档权重 + 默认半径"即可复现其结果——优先把精力放在标注质量和 Dice 基线上。

## 坑

- **骨架质量依赖预处理**：标注本身断裂时骨架也断，损失会逼网络把"标注的断"连成"连通的错"——上此损失前先确认标注契约（produced_by/rule/tolerance/verified）与标注一致性上限（见 methods/label-ceiling-diagnosis.md）；骨架化毛刺同样会变成错误约束。
- **空骨架通道 = -1 的坑**：官方 smooth 口径下，某类骨架为空（该类不在 GT 或被 ignore mask 抠掉）时 rec = smooth/clip(smooth) = 1，该通道损失贡献 -1，直接拉低总损失。批量内空类别多时要么心里有数，要么调用侧剔除空通道 / 设 smooth=0。
- **只管召回不管精确率**：单独使用会倾向把预测膨胀过管径（假阳免费）；永远是 CE+Dice+SkelRecall 三件套，不要单飞。
- **不保证管径**：管道容差内的"贴骨架芯"预测即可拿满 recall，管径由 Dice 隐式约束；别误当"管径损失"用。
- **与 clDice 的取舍**：clDice 对预测做 soft 骨架化、拓扑约束更"硬"，但 GPU 开销 +88% 时长/+52% 显存且随类数线性涨；Skeleton Recall 只吃 GT 骨架、资源开销近乎免费，但约束是"管道容差内"的软约束。绝大多数场景选本方法；极端连通性要求且资源充裕时可实验两者组合（论文未做）。
- **骨架通道是类别索引标签图，不是二值掩码**：官方追加通道的值 = 该处 GT 类别索引（骨架·GT 着色），scatter 成 one-hot 后丢背景通道；接手他人实现先确认布局（参考实现自测第 3 项验证了标签图与 one-hot 两分支逐位等价）。
- **布局硬约束**：softmax 布局、通道 0=背景、C≥2；二值任务也要 (背景， 前景) 双通道输出（官方 scatter 分支恒丢通道 0，C==1 时会取空）。
- **贴边骨架**：官方用 skimage dilation（边界 reflect），scipy `binary_dilation`（边界不越界）在骨架贴边时可能差 1-2 圈；预处理时保证骨架距边界 ≥ 2 体素（裁剪/padding 策略里注意）。
- **nnU-Net 集成边界**：region-based 标注不支持（官方 trainer 直接 NotImplementedError）；ignore label 支持标注为 experimental；`batch_dice=True` 的跨样本求和聚合与默认逐 (样本，通道) mean 数值不同，对比实验口径要一致。

## 实验（EXP-*）反链

暂无（experiments/ 目前为空；后续实验启用本损失时在此处互加反链）。

## 参考实现

`assets/objectives/skeleton-recall-loss.py`（同目录）：论文式(1)(2)逐条对应（Section 3.2），含 `skeleton_recall_loss` / `SkeletonRecallLoss`（torch 损失，标签图与 one-hot 双骨架入口、loss_mask、smooth/clamp 口径对齐官方）与 `skeleton_tube_footprint` / `dilate_skeleton_to_tube`（官方同款半径 2 管道膨胀的 scipy 实现）；自测含与论文式(1)朴素双精度对拍（smooth=0，Δ=0）、官方口径对拍（Δ=0）、骨架断裂方向性、forward+backward 有限性、skimage 在场时与官方膨胀逐像素对拍。**与官方实现对拍结论**：同一批输入（2D/3D × 二类/多类 × 完整/断裂/噪声预测，骨架经官方同款 skimage skeletonize+膨胀流程构造）共 12 组，与官方 `SoftSkeletonRecallLoss` **逐位一致（float32，max|Δ| = 0）**。与官方/论文的 7 条不一致处（骨架提取外置、C≥2 布局收紧、smooth 数值口径、scipy 替换 skimage 膨胀、batch_dice 聚合不实现、loss_mask 保留、分母 dtype 提升修复）已在 .py 文件头逐条编号注明。
