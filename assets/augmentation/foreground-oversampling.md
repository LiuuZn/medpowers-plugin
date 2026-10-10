# 前景过采样（foreground-oversampling）

类别：`[augmentation]`

## 论文支撑（统一四问）

- **哪篇论文用了它**：Isensee, F., Jaeger, P.F., Kohl, S.A.A. 等（DKFZ），**nnU-Net**，Nature Methods 2021 18:203–211：<https://www.nature.com/articles/s41592-020-01008-z>；方法论文 arXiv:1904.08128（ar5iv：<https://ar5iv.labs.arxiv.org/html/1904.08128>）。33% 比例一手核自 PyPI nnunet==1.7.0（`nnUNetTrainerV2.py`）与 nnunetv2 2.6.4 源码，v1/v2 一致。
- **目的**：patch 训练下，随机均匀采样会采到大量纯背景 patch，小目标/稀有类别在 batch 中出现频率过低、前景梯度信号稀薄；强制一部分 patch 采样在前景类上保证前景始终参与监督。
- **效果**：无单独消融核实（nnU-Net 的增强单项消融在 Nature 2021 付费墙补充材料内未能核实）；作为固定默认进入整条流水——49 项任务 29 项目标结构刷新 SOTA、MSD 10 项中 7 项第一。
- **结论**：33% 是默认常数，v1/v2 两代一致，工程上被视为"不可拆的默认项"，无需按任务调参。

## 机制与动机

分割训练以 patch 为单位：一个 3D patch 相对整例只是局部视野，小结构（病灶、细小器官、神经血管）在均匀随机采样下大量 patch 完全不含前景，梯度被背景主导。前景过采样把每 batch 约 1/3 的 patch 强制采样到前景类体素上，其余 2/3 仍均匀随机——保留足量背景 patch 以维持假阳性压力。它不改变图像内容，只改变**样本分布**，因此与几何/强度增强正交，可叠加使用；它解决的问题与类不平衡损失（加权 CE、Dice 族）同向但层次不同：一个在采样层、一个在损失层。

## 适用（什么时候用）

- **任务**：分割（2D/2.5D/3D patch 训练）；目标占图比例小、类别不平衡（稀有结构）时收益最大。
- **数据量级**：几十~几百例小数据——目标本来就少，均匀采样更难命中前景。
- **不适用**：分类任务（无 patch 前景概念）；目标几乎充满视野的任务收益趋零。
- **与预处理联动**：先行前景裁剪（[../preprocessing/foreground-crop.md](../preprocessing/foreground-crop.md)）后，均匀采样的前景密度已提高，但小结构稀疏问题仍在——两者是串联关系，不是替代。

## 预期收益

- **无单独消融数字**（如实注明）：nnU-Net 论文未提供前景过采样的独立消融。
- 定性结论：v1/v2 两代框架固定启用且比例一致；组合级新配方对增强零改动（arXiv:2404.09556，见 [nnunet-default-augmentation.md](nnunet-default-augmentation.md)）。
- 整条流水的组合战绩（29/49 SOTA、MSD 7/10）含此成分，但不可单独归因。

## 风险与坑

- **放大标注错误**：过采样提高含前景 patch 的频率，标注缺失/错标的前景同样被放大。模型进入平台期时先测标注一致性上限（methods/label-ceiling-diagnosis.md），再怀疑增强策略。
- **不是外观变异**：它不产生新的形态/强度/噪声样本，不能替代几何与强度增强；数据量瓶颈仍在增强组合的其他成员上。
- **比例不是越大越好**：前景 patch 占比过高会削弱背景负样本覆盖、推高假阳性；nnU-Net 取 33% 而非更高，且无消融支持擅自调大。
- **前景判定依赖标注质量**：训练前先满足数据集登记的 `annotation` 标注契约（dataset.json 的 produced_by/rule/tolerance/verified），标注未核实前不建议叠加激进采样策略。

## 参数与经验值

- **33%**：每 batch 强制采样在前景类上的 patch 比例；nnunet==1.7.0 与 nnunetv2 2.6.4 代码核实，v1/v2 一致。
- **其余 2/3 patch**：均匀随机采样（允许纯背景）。
- **无调参建议**：论文与代码均无该比例的消融；默认照用。

## 实现参考

- nnU-Net v1：PyPI nnunet==1.7.0 `nnUNetTrainerV2.py`（oversampling 逻辑）。
- nnU-Net v2：nnunetv2 2.6.4 源码（patch 采样部分）。
- 本目录卡片不配代码；实现口径以 nnU-Net 官方源码为准。

## 相关卡

- [nnunet-default-augmentation.md](nnunet-default-augmentation.md) — 所属组合级策略卡
- [lesion-copy-paste.md](lesion-copy-paste.md) — 病灶复制粘贴：提高稀有目标出现率的另一手段
- [bcp-semi-supervised-copy-paste.md](bcp-semi-supervised-copy-paste.md) — 半监督设定下的复制粘贴
- [../preprocessing/foreground-crop.md](../preprocessing/foreground-crop.md) — 同一"前景"概念在预处理侧的对应物

## 实验反链

暂无。
