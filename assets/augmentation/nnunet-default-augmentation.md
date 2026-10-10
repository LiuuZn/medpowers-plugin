# nnU-Net 默认增强组合（nnunet-default-augmentation）

类别：`[augmentation]`

## 论文支撑（统一四问）

**支撑论文 1：nnU-Net（Isensee 等，DKFZ）**

- **哪篇论文用了它**：Isensee, F., Jaeger, P.F., Kohl, S.A.A. 等，nnU-Net，Nature Methods 2021 18:203–211：<https://www.nature.com/articles/s41592-020-01008-z>；方法论文全文 arXiv:1904.08128（ar5iv：<https://ar5iv.labs.arxiv.org/html/1904.08128>）。默认参数一手核自 PyPI nnunet==1.7.0 的 `data_augmentation_moreDA.py` + `nnUNetTrainerV2.py`，以及 nnunetv2 2.6.4 源码。
- **目的**：用一套对所有分割任务通用的固定默认增强，免除逐任务调参——这是 nnU-Net"自动配置"设计的组成部分。
- **效果**：含此增强组合的整条流水在 49 项任务中 29 项目标结构刷新 SOTA、MSD 10 项中 7 项第一（arXiv:1904.08128 核实）。
- **结论**：增强全套默认照用、不逐项调参。2018 预印本自述增强参数 "were not properly validated"（未经恰当验证）——默认值来自经验而非系统消融。

**支撑论文 2：nnU-Net Revisited（Isensee 等，MICCAI 2024）**

- **哪篇论文用了它**：Isensee 等，MICCAI 2024，arXiv:2404.09556：<https://arxiv.org/abs/2404.09556>。
- **目的**：推出 ResEnc L/M/XL 大容量新配方冲击新基准，检验默认增强是否需要随之改变。
- **效果**：新配方对增强**零改动**，原文 "augmentations were implemented as in the standard nnU-Net framework"；2023–2024 全部候选架构（含 Transformer/Mamba）都在这套增强下训练。
- **结论**：从 nnU-Net v1 到 ResEnc XL，模型规模跨多个数量级，这套增强组合保持不变——行业事实上的默认基线。

## 机制与动机

组合按"空间 → 强度 → 采样"三层覆盖不同变异源：

1. **空间**：旋转 + 缩放在 `SpatialTransform` 内一次插值完成（避免多次插值累加伪影）；三轴镜像各自独立。
2. **强度**：高斯噪声、高斯模糊、亮度乘法、对比度、Gamma（正/反向），模拟扫描仪与序列差异。
3. **采样**：前景过采样（每 batch 33% patch 强制落在前景类）保证小目标的出现频率。

低分辨率模拟（p=0.25）与弹性形变（默认关闭）也是成员，分别见 [low-resolution-simulation.md](low-resolution-simulation.md) 与 [elastic-deformation.md](elastic-deformation.md)。

v1 → v2 的差异（以代码核实值为准）：v2 彻底移除弹性形变参数；缩放改为跨轴同步；其余变换概率与 v1 相同。

## 适用（什么时候用）

- **模态/维度**：CT/MRI、2D/3D 通用；几十~几百例的小数据分割是这套组合的主场——数据越少，越依赖在线增强提供变异。
- **数据规模规律（轻增强反例证据）**：数据规模越大增强越轻。MedSAM（1570 万图像-掩码对）Methods 原文 "data augmentation was not used"，唯一扰动是 box prompt 随机移 0–20px（<https://pmc.ncbi.nlm.nih.gov/articles/PMC10803759>）；MONAI Auto3DSeg 模板（research-contributions 仓库 segresnet/scripts/segmenter.py，一手核实）与 nnU-Net 同构但更轻（flip/affine/噪声/模糊概率 0.2–0.5、强度扰动范围更宽）；wholeBody_ct_segmentation 官方 model-zoo bundle 的随机增强只剩 RandSpatialCropd 96³；SwinUNETR（<https://arxiv.org/abs/2111.14791>）微调 flip/rotation 概率仅 0.1–0.3。百万级预训练或大数据微调时，照抄全套默认未必最优。
- **跟随证据**：TotalSegmentator（<https://pmc.ncbi.nlm.nih.gov/articles/PMC10546353>）自述用 nnU-Net 框架、无公开偏离；Merlin（<https://arxiv.org/abs/2406.06512>）分割微调显式回归 nnU-Net 默认增强；VoCo（<https://arxiv.org/abs/2402.17300>）微调用 MONAI 标准管线，BTCV 80.53→83.85。

## 预期收益

- **组合级战绩**：49 项任务 29 项 SOTA、MSD 7/10 第一（流水级，不可拆分归因）。
- **唯一直接消融数字（镜像）**：STU-Net（<https://arxiv.org/abs/2304.06716>，基于 nnU-Net 1.7.0）Table 8：预训练启用镜像使下游迁移 Dice 86.16→88.25（+2.09，FLARE22）、75.88→77.04（+1.16，AutoPET）；微调不用镜像则掉到 87.65/76.34——预训练与微调**同时启用**镜像最佳。
- **消融证据稀缺本身是结论**：Nature 2021 的增强单项消融表在付费墙补充材料内未能核实；全行业实践是"全套默认、不逐项调参"。任何单项改动都要靠自己的实验记录（双时点）验证。

## 风险与坑

- **引用陷阱：论文附录 ≠ 代码默认**。至少五处不一致（高斯噪声 p、模糊 σ、亮度/对比度范围、Gamma 概率），以代码核实值为准（见下表）；引用时注明出处是附录还是代码。
- **弹性形变叙述陷阱**：论文正文长期保留弹性形变描述，代码 v1.7.0 起默认禁用、v2 彻底移除（详见 [elastic-deformation.md](elastic-deformation.md)）。
- **不要凭感觉逐项调参**：默认集未经系统消融，但无证据支持的单项改动同样不可信；要动就配实验记录与预期信号。
- **各向异性数据**不能直接套 3D 大角度旋转，组合内部靠"2D 收窄 + dummy 2D"规避（见 [anisotropy-aware-augmentation.md](anisotropy-aware-augmentation.md)）。

## 参数与经验值（nnunet==1.7.0 / nnunetv2 2.6.4 代码核实值）

| 变换 | 参数 | 概率 |
|---|---|---|
| 旋转+缩放（一次插值） | 3D 各向同性三轴各 ±30°；2D 面内 ±180°（patch 长短边比>1.5 收窄 ±15°）；缩放 U(0.7, 1.4) | 各 p=0.2 |
| 镜像 | 三轴各自独立 | 每轴 0.5 |
| 高斯噪声 | 方差 U(0, 0.1) | p=0.1（论文附录写 0.15） |
| 高斯模糊 | σ ∈ (0.5, 1.0) | p=0.2（论文附录写 σ(0.5,1.5)） |
| 亮度乘法 / 对比度 | (0.75, 1.25) | 各 p=0.15（论文附录写 (0.7,1.3)/(0.65,1.5)） |
| 低分辨率模拟 | zoom U(0.5, 1) | p=0.25 |
| Gamma | γ∈(0.7, 1.5)，retain_stats；反向=先反转再幂变换 | 正向 p=0.3 + 反向 p=0.1（论文附录 0.15+0.15） |
| 弹性形变 | 默认关闭（备用 α(0,900)、σ(9,13)） | — |
| 前景过采样 | 每 batch 33% patch 强制采样在前景类（v1/v2 一致） | — |

dummy 2D：z 轴强各向异性时把 3D patch 临时转 2D 逐切片做面内增强（±180°）再转回；此时弹性 α 降为 (0,200)（默认仍关闭）。

## 实现参考

- nnU-Net v1：PyPI nnunet==1.7.0 `data_augmentation_moreDA.py`（变换组装与参数）+ `nnUNetTrainerV2.py`（开关）。
- nnU-Net v2：nnunetv2 2.6.4 源码（同名训练器体系）。
- 更轻的同构模板：MONAI Auto3DSeg（research-contributions 仓库 segresnet/scripts/segmenter.py）——大数据微调时的参照。

## 相关卡

- [foreground-oversampling.md](foreground-oversampling.md) — 组合的采样层成员
- [anisotropy-aware-augmentation.md](anisotropy-aware-augmentation.md) — 各向异性数据上的修正规则
- [low-resolution-simulation.md](low-resolution-simulation.md) — p=0.25 的成员变换
- [elastic-deformation.md](elastic-deformation.md) — 被此组合弃用的成员
- [mri-bias-field-augmentation.md](mri-bias-field-augmentation.md) / [ct-hu-perturbation.md](ct-hu-perturbation.md) — 更专门的模态强度域增强
- [test-time-augmentation.md](test-time-augmentation.md) — 训练侧镜像与推理侧 TTA 的对应

## 实验反链

暂无。
