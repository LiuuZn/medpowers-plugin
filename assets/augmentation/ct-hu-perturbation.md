# CT HU/窗宽窗位扰动（ct-hu-perturbation）

类别：`[augmentation]`

## 论文支撑（统一四问）

- **哪篇论文用了它**：Zhang L, Wang X, Yang D, Roth H, Myronenko D, Xu D, Xu Z, Sanford T, Turkbey B, Wood BJ, Harmon S，BigAug（Generalizing Deep Learning for Medical Image Segmentation to Unseen Domains via Deep Stacked Transformations），IEEE TMI 2020;39(7):2531–2540：<https://pmc.ncbi.nlm.nih.gov/articles/PMC7393676>（PubMed 32070947）。
- **目的**：其中的"强度扰动（intensity perturbation：scale+shift）"成分在 HU 域做随机仿射偏移，模拟**重建核、辐射剂量、对比剂时相与阅片窗宽窗位**的差异，提升跨域分割泛化。
- **效果**：作为 BigAug 九种变换之一参与组合：三任务未见域平均 80.0%（传统增强仅 39%）；本成分**无单独消融数字**。
- **结论**：强度仿射扰动是 CT 跨域训练的基础成分，但必须作用在 HU 物理域、并与其他强度/几何变换组合使用（组合卡见 [bigaug-stacked-perturbation.md](bigaug-stacked-perturbation.md)）。

## 机制与动机

CT 的 HU 值有绝对物理含义（水≈0、空气≈−1000），但同一解剖在**不同重建核、不同剂量、不同对比剂时相**下 HU 分布整体漂移或压缩；阅片端又存在**窗宽窗位习惯差异**。训练时对 HU 域做随机 scale+shift（随后按训练口径重新归一化），把这些采集与显示变异搬进训练分布，让网络不再依赖某一台设备、某一套窗位的绝对强度习惯。

## 适用（什么时候用）

- **模态**：CT 分割/分类；跨中心、跨厂商、跨重建核部署前必配。
- **审慎扩展**：定量 MRI 参数图（DTI 参数图等）同理携带绝对物理量纲，扰动会破坏量纲，需谨慎。
- **数据量级**：无关；几十例小数据零成本受益。
- **2D/3D**：皆可。

## 预期收益

- 无本成分单独消融；组合证据（BigAug，IEEE TMI 2020）：源域 91.6% → 未见域 80.0%，传统增强 39%，CycleGAN（用目标域数据）63.5%。
- 旁证（均**未核实**，照实标注）：CT 分割鲁棒性压力测试把窗宽窗位扰动列为腐蚀轴之一（[arXiv:2606.00491](https://arxiv.org/abs/2606.00491)，2026，仅检索确认）；物理化低剂量 CT 增强（[PMC13366036](https://pmc.ncbi.nlm.nih.gov/articles/PMC13366036)，仅检索确认、数字未核实）。

## 风险与坑

- **归一化顺序是最关键的坑**：若管线先做固定窗宽窗位映射再归一化，扰动必须施加在**窗位之前的 HU 域**、之后重新归一化；直接扰动归一化后的图像会退化为普通亮度扰动，"跨窗宽窗位鲁棒"的物理意义尽失。
- **高对比结构对大扰动敏感**：骨骼、钙化等结构靠 HU 绝对值与软组织区分，大幅度 scale+shift 可能抹掉其边界——幅度从小起步。
- **定量参数图慎用**：DTI 等定量 MRI 的值本身是诊断量，扰动破坏量纲。

## 参数与经验值

- BigAug 的 scale+shift 具体幅度范围本次未逐项核实，以论文/官方实现为准。
- 实施次序：**HU 域扰动 → 训练窗宽窗位归一化**（次序不可倒）。
- 起步策略：小幅扰动起步，验证集盯高对比结构（骨/钙化）是否退化，再逐步放大。

## 实现参考

- MONAI：`RandScaleIntensity` / `RandShiftIntensity`（即 scale+shift；注意挂到归一化**之前**的管线位置）。
- BigAug 完整组合与幅度经验见 [bigaug-stacked-perturbation.md](bigaug-stacked-perturbation.md)；论文细节见 PMC7393676 页。

## 相关卡

- [bigaug-stacked-perturbation.md](bigaug-stacked-perturbation.md)——本卡变换所属的组合
- [mri-bias-field-augmentation.md](mri-bias-field-augmentation.md)——MRI 侧的强度扰动（乘性低频场，机理不同；CT 无偏置场）
- [nnunet-default-augmentation.md](nnunet-default-augmentation.md)——通用默认强度扰动参照

## 实验反链

暂无。
