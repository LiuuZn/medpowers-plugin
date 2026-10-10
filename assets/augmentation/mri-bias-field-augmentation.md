# MRI 偏置场增强（mri-bias-field-augmentation）

类别：`[augmentation]`

## 论文支撑（统一四问）

**物理建模出处**（现象与机制的论文源头，非增强出处）：Van Leemput, Maes, Vandermeulen, Suetens，"Automated model-based bias field correction of MR images of the brain"，IEEE TMI 1999;18(10)：<https://pubmed.ncbi.nlm.nih.gov/10628948/>——用 EM + 图谱统计模型为脑 MR 的偏置场建立了标准物理建模，后续把偏置场当增强用的工作都在这个物理图像上做文章。

**AdvBias**（作为增强使用的第一条线）：

- **哪篇论文用了它**：Chen C, Qin C, Qiu H, Ouyang C, Wang S, Chen L, Tarroni G, Bai W, Rueckert D（Imperial College London，Rueckert/Bai 团队），AdvBias（Realistic Adversarial Data Augmentation），MICCAI 2020：<https://arxiv.org/abs/2006.13322>。
- **目的**：学习生成"逼真而非最大化"的偏置场扰动，作为即插即用增强，提升心脏 MR 分割的跨域泛化与低数据量表现。
- **效果**：心脏 MR 上泛化与低数据表现均提升；摘要层无具体 Dice 数字，照实注明。
- **结论**：偏置场扰动的"逼真性"比"最大化对抗"更重要——偏离真实采集变异的强扰动反而无效；方法即插即用。期刊扩展为 AdvChain（Medical Image Analysis 2022），见 [adversarial-augmentation.md](adversarial-augmentation.md)。

**SynthSeg**（作为增强使用的第二条线，域随机化成分）：

- **哪篇论文用了它**：Billot 等，SynthSeg，Medical Image Analysis 2023;86:102789：<https://pmc.ncbi.nlm.nih.gov/articles/PMC10154424>（PMC10154424）。
- **目的**：在"全随机合成图像"的域随机化训练里，用随机偏置场调制模拟表面线圈灵敏度不均，使单个网络适配任意对比度的 MRI。
- **效果**：消融实验中去掉偏置场增强后 Dice 下降 3.7（9 个未见域口径）——是组合中贡献量化明确的成分。
- **结论**：偏置场增强是 MRI 跨域泛化的关键成分之一；生成方式为小尺寸 3D 高斯场上采样后取指数、乘到图像上。

## 机制与动机

MRI 多线圈采集用表面线圈接收信号，线圈灵敏度随空间变化，图像因此叠加一层**平滑的低频乘性亮度场**——同一组织在图像不同位置呈现不同强度。偏置场增强在训练时随机生成这样的平滑场乘到输入上，逼网络学会对低频亮度漂移不敏感；它模拟的真实变异源是线圈灵敏度伪影，以及 N4 等校正后仍然残留的残余场。

## 适用（什么时候用）

- **模态**：仅 MRI。偏置场来自表面线圈灵敏度，CT 无此现象——CT 上加它是无物理来源的纯噪声。
- **任务**：MRI 分割/分类皆可；2D/3D 均可，3D 上生成连续场更自然（SynthSeg 即 3D 小场上采样）。
- **数据量级**：几十例小数据尤其实用——零样本成本、不需要目标域数据、即插即用。
- **已做过 N4 校正也要加**：校正后仍有残余场，训练期扰动可继续覆盖残余，并非"校正过就不用增强"。

## 预期收益

- SynthSeg 消融（Medical Image Analysis 2023;86:102789）：去掉偏置场增强 **−3.7 Dice**（9 个未见域口径）。
- AdvBias（MICCAI 2020）：心脏 MR 泛化与低数据量场景提升（论文摘要无具体数字）。
- 无其他通用管线的单独消融；小数据上属于"低风险、默认值得开"的变换。

## 风险与坑

- **用在 CT 上无意义**：CT 无偏置场物理来源（见 ct-hu-perturbation.md 的对照）。
- **逼真度是成败关键**：AdvBias 的核心论点——单纯最大化幅度的对抗扰动不逼真、反而偏离真实变异；随机场若频率过高、幅度过大、边界突兀，同理会制造假分布。
- **与强度归一化的次序要在管线里固定**：乘性场改变强度分布，归一化统计口径随之变化，训练与推理必须同序，否则训练增强与推理分布错位。

## 参数与经验值

- **SynthSeg 方案**：小尺寸 3D 高斯随机场 → 上采样到全图分辨率 → 逐体素取指数 → 乘性作用于图像（具体 σ、幅度超参以官方实现为准，本次未逐项核实）。
- **AdvBias 方案**：对抗约束下生成逼真场，"逼真"优先于"大"（具体幅度未核实）。
- 定性三要素：**低频、平滑、乘性**——实现偏离这三条就在模拟别的东西了。

## 实现参考

- MONAI：`RandBiasField`（低频乘性随机场；系数范围需按上述三要素与论文方案对齐后再用）。
- SynthSeg：官方代码与合成流程入口见 PMC 论文页（PMC10154424）的代码/数据链接；完整域随机化组合见 [synthseg-domain-randomization.md](synthseg-domain-randomization.md)。
- AdvBias：按论文 arXiv:2006.13322 复现，即插即用挂在常规增强之后。

## 相关卡

- [synthseg-domain-randomization.md](synthseg-domain-randomization.md)——偏置场是 SynthSeg 域随机化组合的一员
- [adversarial-augmentation.md](adversarial-augmentation.md)——AdvBias 的期刊扩展 AdvChain 记在此卡
- [bigaug-stacked-perturbation.md](bigaug-stacked-perturbation.md)——模态无关的大幅组合扰动
- [ct-hu-perturbation.md](ct-hu-perturbation.md)——CT 侧强度扰动对照（HU 域仿射，非乘性场）
- [nnunet-default-augmentation.md](nnunet-default-augmentation.md)——通用默认增强组合参照

## 实验反链

暂无。
