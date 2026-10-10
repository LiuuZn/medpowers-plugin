# BigAug 大幅组合扰动（bigaug-stacked-perturbation）

类别：`[augmentation]`

## 论文支撑（统一四问）

- **哪篇论文用了它**：Zhang L, Wang X, Yang D, Roth H, Myronenko D, Xu D, Xu Z, Sanford T, Turkbey B, Wood BJ, Harmon S，"Generalizing Deep Learning for Medical Image Segmentation to Unseen Domains via Deep Stacked Transformations"（通称 **BigAug**），IEEE TMI 2020;39(7):2531–2540：<https://pmc.ncbi.nlm.nih.gov/articles/PMC7393676>（PubMed 32070947）。
  **作者纠错（勿再误引）**：一作是 **Zhang L**，该方法常被误归 Dong Yang——Yang 只是合作者之一。
- **目的**：训练时**大幅度**堆叠多样化变换，把未见域（跨中心/跨设备）的采集变异尽量覆盖进训练分布，使模型在**零目标域数据**下也能跨域泛化。
- **效果**：前列腺 T2、左心房、左心室超声三任务：源域 91.6% → 未见域 **80.0%**（仅降约 11 点）；传统增强（只做随机裁剪）未见域退化到 **39%**；甚至优于用**目标域数据**做域适应的 CycleGAN（**63.5%**）；表现最强的 4 种变换子集 **74.9%**，反而不及 9 种全开。
- **结论**："大幅 + 组合多样性"本身有效——挑子集不如全开；增强零成本，比需要目标域数据与 GAN 训练的域适应路线划算。

**相关系列**（同一思路的对抗加强版）：

- AdvNoise（Chen C 等，MICCAI 2019 workshop，[arXiv:1810.04743](https://arxiv.org/abs/1810.04743)；仅检索确认存在，细节未核实）——噪声类扰动的对抗加强。
- AdvBias（Chen C 等，MICCAI 2020，[arXiv:2006.13322](https://arxiv.org/abs/2006.13322)）——偏置场扰动的对抗加强，见 [mri-bias-field-augmentation.md](mri-bias-field-augmentation.md) 与 [adversarial-augmentation.md](adversarial-augmentation.md)。

## 机制与动机

九种变换在训练时大幅堆叠，各自对应一类真实采集变异源：

- **强度类**：高斯模糊（低分辨率/低剂量重建）、锐化 unsharp masking（锐利重建核）、高斯噪声（电子噪声）、亮度平移、Gamma 对比度、强度 scale+shift（增益/窗位差异，见 [ct-hu-perturbation.md](ct-hu-perturbation.md)）。
- **几何类**：3D 旋转 ±20°、3D 缩放 0.4–1.6、高斯平滑随机位移场形变（体位/FOV/软组织形变，见 [elastic-deformation.md](elastic-deformation.md)）。

单一变换只覆盖变异的一根轴；堆叠起来才能拼出未见域分布——论文的消融（子集 74.9% < 全开 80.0%）直接支持"多样性本身有效"。

## 适用（什么时候用）

- **场景**：跨中心/跨厂商/跨设备部署——最划算的场景；单中心固定设备的常规训练必要性低。
- **模态**：论文覆盖 MR（前列腺 T2）与超声（左心房/左心室）；强度成分的 CT 用法见 [ct-hu-perturbation.md](ct-hu-perturbation.md)。
- **算力**：增强只在训练侧起作用、推理零开销，对 2×24GB 单卡（RTX 4090）友好；域适应路线（CycleGAN 等）额外消耗训练算力还需要目标域数据。
- **小数据**：几十例上 0.4–1.6 的缩放过激，从 nnU-Net 量级起步、视验证曲线放宽（见参数栏）。

## 预期收益

（IEEE TMI 2020，三任务平均，无目标域数据）

- BigAug 九种全开：源域 91.6% → 未见域 **80.0%**
- 最强 4 变换子集：74.9%（**低于**全开——多样性本身有收益）
- CycleGAN 域适应（用到目标域数据）：63.5%
- 传统增强（仅随机裁剪）：39%

## 风险与坑

- **3D 缩放 0.4–1.6 非常激进**：几十例小数据直接照搬可能把训练分布拉得过散——小数据从温和量级起步再放宽。
- **不要做减法**：只挑"表现最好"的变换子集反而更差（74.9% < 80.0%）——组合多样性价值大于单变换强度。
- 大幅增强会拉宽训练分布，收敛与评估都要给足轮次再下结论（经验，无论文数字）。

## 参数与经验值

- 九变换与幅度：高斯模糊、锐化（unsharp masking）、高斯噪声、亮度平移、Gamma 对比度、强度 scale+shift、3D 旋转 **±20°**、3D 缩放 **0.4–1.6**、高斯平滑随机位移场形变（模糊/噪声/Gamma 的具体 σ 与范围本次未逐项核实，以论文与官方实现为准）。
- 子集教训：top-4 子集 74.9% < 全 9 种 80.0%。
- 起步策略：小数据从 nnU-Net 量级起步（参照 [nnunet-default-augmentation.md](nnunet-default-augmentation.md)），验证曲线有提升后再放宽到 BigAug 幅度。

## 实现参考

- BigAug：论文 PMC7393676 页附实现说明（本次未逐项核对超参数值）。
- MONAI：成员变换均有现成对应——`RandGaussianBlur`、`RandGaussianSharpen`、`RandGaussianNoise`、`RandShiftIntensity`、`RandAdjustContrast`（Gamma）、`RandScaleIntensity`、`RandAffine`（旋转/缩放）与随机位移场形变，组合挂载即可。

## 相关卡

- [ct-hu-perturbation.md](ct-hu-perturbation.md)、[mri-bias-field-augmentation.md](mri-bias-field-augmentation.md)——强度成分的模态专化细化
- [elastic-deformation.md](elastic-deformation.md)——几何成分之一
- [adversarial-augmentation.md](adversarial-augmentation.md)、[gan-synthesis-augmentation.md](gan-synthesis-augmentation.md)——对抗/生成式对照路线（AdvBias、CycleGAN）
- [nnunet-default-augmentation.md](nnunet-default-augmentation.md)——温和版默认组合（小数据起步参照）

## 实验反链

暂无。
