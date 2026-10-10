# 扩散模型合成增强（diffusion-synthesis-augmentation）

类别：`[augmentation]`

## 论文支撑（统一四问）

### ① brainSPADE（Fernandez, Pinaya, …, Cardoso，KCL，[arXiv:2209.08256](https://arxiv.org/abs/2209.08256)，2022 预印本）

- **哪篇论文用了它**：brainSPADE，King's College London 团队，2022。
- **目的**：潜扩散模型生成标签图 + SPADE 语义图像生成器按标签上色，产出全合成（图像, 标签）对，做极小样本肿瘤分割的数据扩充。
- **效果**：BraTS 真实数据仅 5 例时：混合（真实+合成）0.876 > 仅真实 0.813 > 纯合成 0.760；健康组织远 OOD 域 WM Dice 0.433→0.809。
- **结论**：2D 切片级；"标签扩散 + 语义合成"两段式比端到端图像扩散更省数据；合成数据是真实数据的补充而非替代（混合最优）。

### ② LesionDiffusion（Lei, Tian 等），MICCAI 2025（[papers.miccai.org/2025](https://papers.miccai.org/2025/) 的 0493-Paper0764；[arXiv:2503.00741](https://arxiv.org/abs/2503.00741)）

- **哪篇论文用了它**：LesionDiffusion，MICCAI 2025。
- **目的**：文本可控的 3D CT 病变合成——一次同时生成病变图像与掩码，覆盖 14 类病变/8 器官，为下游分割造数据。
- **效果**：底库 1,505 例 CT；下游 nnU-Net v2 分割 67.7→70.1 DSC（+2.4）；对未见病变类型（颅内出血），预训练后再微调超仅真实训练 15%。
- **结论**：3D 潜扩散能产出带掩码的 3D 病变样本；但训练需 1,500+ 例底库、单卡 24GB 吃紧、采样慢——底库规模是硬门槛。

> **旁注**：Saragiah 等（[arXiv:2310.16794](https://arxiv.org/abs/2310.16794)，息肉分割）报告扩散优于 GAN 且训练算力更低——仅摘要级核实，具体 Dice 数字未核实。

## 机制与动机

扩散模型逐步去噪生成图像，训练稳定性与样本质量普遍优于 GAN。医疗合成的两种有效形态：① **brainSPADE 两段式**——潜扩散学标签图分布产出新标签，SPADE 语义生成器按标签合成图像；把"学生成什么结构"与"学生成什么外观"拆成两段，各自都更省数据（论文结论：比端到端图像扩散省）。② **LesionDiffusion 端到端式**——3D 潜扩散在文本提示控制下同时生成病变图像与掩码，标注免费，并可用文本控制病变类型/位置做定向补缺。与 GAN 相比少了模式坍塌之忧；与 SynthSeg 域随机化相比，扩散学的是"逼真分布"，域内表型更真实，但必须先有真实数据训生成器。

## 适用（什么时候用）

- **模态**：MRI（BraTS）与 CT（LesionDiffusion）均有验证。
- **2D/3D**：brainSPADE 为 2D 切片级；LesionDiffusion 是 3D 但要 1,500+ 例底库。**几十~几百例的本仓库典型量级：只考虑 2D 两段式（brainSPADE 形态）**。
- **数据量级**：5 例真实数据即能从混合合成中获益（brainSPADE）；训 3D 生成器则需 1,500+ 例底库（LesionDiffusion）。
- **任务特征**：病灶/稀有表型稀缺、真实样本极小、有配套标签图时；跨对比度/跨中心泛化优先 SynthSeg 流派。

## 预期收益

- 极小样本（仅 5 例真实）BraTS：混合 0.876 vs 仅真实 0.813（+6.3 点）；注意纯合成 0.760 反而最低。
- 远 OOD 健康组织：WM Dice 0.433→0.809（brainSPADE）。
- 下游 nnU-Net v2：67.7→70.1 DSC（+2.4，LesionDiffusion，底库 1,505 例）；未见病变类型上预训练+微调超仅真实训练 15%。

## 风险与坑

- **纯合成不如混合**：brainSPADE 的排序（混合 > 仅真实 > 纯合成）直接给出用法——合成补真实，绝不替换真实。
- **底库门槛**：3D 潜扩散训练需 1,500+ 例，单卡 24GB 吃紧且采样慢；小数据团队训 3D 生成器不现实，2D 切片级是本仓库量级的现实选择。
- **2D 切片级无层间一致性**：brainSPADE 逐切片合成，不带 3D 连续性约束（原论文边界即 2D）。
- **算力代价格局**：GenSeg 消融中扩散生成器（BBDM）质量最高但算力代价大（见 GAN 卡）；Saragiah 等报告的"算力更低"指对比 GAN，数字未核实——扩散的代价主要在采样阶段。
- **生成器质量与下游收益不对齐**：评估必须落在下游分割指标上，不能停在合成样本的观感指标。

## 参数与经验值

- brainSPADE：潜扩散产标签 + SPADE 语义生成；BraTS 5 例真实 + 合成混合最佳；2D 切片级。
- LesionDiffusion：3D 潜扩散；文本控制 14 类病变/8 器官；底库 1,505 例 CT；下游以 nnU-Net v2 验证。
- 两文具体网络规模/步数等推荐值未在核实材料中给出——以各自官方实现为准。

## 实现参考

- brainSPADE：arXiv:2209.08256（官方代码仓本次未核实，以论文页为准）。
- LesionDiffusion：MICCAI 2025 论文 0493-Paper0764 / arXiv:2503.00741（官方代码仓本次未核实）。
- 生成数据属**离线扩充**：合成样本入库按 data-mgmt 规范登记（processed 层），再照常训练；与在线增强不冲突、可叠加。

## 相关卡

- [gan-synthesis-augmentation.md](gan-synthesis-augmentation.md)：同题的 GAN 路线与两路线对比结论。
- [synthseg-domain-randomization.md](synthseg-domain-randomization.md)：不训生成器的合成路线——零底库门槛，跨域与极小标注首选。
- [lesion-copy-paste.md](lesion-copy-paste.md)：非学习的病灶级"合成"，零生成器成本。
- [bcp-semi-supervised-copy-paste.md](bcp-semi-supervised-copy-paste.md)：无生成模型的样本级扩充。

## 实验反链

暂无。
