# 多序列错位增强（misalignment-augmentation）

类别：`[augmentation]`

## 论文支撑（统一四问）

- **哪篇论文用了它**：Kovács B, Netzer N, Baumgartner M, Schütz V 等（DKFZ），"Addressing image misalignments in multi-parametric prostate MRI for enhanced computer-aided diagnosis of prostate cancer"，Scientific Reports 2023：<https://pubmed.ncbi.nlm.nih.gov/37957250/>；官方代码 <https://github.com/MIC-DKFZ/misalignment_DA>（nnU-Net 插件，DKFZ 官方）。
- **目的**：多参数前列腺 MRI（mpMRI）计算机辅助诊断中，真实采集常存在**序列间错位**（患者运动、软组织形变），训练期主动施加错位使模型容忍。
- **效果**：**具体错位幅度与 AUROC 数字未核实**（原文全文页本次不可达，照实注明）。
- **结论**：序列间错位应作为训练期增强处理，而不是只依赖采集端配准；实现以 nnU-Net 插件形式官方开源。

**后续对照**（配准路线 vs 增强路线）：

- **哪篇论文用了它**：Hering 等，arXiv:2404.09666（2024）：<https://arxiv.org/abs/2404.09666>。
- **目的**：比较"配准校正错位"与"训练期容忍错位（增强）"两条路线。
- **效果**：配准路线提升病灶重叠 **+10% median Dice**，但 AUROC 仅 **+0.3%**（p=0.18，不显著）。
- **结论**：在诊断终点（AUROC）上，**增强路线已难被配准超越**——配准对分割重叠有帮助，却带不来显著诊断收益。

## 机制与动机

多序列联合输入（如 T2 + DWI + ADC）在真实采集里并非像素级对齐：患者运动与软组织形变让各序列之间偏移几毫米量级。训练时随机挑选一个（或部分）序列，**相对**其余序列施加随机平移/形变，其余通道与标注保持不动——让网络学会在存在错位的前提下仍融合多序列信息，而不是默认输入永远完美对齐。

注意这与几何增强的本质区别：普通几何增强对**所有通道施加同一变换**，通道间相对关系不变，模拟不了"序列之间彼此错开"。

## 适用（什么时候用）

- **输入形态**：多序列/多模态联合输入——mpMRI（T2+DWI+ADC）、PET/CT 等多通道融合任务。
- **单模态无意义**：只有一个通道时不存在"相对错位"，此增强无从谈起。
- **数据形态**：采集端未做严格配准、或配准质量存疑的临床数据——这正是真实数据的常态。
- **任务**：分割与诊断（分类）皆适用（论文场景即 CAD 诊断；Hering 对照也是诊断终点）。

## 预期收益

- Kovács 2023（Scientific Reports）的量化数字**未核实**（原文全文页不可达）。
- 间接证据（Hering 2024，已核实）：配准路线病灶重叠 +10% median Dice，但诊断终点 AUROC +0.3%（p=0.18 不显著）——容忍路线（增强）已是诊断终点上难被超越的基线。

## 风险与坑

- **错位幅度必须与真实运动量级一致**（几毫米级）：过大则网络学会**忽略该模态**——错位增强过强等于教模型"这个通道不可靠"，反而摧毁多序列融合。
- **必须通道级独立变换**：用普通几何增强错当错位增强（全通道同变换）时不起任何作用。
- **参考序列要固定**：标注挂在哪个序列（通常 T2）须在数据管线里定死，错位只施加在非参考通道上，否则标签与图错位成真噪声。

## 参数与经验值

- 幅度：与真实序列间运动量级（**几毫米**）一致起步；具体数值 Kovács 原文本次未核实。
- 平移与低幅形变（随机位移场）都可施加（论文机制口径）；DKFZ 官方 nnU-Net 插件 `misalignment_DA` 可直接抄配置。

## 实现参考

- 官方：<https://github.com/MIC-DKFZ/misalignment_DA>（nnU-Net 插件，DKFZ 官方）。
- 自实现要点（指路，不写代码）：在 dataloader 内对选中通道单独施加随机平移/低幅形变，其余通道与标签不动；与 [nnunet-default-augmentation.md](nnunet-default-augmentation.md) 的常规几何增强区分挂载。

## 相关卡

- [modality-dropout.md](modality-dropout.md)——多模态输入的另一类鲁棒性增强（丢整模态 vs 错开通道）
- [bigaug-stacked-perturbation.md](bigaug-stacked-perturbation.md)——大幅组合扰动（含全通道几何变换，注意与本卡的区别）
- [mri-bias-field-augmentation.md](mri-bias-field-augmentation.md)——mpMRI 常与之配合的 MRI 强度增强

## 实验反链

暂无。
