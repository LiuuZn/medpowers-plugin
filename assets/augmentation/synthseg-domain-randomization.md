# 合成数据域随机化训练（synthseg-domain-randomization）

类别：`[augmentation]`

## 论文支撑（统一四问）

- **哪篇论文用了它**：Billot, Greve, Puonti, Thielscher, Van Leemput, Fischl, Dalca, Iglesias，**SynthSeg**，Nature Biomedical Engineering 2023（[DOI 10.1038/s41551-023-01018-w](https://doi.org/10.1038/s41551-023-01018-w)，PubMed 36857946）。
  > **所读版本**：量化数字全部读自其详版 Medical Image Analysis 2023;86:102789（[PMC10154424](https://pmc.ncbi.nlm.nih.gov/articles/PMC10154424/)）。
- **目的**：完全不用真实图像训练分割网络——从随机标签图出发做域随机化合成，一并解决跨对比度/跨中心/跨分辨率泛化与标注稀缺。
- **效果**：5500 例测试扫描、6 种模态、10 种分辨率：域内 Dice 0.88（监督基线 0.91）；9 个未见域全面领先（6 个统计显著）；5mm FLAIR 0.78 vs SAMSEG 0.64、3mm CT 0.76；分辨率 1mm→7mm 仅掉 3.8 Dice（对比法掉 7.6）。
- **结论**：合成分布"越随机越泛"，逼真度是次要目标（消融直接证明，见"预期收益"）；标签图需求低到个位数；范式可迁移——SynthSeg+、FetalSynthSeg（github.com/Medical-Image-Analysis-Laboratory）已复用到别的器官与人群。

## 机制与动机

训练全程不接触真实图像，流水四步：① 从标签图库抽随机标签图；② 施加随机空间形变（模拟个体解剖与摆位差异）；③ **对每个标签独立做高斯混合强度采样，均值与方差全随机**——这是"模拟任意对比度"的核心：MRI 各序列、CT 各窗位之间的差异被归约为"每个结构的强度分布不同"，网络被迫按结构而非按灰度分割；④ 叠随机偏置场（磁场不均）、Gamma 变换（信号非线性）、层厚/分辨率随机化（模拟厚层采集）。这一整套就是"域随机化"：故意把合成域拉得比任何真实域都宽，真实图像只是其中一小角，推理时自然落在训练分布内。与普通增强的本质区别：普通增强在**固定真实样本**上扰动，本方法**连图像本身都是合成的**——标注天然精确且免费。

## 适用（什么时候用）

- **模态**：为任意对比度设计，论文验证覆盖 6 种模态（含 CT/MRI）；跨对比度、跨中心、跨分辨率场景收益最大。
- **2D/3D**：原版 3D（3D U-Net）；管线可裁剪为 2D（未见单独验证）。
- **数据量级**：**核心卖点——约 5–10 张标签图 Dice 即平台化**，单张标签图已得 0.68–0.80；标签图还可以是自动标注工具产的伪标签（主模型 1020 张中 1000 张为 FreeSurfer 自动标注）。
- **任务特征**：跨域泛化、分辨率参差、标注极度稀缺时首选；纯域内精度压倒一切且无任何标签图时不合适。

## 预期收益

- 域内 Dice 0.88，接近监督基线 0.91；9 个未见域全面领先对比法（6 个统计显著）。
- 跨分辨率鲁棒：1mm→7mm 仅掉 3.8 Dice，对比法掉 7.6；厚层数据 5mm FLAIR 0.78 vs SAMSEG 0.64，3mm CT 0.76。
- **消融是"多样性即收益"的直接证据**：把合成分布约束得更真实反而 −1.4~−2.6 Dice；去偏置场 −3.7、去空间形变 −4.3、去病灶粘贴 −3.9——每个随机化组件都是净增益。
- 标注成本：5–10 张标签图即达平台（单张已有 0.68–0.80），与"几百例标注"的监督路线差一个量级。

## 风险与坑

- **必须有标签图（或伪标签）**：方法省的是"图像-标注配对采集"，不是"连标签都不要"；一张标签图都没有时先用自动标注工具产伪标签，再入此管线。
- **标签图库的结构覆盖决定上限**：合成所用的标签图解剖结构应与目标任务类别对齐（论文用 FreeSurfer 自动标注图库）；类别体系错位则学到错的分割定义。
- **域内精度通常略输监督**（0.88 vs 0.91）：泛化不是白拿的——单中心纯域内任务别盲目全合成，可考虑合成+真实混训。
- **训练时长**：300k 步约 7 天（RTX 6000 级单卡）；本仓库 24GB 的 4090 可复现，但属长任务，按 lease 协议登记并后台分离启动。
- **别把它当"免训练数据"神话**：它免的是真实**图像**，不免标签图、不免算力。

## 参数与经验值

- 标签图数量：主模型 1020 张（1000 张 FreeSurfer 自动标注）；**平台门槛约 5–10 张**，单张 0.68–0.80——起步验证一张就够。
- 网络/训练：3D U-Net，300k 步，约 7 天（RTX 6000 级单卡，24GB 可复现）。
- 随机化组件清单（缺一伤 Dice，消融数字见"预期收益"）：逐标签高斯混合强度采样（均值/方差全随机）、随机空间形变、随机偏置场、Gamma 变换、层厚/分辨率随机化、病灶粘贴。
- 前沿延伸：Learn2Synth（Hu, Zeng, …, Balbastre，ICCV 2025，[arXiv:2411.16719](https://arxiv.org/abs/2411.16719)）用超梯度从少量真实标注自动学合成引擎参数，代替手工调随机化分布——脑分割早期验证，本仓库未复现。

## 实现参考

- 官方代码：github.com/BBillot/SynthSeg（含预训练模型与完整训练管线，可先推理再复训）。
- 范式迁移参考：SynthSeg+ / FetalSynthSeg（github.com/Medical-Image-Analysis-Laboratory）。
- MONAI 无开箱全合成管线；自组思路是"标签图库 + RandAffine 类形变 + 逐标签强度采样 + RandBiasField/RandAdjustContrast 类灰度扰动 + 低分辨率模拟"（本目录卡片不配代码，组件单体卡见相关卡）。

## 相关卡

- [elastic-deformation.md](elastic-deformation.md)：随机空间形变组件的单体版。
- [mri-bias-field-augmentation.md](mri-bias-field-augmentation.md)：偏置场组件的单体版。
- [low-resolution-simulation.md](low-resolution-simulation.md)：层厚/分辨率随机化组件的单体版。
- [lesion-copy-paste.md](lesion-copy-paste.md)：病灶粘贴组件（消融 −3.9）的单体版。
- [modality-dropout.md](modality-dropout.md)：多模态缺失场景的另一思路。
- [gan-synthesis-augmentation.md](gan-synthesis-augmentation.md) / [diffusion-synthesis-augmentation.md](diffusion-synthesis-augmentation.md)：训生成器学"逼真分布"的对照路线——SynthSeg 证明逼真度并非必要。
- [nnunet-default-augmentation.md](nnunet-default-augmentation.md)：真实样本在线增强的默认基线。

## 纠错注记

- **"SynthTrainer" 无独立论文**：只是 Billot 团队代码库内部训练模块名（经 GitHub 404、bioRxiv API、多路检索四重复核）；学术引用一律以 SynthSeg 系列论文为准。

## 实验反链

暂无。
