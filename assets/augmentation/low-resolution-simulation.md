# 低分辨率模拟（low-resolution-simulation）

类别：`[augmentation]`

## 论文支撑（统一四问）

**支撑论文 1：nnU-Net（Isensee 等，Nature Methods 2021 / arXiv:1904.08128）**

- **哪篇论文用了它**：Isensee, F., Jaeger, P.F., Kohl, S.A.A. 等（DKFZ），nnU-Net，2021：<https://www.nature.com/articles/s41592-020-01008-z>；参数一手核自 PyPI nnunet==1.7.0 的 `data_augmentation_moreDA.py` 与 nnunetv2 2.6.4 源码。
- **目的**：让模型对测试/部署时更粗的采集分辨率保持鲁棒——训练数据常是重建精细的研究序列，临床推理常遇厚层扫描。
- **效果**：作为默认成员进入整条流水（p=0.25，zoom U(0.5,1)）；无单独消融（Nature 2021 增强单项消融在付费墙补充材料内未能核实）。
- **结论**：小概率、温和幅度（最低降到一半再还原）即可作为通用鲁棒项；2D/各向异性数据仅做面内。

**支撑论文 2：SynthSeg（Billot 等，Medical Image Analysis 2023）**

- **哪篇论文用了它**：Billot, B. 等，Medical Image Analysis 2023;86:102789（PMC 全文：<https://pmc.ncbi.nlm.nih.gov/articles/PMC10154424>）。
- **目的**：把"任意分辨率/层厚"系统化为域随机化的一个维度——训练时随机模拟各种层间距与层厚，使模型零重训适配临床厚层扫描。
- **效果**：1mm→7mm 层间距仅掉 3.8 Dice（对照 SAMSEG 掉 7.6）；5mm 轴向 FLAIR 0.78 vs 0.64；3mm CT 0.76（零重训）。
- **结论**：显式模拟层间距 + 层厚（高斯核模拟切片敏感度）远优于不模拟；测试域含厚层扫描时这是决定性手段。

## 机制与动机

模拟分辨率退化的两个来源：① **采样稀疏**——按更粗网格降采样（丢高频、产生混叠），再插值还原回原网格，网络在原分辨率的坐标上学"看起来是厚层数据"的输入；② **层厚模糊**——真实厚层的信号是切片敏感度曲线内的积分，近似为高斯核模糊，SynthSeg 先模糊、再降采样。nnU-Net 的实现是"最近邻降采样 1–2 倍再插值还原"的轻量版；SynthSeg 是随机选轴向/冠/矢状位、层间距 r_spac~U(r_HR, b_res)、层厚 r_thick~U(r_HR, r_spac) 的系统版。变换只作用于图像，分割标签保持在原网格上。

## 适用（什么时候用）

- **必备场景**：训练数据高分辨率/薄层（如 1mm 研究序列），而测试/部署域含临床厚层（常规 MRI 序列、低场 MRI、5mm 层厚 CT）、多中心不同分辨率——不模拟则域差距直接压垮 Dice。
- **收益有限场景**：训练与测试同为各向同性高分辨率的科研数据（nnU-Net 仍保留 p=0.25 作通用鲁棒项，但非决定因素）。
- **2D/各向异性**：只在面内做；出面方向的分辨率问题应交给各向异性预处理与网络规则（见 [anisotropy-aware-augmentation.md](anisotropy-aware-augmentation.md)）。

## 预期收益

- **SynthSeg**：1mm→7mm 层间距 Dice 仅 −3.8（SAMSEG −7.6）；5mm 轴向 FLAIR 0.78 vs 0.64；3mm CT 0.76 零重训。
- **nnU-Net**：组合流水战绩（29/49 SOTA）含此成员；无单独消融数字（如实注明）。

## 风险与坑

- **幅度过强抹掉小目标**：降采样过狠时小病灶/细结构可能整体消失，而标签仍在原网格——监督信号自相矛盾；nnU-Net 把下限定在 zoom 0.5。
- **与缩放增强区分**：缩放（zoom）改变目标在视野中的大小；低分辨率模拟是"降质后还原到原网格"。nnU-Net 中两者共用 zoom 算子，但语义不同，调参时别混为一谈。
- **各向异性数据只做面内**：出面方向本来就是粗分辨率，再叠加"模拟粗化"会造出脱离真实采集的伪影分布。
- **退化分布要贴近真实**：nnU-Net 用最近邻降采样再插值还原；SynthSeg 强调层厚用高斯核模糊近似切片敏感度——简单套一种降质未必对应真实扫描仪的退化方式。

## 参数与经验值

- **nnU-Net**：p=0.25；zoom U(0.5, 1)（≈最近邻降采样 1–2 倍再插值还原）；2D/各向异性仅面内。
- **SynthSeg 系统版**：随机选轴向/冠/矢状位；层间距 r_spac~U(r_HR, b_res)（b_res 为目标域最低分辨率上限）；层厚 r_thick~U(r_HR, r_spac)；先高斯核模糊模拟层厚 → 降采样 → 重采样回 1mm。
- **经验边界**（定性）：目标域最粗层间距是多少，模拟上限 b_res 就设到多少，不必超过。

## 实现参考

- nnU-Net v1：nnunet==1.7.0 `data_augmentation_moreDA.py`（低分辨率模拟变换及其 p/zoom 配置）。
- SynthSeg：论文 PMC 页（<https://pmc.ncbi.nlm.nih.gov/articles/PMC10154424>）及其官方代码仓库。
- 系统化用法（全域随机化框架）见 [synthseg-domain-randomization.md](synthseg-domain-randomization.md)。

## 相关卡

- [nnunet-default-augmentation.md](nnunet-default-augmentation.md) — 所属组合
- [anisotropy-aware-augmentation.md](anisotropy-aware-augmentation.md) — "仅面内"限制的来源
- [synthseg-domain-randomization.md](synthseg-domain-randomization.md) — 把分辨率模拟纳入全域随机化
- [gan-synthesis-augmentation.md](gan-synthesis-augmentation.md) / [diffusion-synthesis-augmentation.md](diffusion-synthesis-augmentation.md) — 合成数据路线的对比参照

## 实验反链

暂无。
