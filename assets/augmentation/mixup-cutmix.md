# 朴素混合样本增强（mixup-cutmix）

类别：`[augmentation]`

## 论文支撑（统一四问）

- **哪篇论文用了它**：Zhang, Cisse, Dauphin, Lopez-Paz，**mixup**，ICLR 2018，[arXiv:1710.09412](https://arxiv.org/abs/1710.09412)。
  **目的**：源头方法——对两张图像及其标签按同一随机系数做线性插值后训练。
  **效果**：原论文分类基准数字本仓库事实包未收录，此处不引。
  **结论**：自然图像分类的通用正则；医疗分割下的表现由下述基准给出——器官级有正证据，小病灶分割是负证据。
- **哪篇论文用了它**：Yun 等，**CutMix**，ICCV 2019 oral，[arXiv:1905.04899](https://arxiv.org/abs/1905.04899)。
  **目的**：把 mixup 的全局线性插值改成区域替换——剪一块贴另一图，标签按替换面积比例混合。
  **效果**：原论文分类基准数字本仓库事实包未收录，此处不引。
  **结论**：同 mixup，医疗分割证据见下。
- **哪篇论文用了它**：Liu, Fan, Schwarz, Maier，**"Cut to the Mix" 基准**，MICCAI 2024，[arXiv:2602.03555](https://arxiv.org/abs/2602.03555)。
  **目的**：在两个器官分割数据集上系统对比 mix 系方法，基线为**无增强** nnU-Net。
  **效果**：平均 Dice 增益 CutMix +4.9、CarveMix +2.0、AnatoMix +1.9；与传统增强叠加仍有增益。
  **结论**：器官级大结构上简单 CutMix 最稳；"直觉上错误的混合图"也奏效。
- **哪篇论文用了它**：同组（Liu 等），**AnatoMix**，BVM 2024，[arXiv:2403.03326](https://arxiv.org/abs/2403.03326)。
  **目的**：把一个病人的器官移植到另一病人 CT 背景，保持解剖正确的掩码。
  **效果**：公共多器官 CT 平均 Dice 76.1 vs 基线增强 74.8。
  **结论**：器官级混合可以做到"解剖正确"，与 CutMix 的"粗暴但有效"形成两极。
- **哪篇论文用了它**：Zhang, Zhang, Xu，**ObjectAug**，IJCNN 2021，[arXiv:2102.00221](https://arxiv.org/abs/2102.00221)。
  **目的**：对象/背景解耦，逐对象增强并用 inpainting 补洞。
  **效果**：无单独消融数字。
  **结论**：对象级解耦增强的早期尝试。
- **对照证据（负）**：**CarveMix** 论文（[lesion-copy-paste.md](lesion-copy-paste.md)）的对照实验：nnU-Net、ATLAS 卒中 T1w 全量数据下 mixup 59.33、CutMix 56.11 均低于传统增强 59.39（CarveMix 63.91）；12.5% 数据 CutMix 仅 24.57；DWI 急性卒中 12.5% 数据 CutMix 7.72——小病灶分割上朴素混合低于传统增强甚至近乎崩坏。

## 机制与动机

mixup 对两张图按同一随机系数逐像素线性插值，标签按同系数插值；CutMix 把一张图的矩形区域替换为另一张图的对应区域，标签按替换面积比例混合。两者的共同卖点是**不需要任何标注结构信息**（不必知道器官/病灶在哪），实现成本几乎为零。代价是标签与图像内容解耦：矩形区域里实际贴进来的内容与"按面积配比"的标签并不对应。器官级大结构下这个错配无害甚至有益（网络学到更平滑的决策边界）；但在小病灶分割里，一个盖住病灶的背景矩形会把真阳性直接污染成错误的软标签。

## 适用（什么时候用）

- **模态**：CT/MRI 均可；医疗证据集中在 CT 器官分割。
- **2D/3D**：2D 原生；3D 体数据矩形混合无独立论文（"CutMix3D"非正式名称，真实对应物见 [lesion-copy-paste.md](lesion-copy-paste.md) 注记）。
- **数据量级**：器官分割各量级均有正证据；小病灶分割不适用。
- **任务特征**：**大器官/大结构分割首选**（CutMix 增益最大且实现最简）；病灶小而稀疏时改用病灶感知复制粘贴。
- **叠加**：与传统增强叠加仍有增益（MICCAI 2024 基准确认），可作廉价加分项。

## 预期收益

- 器官分割（"Cut to the Mix"，MICCAI 2024，基线=无增强 nnU-Net）：CutMix 平均 +4.9 Dice，优于 CarveMix +2.0、AnatoMix +1.9；叠加传统增强仍有增益。
- 多器官（AnatoMix，BVM 2024）：平均 Dice 76.1 vs 基线增强 74.8。
- 小病灶分割为**负收益**：CarveMix 对照中 CutMix 全量数据 56.11 < 传统增强 59.39；12.5% 数据仅 24.57；DWI 急性卒中 7.72。

## 风险与坑

- **小病灶分割禁用/慎用**：矩形区域盖住病灶时标签按面积混合，真阳性被稀释甚至翻转——CarveMix 的对照数字是直接证据。病灶小而稀疏的任务不要因为实现简单就默认加 mixup/CutMix。
- **对照口径要看清**：器官级 +4.9 的基线是**无增强** nnU-Net，不是与默认增强对比；引用数字时勿混淆口径（与传统增强叠加另有增益，是独立结论）。
- **"CutMix3D" 不是独立论文**：只是个别论文对 3D CutMix 的非正式称呼；3D 医疗分割的真实对应物是 CarveMix/LesionMix（见 [lesion-copy-paste.md](lesion-copy-paste.md)）。
- **医疗分类证据缺失**：mixup 用于乳腺/肺结节分类的高质量专门论文未核实到（只有低影响力期刊），引用"医疗 mixup 分类有效"的结论前需自行核实出处质量。

## 参数与经验值

- **插值/区域系数**：mixup 的插值系数与 CutMix 的裁剪区域均随机采样；具体分布参数以原论文为准（本仓库事实包未收录）。
- **CutMix 标签配比** = 替换面积比例（机制内置，无额外参数）。
- **叠加位置**：作为额外增强叠在 nnU-Net 默认增强之上，仅限器官级任务；小病灶任务不加。
- **组合选择**（MICCAI 2024 基准排序）：器官级任务里 CutMix > CarveMix > AnatoMix（平均 Dice 增益口径）。

## 实现参考

- mixup / CutMix 官方代码入口以各自 arXiv 页面链接为准；两者在主流框架里均为数行实现（batch 内配对 + 插值/区域替换 + 标签同步），本目录不配代码。
- 3D 版勿再检索"CutMix3D"，按 [lesion-copy-paste.md](lesion-copy-paste.md) 选 CarveMix/LesionMix。

## 相关卡

- [lesion-copy-paste.md](lesion-copy-paste.md)：标签一致的病灶级替代，小病灶分割的正解。
- [bcp-semi-supervised-copy-paste.md](bcp-semi-supervised-copy-paste.md)：把"贴"用在半监督一致性框架里。
- [cutout-random-erasing.md](cutout-random-erasing.md)：只减不增的区域操作家族成员。
- [nnunet-default-augmentation.md](nnunet-default-augmentation.md)：叠加用的传统增强基座。
- [elastic-deformation.md](elastic-deformation.md) / [anisotropy-aware-augmentation.md](anisotropy-aware-augmentation.md)：可与 mix 系叠加的传统几何增强。

## 实验反链

暂无。
