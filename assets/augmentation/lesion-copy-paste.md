# 病灶感知复制粘贴增强（lesion-copy-paste）

类别：`[augmentation]`

## 论文支撑（统一四问）

- **哪篇论文用了它**：Xinru Zhang, Chenghao Liu 等，**CarveMix**，MICCAI 2021，LNCS 12901 pp.196–205，[arXiv:2108.06883](https://arxiv.org/abs/2108.06883)；期刊扩展版发表于 NeuroImage 2023。
  **目的**：3D 脑病灶分割数据稀缺，传统强度/形变增强不改变病灶的数量与空间分布；CarveMix 按病灶位置/几何从一例抠出 ROI（尺寸按概率采样）贴入另一例，标签相应更新（"雕刻混合"）。
  **效果**：nnU-Net 基座、ATLAS 卒中 T1w：全量数据 Dice 63.91，优于传统增强 59.39、mixup 59.33、CutMix 56.11；数据降到 12.5% 时 Dice 54.77，传统增强 41.86、CutMix 仅 24.57；DWI 急性卒中 12.5% 数据 Dice 71.32 vs CutMix 7.72（CutMix 近乎崩坏）。
  **结论**：数据越少收益越大；同时证明朴素 mixup/CutMix 在小数据病灶分割上低于传统增强。适用 3D 病灶分割（脑）。
- **哪篇论文用了它**：Yang, Zhang 等，**TumorCP**，MICCAI 2021，[arXiv:2107.09843](https://arxiv.org/abs/2107.09843)。
  **目的**：自述首个把 Copy-Paste 引入医疗分割的工作；在线随机对象级增强——对肿瘤的主体/位置/形态做变换后贴入图像，缓解肿瘤样本稀缺。
  **效果**：肾肿瘤分割比强基线 +7.12% 肿瘤 Dice、超当时 SOTA +2.32%；标注数据只有 10% 时提升 +21.87%。
  **结论**：有效但边界明确——2D 切片级、目标限于肾/肿瘤类凸形结构。
- **哪篇论文用了它**：Basaran 等，**LesionMix**，MICCAI 2023 DALI Workshop，[arXiv:2308.09026](https://arxiv.org/abs/2308.09026)。
  **目的**：病灶级双向操作——populate（注入新病灶）+ inpainting（抹除既有病灶），同时改变病灶的形状/位置/强度/负荷。
  **效果**：在 4 个脑 MR 病灶数据集 + 1 个肝 CT 病灶数据集上优于多个 mix 系方法（正文数字未逐条核实）。
  **结论**：增/减病灶的病灶级操作比图像级混合更贴合病灶分割需求。
- **哪篇论文用了它**：Lama 等，**LAMA**，J Imaging Inform Med 2024;37(4):1812–1823，[PMC11300415](https://pmc.ncbi.nlm.nih.gov/articles/PMC11300415)。
  **目的**：皮肤镜病灶感知 mixup——按病灶尺寸分组取 patch，贴到他人图像的非病灶区，可混合两幅以上图像。
  **效果**：多病灶测试集 MuLe Dice 0.7923→0.8321（+5%）；ISIC2017 单病灶集不掉点。
  **结论**：按病灶尺寸分组、贴入非病灶区的思路可迁移到多病灶合成。
- **哪篇论文用了它**：Dai 等，**Soft-CP**，[arXiv:2203.10507](https://arxiv.org/abs/2203.10507)，2022 预印本未见正式 venue。
  **目的**：离线软粘贴，保留病灶边缘信息。
  **效果**：KiTS19：10% 数据 +26.5% DSC、全量 +10.2%（摘要口径、未复核）。
  **结论**：作者主张保留边缘过渡的"软"粘贴是增益来源；预印本引用需谨慎。
- **哪篇论文用了它**：Pezeshk, Petrick, Chen, Sahiner（FDA），无缝病灶注入，IEEE TMI 36(4):1005–1015, 2017，[PMC5509514](https://pmc.ncbi.nlm.nih.gov/articles/PMC5509514)。
  **目的**：用 Poisson 图编辑（保目标梯度 + 最优边界）把 LIDC 真实肺结节无缝插入同一病例的其他肺叶或贴血管位置，扩充 CT 肺结节检测正样本。
  **效果**：正样本增广 6 倍后，一半病人数据的 AUC 追平全量数据；假阳性率大幅下降（Naive Bayes FPR@90% 敏感度 0.715→0.287）；优于 SMOTE。
  **结论**：融合质量决定合成病灶的可信度；检测任务同样受益。
- **哪篇论文用了它**：He, Duan 等，粘贴融合工程，Quant Imaging Med Surg 2023;13(8):5043–5057，[PMC10423356](https://pmc.ncbi.nlm.nih.gov/articles/PMC10423356)。
  **目的**：在 LiTS + 自建 LSM（706 卷）上系统研究"怎么贴"——病灶周围上下文带宽度与融合方式对肝病灶分割的影响。
  **效果**：朴素 copy-paste 已有效（小病灶 Dice 0.717→0.770）；病灶周围 5–7 像素均匀上下文带 + 高斯加权融合最优（小病灶 Dice 0.824，大病灶 0.932）；每病灶贴 2 次最优，过多反而掉点。
  **结论**：贴入物的边缘过渡与上下文量比"贴哪"更重要。

### 源头与历史注记

- 医疗 copy-paste 系的直接源头是自然图像实例粘贴 **Simple Copy-Paste**（Ghiasi 等，CVPR 2021 oral，[arXiv:2012.07177](https://arxiv.org/abs/2012.07177)，COCO 49.1 mask AP）。
- 更早的历史源头：Prastawa, Bullitt, Gerig 在健康脑上以 mass effect + 反应扩散合成带金标准肿瘤（Medical Image Analysis 2009;13(2):297–311，[PMC2660387](https://pmc.ncbi.nlm.nih.gov/articles/PMC2660387)）——为**评测**设计而非训练增强，作历史注记。

## 机制与动机

病灶分割的核心矛盾是正样本（病灶体素）极度稀疏：几十例的小数据集里病灶可能只占极小比例的体素，传统强度/形变增强不改变病灶数量、尺寸与空间分布，网络见到的真阳性模式有限。病灶感知复制粘贴把别的病例的**真实病灶（连同其标注）**搬进当前病例：一次粘贴同时造出一个新的真阳性区域和一份随之更新的标签。与朴素 CutMix 的本质区别是贴入物带结构化标注、标签与图像始终一致，不会把矩形区域内的背景误标成病灶。

## 适用（什么时候用）

- **模态**：CT/MRI 均有证据（肝 CT、脑 MR、肾 CT、肺 CT 结节、皮肤镜）。
- **2D/3D**：3D 体数据首选 CarveMix/LesionMix 系（体数据内雕刻/注入）；2D 切片级可用 TumorCP 类。
- **数据量级**：几十~几百例收益最大——CarveMix 明确显示数据越少收益越大。
- **任务特征**：病灶稀疏、正样本少、类不平衡；肺结节检测同样适用（无缝注入系）。
- **不适用**：大器官整器官分割（器官级应优先朴素 CutMix，见 [mixup-cutmix.md](mixup-cutmix.md)）；无病灶标注的纯背景病例提供不了贴入物。

## 预期收益

- 小数据病灶分割收益最大：CarveMix 在 12.5% 数据上把 ATLAS 卒中 Dice 从 41.86（传统增强）提到 54.77；DWI 急性卒中 12.5% 数据 71.32（CutMix 仅 7.72）。
- 肾肿瘤（TumorCP）：+7.12% 肿瘤 Dice；10% 标注时 +21.87%。
- KiTS19（Soft-CP，摘要口径未复核）：10% 数据 +26.5% DSC。
- 皮肤镜多病灶（LAMA）：MuLe Dice 0.7923→0.8321。
- 肺结节检测（无缝注入）：6 倍正样本增广，一半数据 AUC 追平全量。
- 肝病灶（融合工程）：朴素粘贴小病灶 Dice 0.717→0.770；加 5–7 像素上下文带 + 高斯融合到 0.824。

## 风险与坑

- **硬粘贴的边缘伪影**：QIMS 融合工程显示朴素粘贴虽有效，但边缘过渡带决定上限——加 5–7 像素上下文带 + 高斯加权融合后小病灶 Dice 再 +0.054；不处理过渡带的实现吃不到这部分收益。
- **贴入位置要解剖合理**：病灶贴到解剖学不可能的位置会被网络当噪声；Pezeshk 系把结节贴到同例其他肺叶/贴血管位置是"合理位置"的范例。
- **粘贴次数有上限**：每病灶贴 2 次最优，过多反而掉点（QIMS 系统消融）。
- **2D 粘贴丢 3D 上下文**：TumorCP 自述边界是 2D 切片级、凸形目标；3D 体数据分割优先 CarveMix/LesionMix。
- **在线 vs 离线**：Soft-CP/LAMA 类离线合成占磁盘且组合空间固定；CarveMix/TumorCP 在线随机、每 epoch 组合不同。
- **"CutMix3D" 不是独立论文**：只是个别论文对 3D CutMix 的非正式称呼；3D 医疗分割的真实对应物就是本卡的 CarveMix/LesionMix，检索时勿被误导。

## 参数与经验值

- **ROI 尺寸**：按概率采样（CarveMix），非固定尺寸，保留病灶尺度多样性。
- **上下文带**：病灶周围 5–7 像素均匀上下文带 + 高斯加权融合（QIMS，LiTS+LSM 706 卷消融最优）。
- **粘贴次数**：每病灶 2 次（QIMS 消融最优，更多反而掉点）。
- **操作层级**：populate（注入）与 inpainting（抹除）双向并用（LesionMix），同时扰动形状/位置/强度/负荷。
- **在线（CarveMix/TumorCP）优先于离线（Soft-CP）**，除非需要可复现的合成数据集。

## 实现参考

- CarveMix / TumorCP / LesionMix / Soft-CP 的代码入口以各自 arXiv/出版页给出的链接为准；本目录卡片不配代码。
- 融合配方（上下文带 + 高斯加权）按 QIMS 论文方法节描述实现，参数见上栏。
- 自行实现时注意 MONAI 等框架的 transform 约定：粘贴必须**同步变换 image 与 label** 两个通道。

## 相关卡

- [mixup-cutmix.md](mixup-cutmix.md)：朴素混合的对照与器官级正证据。
- [bcp-semi-supervised-copy-paste.md](bcp-semi-supervised-copy-paste.md)：半监督场景的复制粘贴。
- [cutout-random-erasing.md](cutout-random-erasing.md)：同属区域操作家族但只减不增，分割慎用。
- [nnunet-default-augmentation.md](nnunet-default-augmentation.md)：叠加其上的传统增强基座。
- [foreground-oversampling.md](foreground-oversampling.md)：过采样解决"patch 里有没有病灶"，复制粘贴解决"病灶模式够不够多"。
- [gan-synthesis-augmentation.md](gan-synthesis-augmentation.md) / [diffusion-synthesis-augmentation.md](diffusion-synthesis-augmentation.md)：生成式合成，与真实病灶粘贴互补。

## 实验反链

暂无。
