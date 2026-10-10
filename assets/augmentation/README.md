# assets/augmentation — 任务无关的数据增强方法

收录**任务无关**的数据增强（data augmentation）方法：训练时**在线随机施加**的图像/标签变换，含测试时增强（TTA）与合成数据训练（域随机化）等广义增强。与本仓库"预处理"（[../preprocessing](../preprocessing)，训练前对数据**固定执行一次**的变换）区分：增强作用于每个训练步、带随机性、标签需同步变换。

一条 = 一个方法卡（一个 md 文件），命名按方法名（如 `elastic-deformation.md`）。**本目录卡片不配代码**（无 `<slug>.py` 配对，不受 [../CODE_STYLE.md](../CODE_STYLE.md) 约束）：只固化"什么情况下用、有什么收益、有什么风险"；实现按各卡"实现参考"栏指向 MONAI / nnU-Net / 官方仓库。

## 增强卡字段

- **论文支撑（必填）**：按文末统一四问格式，简体中文写明；有多篇支撑论文时逐篇列四问
- **机制与动机**：变换做什么、模拟哪类真实采集变异源（一两句话）
- **适用（什么时候用）**：模态（CT/MRI）、2D/2.5D/3D、数据量级、任务特征（小目标/类不平衡/跨域/多模态）
- **预期收益**：核实过的量化数字 + 出处；无数字的写定性结论并注明"无单独消融"
- **风险与坑**：什么情况下有害、参数越界后果、失效模式
- **参数与经验值**：典型取值与出处（只给数值与配置，不写代码）
- **实现参考**：MONAI transform 名 / nnU-Net 配置位置 / 官方仓库（只指路）
- **相关卡**：同目录互链（相对链接）
- **实验反链**：相关 `EXP-*` 记录

## 收录边界

只写方法本身；某任务用哪套增强组合、为何选它，写进实验记录。组合级策略（如 nnU-Net 默认全套）作为一张卡收录，成员变换不再拆卡重复。

## 论文支撑统一四问（每项一两句话，写简单）

- **哪篇论文用了它**：团队/作者 + 简称 + 年份（附链接）
- **目的**：论文用它来解决什么问题
- **效果**：取得了什么提升（有数字写数字）
- **结论**：论文下的结论 / 适用边界

无论文支撑的方法不作为正式条目，先记入"补录起点"。

## 补录起点（待充实）

- AdvNoise（Chen C 等，MICCAI 2019 workshop，[arXiv:1810.04743](https://arxiv.org/abs/1810.04743)；仅检索确认存在，细节未核实）
- 物理化低剂量 CT 增强（[PMC13366036](https://pmc.ncbi.nlm.nih.gov/articles/PMC13366036)；仅检索确认，数字未核实）
- Tumor Fabrication（[arXiv:2511.18654](https://arxiv.org/abs/2511.18654)，2025 预印本，两阶段生成式病灶合成，暂无量化数字）
- MONAI Auto3DSeg 模态特化增强模板（mri_1 加 gamma 正反向、ct_ax_1 用直方图控制点 + z 轴单轴旋转），可并入相关卡补充

## 纠错与查无实据名单（勿再收录）

- **"CutMix3D" 不是独立论文**：仅为个别论文对 3D CutMix 的非正式称呼；3D 医疗分割的真实对应物是 CarveMix / LesionMix（见 [lesion-copy-paste.md](lesion-copy-paste.md)）
- **"SynthTrainer" 无独立论文**：只是 Billot 团队代码库内部训练模块名，学术引用以 SynthSeg 系列为准（见 [synthseg-domain-randomization.md](synthseg-domain-randomization.md)）
- **"CAR / ACD"**（注意力感知粘贴的具名方法）：多轮检索查无实据
- **BigAug 一作是 Zhang L**（IEEE TMI 2020），常被误归 Dong Yang；AdvNoise/AdvBias 一作是 Chen C（Imperial College，Rueckert/Bai 团队），亦非 Dong Yang
- nnU-Net 论文正文长期保留弹性形变叙述，但**代码 v1.7.0 起默认禁用、v2 彻底移除**——引用以代码为准（见 [elastic-deformation.md](elastic-deformation.md)）
