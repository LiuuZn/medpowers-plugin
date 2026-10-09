# coarse-to-fine-2stage — 两阶段粗到精 ROI 分割（小器官级联）

> 论文原文所读版本：BMC Medical Imaging 2025;25:249 开放获取全文（PMC12220788，2026-10-08 读）+ 官方补充材料 DOCX（12880_2025_1783_MOESM1_ESM.docx，4.8 MB，2026-10-08 从 Springer static-content 下载并解析出 Table S1–S3 / Fig S1–S11 标题与全文）+ 官方代码仓库 https://github.com/JiangGWei/thymus_analysis（2026-10-08 克隆检查：仅推理/测量骨架，`segment_ct_file` 为空 stub，模型权重未发布、需邮件联系作者）。

## 论文支撑（统一四问）

- **哪篇论文用了它**：Guo、Gong、Jiang、Yang、Zheng 等（华中科技大学同济医学院附属协和医院 + 万东医疗 + 广州医科大学附属第一医院，Thy-uNET），《Development and validation of AI-based automatic segmentation and measurement of thymus on chest CT scans》，BMC Medical Imaging 2025;25:249，https://pmc.ncbi.nlm.nih.gov/articles/PMC12220788/ 。
- **目的**：胸腺在全程胸部 CT 中体素占比极小（小目标 + 类不平衡），单阶段网络难以兼顾全胸上下文与胸腺局部细节；两阶段先全胸粗分割定位，再裁剪 ROI 做高分辨率精分割。
- **效果**：补充材料 Table S1（内部测试集 n=100，Dice / IoU）：3D U-Net 0.83/0.73、UNetR 0.82/0.71、3D TransUNet 0.82/0.72、**单阶段 nnU-Net 0.84/0.74、Thy-uNET 0.85/0.75**（同数据划分与训练设置的对照）。286 例多中心测试总体 Dice 0.83（外部两中心各 0.82），各年龄/性别/层厚/扫描仪亚组一致。注意：相对配置得当的单阶段 nnU-Net，两阶段的净增益是 **+0.01 Dice / +0.01 IoU**，主要优势面是 UNETR/TransUNet 等非自配置架构（+0.03）与跨中心稳健性。
- **结论**：小占比目标上，两阶段 coarse-to-fine 在强单阶段基线（nnU-Net）之上仍有小幅稳定收益，并把"定位"与"精分割"解耦（粗分割输出可直接当 ROI 给下游测量/量化用）；大占比结构无必要。适用边界：粗分割漏检则精分割无法恢复（级联误差传播），目标本身占比不低时收益趋零。

## 核心机制

一句话：第一阶段在全图（全程胸部 CT）上跑一个粗分割模型拿到目标的大致位置，按其输出把图像裁剪成只含 ROI 的小视野，第二阶段在小视野上重新以原（或更高）分辨率精分割——用"视野缩小"换"有效分辨率与目标占比"，让小目标在精分割阶段的体素占比与梯度占比同时上升。

## 作者归因（为何取得 SOTA：作者认为哪些改动起什么效果）

论文未做逐组件消融（无两阶段 on/off、无损失加权 on/off 的对照），以下归因 = 作者机制陈述 + Table S1 同条件对照，按证据强度排序：

1. **骨干本身是最大贡献者**：Table S1 里第二名是"普通单阶段 nnU-Net"（0.84/0.74），比 UNETR/3D TransUNet 高 0.02——自配置（预处理/patch/spacing/过采样全自动化）带来的收益大于两阶段增量。Thy-uNET 的训练配方（SGD momentum 0.99 NAG、初始 lr 0.01 poly 衰减、weight decay 3e-5）与 nnU-Net 默认完全一致，说明骨干基本沿用默认。
2. **两阶段 ROI 聚焦**（作者归因的核心机制陈述）："we cropped the original images to focus solely on the region of interest (ROI) that contains the thymus"——裁剪后"the model is able to accurately identify and segment the thymus, even when it is a relatively small and challenging target"。机制：小视野内目标占比升高，精分割网络不再需要在全图尺度上同时学"在哪"和"长什么样"。
3. **粗分割阶段的胸腺类损失加权**："To address the issue of class imbalance arising from the small size of the thymus, we adjusted the loss function by increasing the weight assigned to the thymus category"，使粗分割模型"to better learn the thymic region and capture its approximate location"。机制：对抗小类梯度被背景淹没，保住两阶段流水线的召回上限（粗分割漏检不可恢复）。**权重数值论文/补充材料/官方代码均未公开**——复现时是待调超参，实现见 [../objectives/class-weighted-dice-ce-loss.md](../objectives/class-weighted-dice-ce-loss.md)。
4. **Dice+CE 复合损失**："using a combination of Dice loss and cross-entropy loss to balance segmentation accuracy and address the class imbalance"——沿用 nnU-Net 默认（详见 [../objectives/dice-ce-loss.md](../objectives/dice-ce-loss.md)），论文未归因增量。
5. **误差面**（作者自述的薄弱处）：扫描仪亚组间有差异（Philips 略优于 GE，Fig S4）；右叶长度/厚度测量一致性弱（R 0.49–0.60），部分归因于测量选层逻辑与专家不同（专家取 AP 径最长层，算法取胸腺面积最大层）。

## 量级

- 两个 nnU-Net 3D 模型：参数量随 plans 自动配置（公开 3D full-res 配置常见几十 M 量级/个），训练与推理成本 ≈ 单阶段的两倍（两遍前向、两套权重、两份预处理数据）。
- 2×4090（每卡 ~23 GB）预算下充裕：nnU-Net 默认 patch/batch 本就按 ~11 GB 级显存设计，照常开 AMP；粗、精两阶段可分卡同时训。

## 适用

- 数据量级：数百例即可获得收益（论文 500 例训练达内部 0.85）。
- 维度：3D 为主（CT 体数据）；2D 也可套用同构流程（逐层/堆层裁剪）。
- 目标尺度：大视野中占比极小的固定解剖结构（胸腺、垂体、肾上腺、小淋巴结区等）；目标占比不小或位置高度不定（如弥漫病灶）时不适用。
- 副产品：粗分割输出天然是下游测量/量化的 ROI 源（Thy-uNET 的 7 项测量即建立在分割结果上）。

## 短板与不适用场景

- **级联误差传播**：粗分割召回是整个系统的上限——精分割不可能找回没被裁进 ROI 的目标。粗分割阶段应偏召回（可放宽阈值/对粗 mask 形态学膨胀后再裁剪）。
- **ROI 边距与尺寸未公开**：论文与补充材料均未给裁剪 margin/固定尺寸，需任务仓库自调并记入实验记录；边距过小截断目标，过大稀释占比收益。
- **两套权重的工程成本**：训练数据流要构造两份（全图版 + ROI 裁剪版）；stage-2 训练集用 GT 裁剪还是 stage-1 输出裁剪，论文未说明（用 stage-1 输出裁剪更贴近推理分布，建议实验对比）。
- **期望管理**：对强基线的净增益小（+0.01 Dice）；若单阶段 nnU-Net 尚未跑通，先跑通它，再上两阶段。
- 与 nnUNet 自带 **cascade**（低分辨率全图 → 原分辨率 patch）机制相近但不同：cascade 是分辨率级通用级联，本方法是目标级 ROI（视野可比 cascade 更小、更贴目标）。选型时值得对比 cascade 配置——它是现成的、无需自建数据流。

## 对比基准与选型建议

- **对比基准**：同数据、同训练配方的**单阶段 nnU-Net**（Dice/IoU 0.84/0.74 对 0.85/0.75，净增益仅 +0.01）与 UNETR/3D TransUNet（0.82/0.71–0.72）；注意 Table S1 第二名就是配置得当的 nnU-Net。
- **更好的条件**：大视野中占比极小的固定解剖结构（胸腺类）；跨中心部署稳健性优先；粗分割输出可复用为 ROI/测量源。
- **值得尝试**：单阶段 nnU-Net（建议直接 ResEnc 预设，`nnunet-resenc.md`）跑通且确认"目标占比/有效分辨率"是瓶颈之后；nnUNet 自带 cascade 配置可作为现成的近似基线先试。
- **不值得**：目标占比不低；粗分割召回没有把握（级联误差传播不可恢复）；作为任务的首个实验——先有强单阶段基线再谈级联。

## 经验备注

- 官方代码现状（2026-10-08 检查）：`thymus_analysis` 仓库是测量/可视化骨架（依赖 nnunetv2、SimpleITK 等），`src/segmentation.py` 的 `segment_ct_file` 为空 stub；`config/config.yaml` 暴露推理口径：tile_step_size 0.5、gaussian、mirroring、fold 0、`checkpoint_best.pth`；README 注明要模型参数需联系 jianggw14@midea.com。**复现只能自训**。
- 复现路径建议：nnUNetv2 两个 dataset（①全图粗分割：胸腺 + 邻近解剖类做上下文；②ROI 精分割：裁剪体积），或直接用 nnUNet cascade 配置作近似基线；粗分割阶段的类加权损失用 [../objectives/class-weighted-dice-ce-loss.py](../objectives/class-weighted-dice-ce-loss.py)（权重 2–10 起步网格搜）。
- 论文训练细节（正文口径）：SGD momentum 0.99 + NAG、初始 lr 0.01 poly 衰减、weight decay 3e-5；epochs/batch/patch/spacing 未公开。
- 数据侧经验：训练集是体检人群（胸腺形态相对规整），外部测试含肺癌患者仍稳定（0.82）；但论文明确排除了前纵隔肿瘤侵犯、胸部术后、邻近肺炎病例——肿瘤/术后场景需另验。

## 实验反链

- 暂无（`experiments/` 无本条目相关记录；首个使用任务仓库为 thymic_seg，见其 `docs/literature/2026-10-08-thymus-seg-survey.md`）。
