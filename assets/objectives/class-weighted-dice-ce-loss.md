# class-weighted-dice-ce-loss（类加权 Dice + 交叉熵复合损失）

**类别标签**：`[loss]`（可导损失）

## 论文支撑（统一四问）

- **哪篇论文用了它**：Guo、Gong、Jiang、Yang、Zheng 等（华中科技大学同济医学院附属协和医院等，Thy-uNET），《Development and validation of AI-based automatic segmentation and measurement of thymus on chest CT scans》，BMC Medical Imaging 2025;25:249，https://pmc.ncbi.nlm.nih.gov/articles/PMC12220788/ 。所读版本：PMC 开放获取全文 + 官方补充材料 DOCX（Table S1–S3，2026-10-08 从 Springer static-content 下载解析）。CE 逐类权重机制本身是标准做法（`F.cross_entropy` 的 weight 参数；nnunetv2 的 `DC_and_CE_loss` 即支持），无单独原始论文；Thy-uNET 是小器官分割场景使用它的直接论文证据。
- **目的**：胸腺等小目标在全程胸部 CT 中体素占比极低，未加权时其损失梯度被背景淹没，粗分割网络易整体忽略小类；而两阶段 coarse-to-fine 流水线（见 [../models/coarse-to-fine-2stage.md](../models/coarse-to-fine-2stage.md)）里粗分割的召回就是系统上限。论文原文："To address the issue of class imbalance arising from the small size of the thymus, we adjusted the loss function by increasing the weight assigned to the thymus category"。
- **效果**：该加权支撑粗分割阶段"to better learn the thymic region and capture its approximate location"；整系统（两阶段 nnU-Net）内部测试 Dice 0.85 / IoU 0.75，多中心外部 0.82/0.70（286 例测试）。**注意证据强度：论文未做加权 on/off 消融，也未公开权重数值**（正文、补充材料 Table S1–S3、官方代码仓库均未给出）——"加权有效"是作者机制陈述，权重数值须任务仓库自调。
- **结论**：小占比目标场景的低成本第一杠杆（改损失不动数据流）；适用边界——softmax 单标签多类分割。权重过大会使背景欠拟合、假阳上升：在"输出只当 ROI 定位用、假阳可被下游精分割纠正"的粗分割阶段可放大，精分割阶段慎用。

## 形式

在 [dice-ce-loss.md](dice-ce-loss.md) 家族（L = weight_ce·L_ce + weight_dice·L_dice）之上，对目标类 k 取 w_k > 1（背景 w_0 = 1）：

- 类加权 CE（= `F.cross_entropy(weight=w)` 标准语义，i 遍历体素、含背景类）：
  `L_ce^w = Σ_i w_{y_i}·(-log softmax(logits)_{i,y_i}) / Σ_i w_{y_i}`
- 类加权 Dice（本实现的可加权扩展；nnunetv2 原版 Dice 项无权重。逐类 soft Dice 后取权重归一化平均）：
  `L_dice^w = -Σ_k w_k·DC_k / Σ_k w_k`（k 遍历参与平均的类；batch_dice=True 时各类跨批合并，False 时按 (样本, 类) 元素平均）
- 非归一化口径（`dice_weight_norm=False`）：`L_dice^w = -Σ_k w_k·DC_k / |K|`——二类分割下加权才生效，但改变损失整体尺度、与 lr 耦合。

参考实现：同目录 `class-weighted-dice-ce-loss.py`（自测：朴素双精度对拍、方向性、退化一致性、有限差分梯度等 10 项）。

## 适用

- softmax 单标签语义分割（多类），2D/2.5D/3D；类别极不平衡场景：小器官、小病灶的**定位/粗分割阶段**。
- 与两阶段流水线配套是它的最佳生态位：粗分割偏召回、假阳由 ROI 裁剪与精分割纠正。
- 兄弟参照：nnunetv2 `DC_and_CE_loss` 的 CE 项即支持 weight（`/ssd/lx/totalsegmentator/nnunetv2/training/loss/compound_losses.py:8`）。

## 行为特征

- 目标类梯度占比上升 → 网络对小类更敏感、召回上升；伴随该类假阳可能上升（背景被相对降权）。
- 观察配合：分看 CE/Dice 分量（见 dice-ce-loss.md 的诊断口径）+ **目标类召回率**（粗分割阶段的关键指标）；最终以逐案验证 Dice 为准。
- 归一化口径（默认）只改类间相对比重、不改变损失尺度，与 lr 解耦，跨实验可比性好。

## 调参经验

- 权重无文献锚点（Thy-uNET 未公开）：目标类 **2–10 起步网格搜**（如 {2, 5, 10}），背景恒 1。
- 默认 CE、Dice 两项同时加权（论文未说明作用项）；要与 nnunetv2 严格对齐时关 `apply_to_dice`（其原版 Dice 项无权重）。
- **单一前景类的二类分割**：归一化加权 Dice 是恒等操作（比值上下同乘抵消），加权只能靠 CE 项；需要 Dice 项生效用 `dice_weight_norm=False` 并固定口径对比。
- nnU-Net 方法论优先数据侧（前景过采样）；加权与过采样**不叠加过头**——先用过采样跑通基线，再在粗分割阶段上加权。

## 坑

- 权重放大 = 背景相对降权：背景欠拟合时全图假阳上升；精分割阶段（ROI 内目标占比已高）通常不该再加权。
- 权重改变 CE 的有效梯度尺度：与 lr 存在耦合（归一化 Dice 口径无此问题），调权重后 lr 不必动，但跨实验比较须固定权重口径。
- 缺席前景类仍参与加权平均（同 dice-ce-loss.md 的坑）；GT 全空时该类 DC 由 smooth 主导。
- `ignore_index` 仅作用于 CE；target 含 ignore 值时必须 `weight_dice=0`（one_hot 不接受负标签，家族同口径）。
- 损失可为负（Dice 项是 -DC，家族口径无 "+1" 常数），完美预测 → 0.5·0+0.5·(-1) = -0.5，勿当异常。

## 实验反链

- 暂无（首个预期使用任务仓库为 thymic_seg，见其 `docs/literature/2026-10-08-thymus-seg-survey.md` 第 4 节选型建议）。
