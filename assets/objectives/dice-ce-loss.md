# dice-ce-loss（Dice + 交叉熵复合损失）

**类别标签**：`[loss]`（可导损失）

## 论文支撑（统一四问）

- **哪篇论文用了它**：Isensee, Jaeger, Kohl, Petersen & Maier-Hein（德国癌症研究中心 DKFZ），《nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation》，Nature Methods 2021，https://www.nature.com/articles/s41592-020-01008-z （Dice+CE 是 nnU-Net 的默认训练损失）。所读版本：Nature 官方文章页（正文付费墙，读摘要与数据集清单）+ 官方免费补充材料 MOESM1（Supplementary Notes 1–9，44 页 PDF，损失论述在 Note 2"Training Scheme"）+ arXiv:1809.10486（同框架早期论文，Dice 公式的式 (1)(2) 出处）；代码口径对照本地 nnunetv2 源码（`/ssd/lx/totalsegmentator/nnunetv2/training/loss/compound_losses.py:8` 的 `DC_and_CE_loss`，实例化于 `nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py:399`）。
- **目的**：生物医学数据类别不平衡严重，稀有前景类在训练中欠代表、易被忽略（补充材料原文）；Dice 损失直接对齐评价指标并能缓解不平衡，但 patch 式训练下只是近似地优化该指标，单独使用训练不稳定。CE 提供稠密逐像素梯度稳定全局收敛，Dice 项把优化对齐到 Dice 指标——原文："Empirically, combining the Dice loss with a cross-entropy loss improved training stability and segmentation accuracy. Therefore, the two loss terms are simply averaged."（两损失项简单取平均）。
- **效果**：作为 nnU-Net 默认配置，nnU-Net 不做人工干预即在 23 个国际分割比赛的公开数据集上超过大多数现有方法（含高度特化方案）（摘要原文）。例（补充材料 §3.1 的 ACDC 应用示例，五折 CV 前景平均 Dice；原文："the 2D U-Net obtains a score of 0.9165, the 3D full resolution a score of 0.9181 and the ensemble of the two a score of 0.9228"；完整五折表在 §6.6 Table SN6.26）：2D U-Net 0.9165、3D 全分辨率 0.9181、集成 0.9228。早期 arXiv 版在 Medical Segmentation Decathlon 七个 phase-1 任务测试集上几乎全部类别取得最高 mean Dice（Heart 92.77、Liver 95.24/73.71、Hippocampus 90.37/88.95 等）。注意：论文没有对损失项本身做消融（补充材料 SN8 的消融只覆盖激活函数/上采样/深监督），"组合提升稳定性与精度"是经验结论。
- **结论**：Dice+CE 简单平均是 nnU-Net 的默认训练损失，随 nnU-Net 成为医学分割的事实标准；适用边界——softmax 多类单标签分割（互斥类别）；区域式任务（一个体素可同属多个结构，如 KiTS 的肿瘤+肾）nnU-Net v2 改用 BCE+Dice 变体（代码 `DC_and_BCE_loss`）。

## 形式

设 logits 过 softmax 得 u（B,C,⋯），真值 one-hot 为 v，i 遍历批内所有体素，K 为前景类集合（不含背景类）：

- soft Dice loss（arXiv:1809.10486 式 (2)，多类平均的 batch Dice；Nature 版正文付费墙未含显式公式，口径由官方补充材料与开源代码确定）：
  `L_dice = -(1/|K|) · Σ_{k∈K} [ 2·Σ_i u_i^k·v_i^k ] / [ Σ_i u_i^k + Σ_i v_i^k ]`
  其中 u = softmax 输出、v = one-hot 真值、i 遍历批内体素；K 为类别集合——arXiv 原文对 K 仅写 "k∈K being the classes"，**未明示排除背景**；不含背景（|K| = C-1）是 nnU-Net 代码约定（`do_bg=False`，nnUNetTrainer.py:400、dice.py:48/96），本实现默认遵循该约定（见 `dice-ce-loss.py` 的 `exclude_background`）。
- 交叉熵：`L_ce = -(1/N) · Σ_i log softmax(logits)_i[y_i]`（逐像素，含背景类）
- 复合（Nature 版补充材料原文"the two loss terms are simply averaged"）：
  `L = 0.5·L_dice + 0.5·L_ce`

参考实现：同目录 `dice-ce-loss.py`。

## 适用

- softmax 单标签语义分割（二类/多类），2D/2.5D/3D 均可；patch 式训练。
- 类别不平衡场景：小器官、小病灶（前景体素占比极低）——Dice 项使小前景类的梯度不随其体素数消失。
- 兄弟项目在用：totalsegmentator 内嵌的 nnunetv2 默认损失即 `DC_and_CE_loss`（`compound_losses.py:8`，`nnUNetTrainer.py:399` 实例化，`weight_ce=1, weight_dice=1, smooth=1e-5, do_bg=False`）。

## 行为特征

- 训练早期 CE 主导：逐像素稠密梯度（背景也参与），收敛平稳，避免纯 Dice 类损失早期的震荡甚至崩塌；训练后期 Dice 项把优化往与指标同向的重叠度上拉，缓解"CE 降了但 Dice 不动"的背离。
- 对小目标相对敏感（机制）：Dice 的分子分母同数量级地小，小前景类的 Dice 项梯度权重不衰减；这正是它抗类别不平衡的原因。
- 论文训练配方（补充材料 Note 2）：SGD、初始 lr 0.01 的 poly 衰减、nesterov 动量 0.99、1000 epoch×250 iteration，配合前景过采样（补不平衡，但"不宜过度"，否则网络见不到数据的全变异性）与深监督（除最低两个分辨率外的解码器输出都加损失）。
- 观察配合：训练监控应分看两个分量——CE 分量与 Dice 分量的下降速度差是诊断信号（见下）；最终以逐案验证 Dice 指标为准。

## 调参经验

- 权重起点 0.5/0.5（论文"简单平均"）。nnU-Net v2 代码用 `weight_ce=1, weight_dice=1`——与平均只差常数因子 2，最优点相同、只改损失尺度（从而与 lr 的耦合），比较实验时固定口径即可；权重不必和为 1。
- batch Dice vs 逐样本 Dice：论文口径——对"近全患者输入"的 3D U-Net 逐样本计 Dice 再对批平均，其余（2D/patch 训练）把整批当伪体素合并计 Dice（nnU-Net 默认 batch_dice=True）。批小或前景占比波动大时可试逐样本。
- 极端不平衡先动数据侧（前景过采样）再动损失；nnU-Net 方法论本身把复杂度放在架构配置与预处理上，不鼓励损失工程——先用好默认，再考虑 focal/Tversky 等替换（见 focal-loss.md、tversky-loss.md）。
- 深监督时多个分辨率输出按 1/2^i 递减加权求同一损失（nnunetv2 实现；论文只规定"除最低两个分辨率外都加辅助损失"）。

## 坑

- **GT 中缺席的前景类**：K 取前景类时，某批/样本中不出现的类仍参与 Dice 平均，其分母含预测概率质量 Σu——若网络在该处有输出会白白拉高损失；nnU-Net v2 不屏蔽缺席类（代码行为）。数据集里根本不存在的类应从类别表剔除，而非指望损失处理。
- **损失可为负（易误判为 bug）**：式 (2) 是 `-DC`，**没有 "1-DC" 的 +1 常数**——完美分割时 Dice 项取负极值 -1，复合损失 → 0.5·0+0.5·(-1) = -0.5。nnU-Net 训练日志里损失为负是正常现象（nnunetv2 同样返回 `-dc.mean()`）；也因此损失幅度不能跨损失函数比较。
- **损失 Dice ≠ 指标 Dice**：损失里是 soft 概率 + batch 口径 + 过采样后的分布，验证指标是逐案二值化 Dice。训练损失下降但验证 Dice 停滞，先核对口径再下结论（典型背离信号）。
- **batch_dice=True 的稀释**：单个样本前景完全丢失时，其误差被批内其他样本摊薄，小 batch 下可能掩盖失败样本。
- **CE 分量占比**：类别极度不平衡时 CE 的绝对值远大于 Dice 分量（背景体素多），平均权重 0.5/0.5 不等于梯度贡献对半；若 Dice 分量几乎不动，可增大 weight_dice 或改用过采样。
- 空前景边界情形（GT 全空且预测也空）：平滑项使该类 DC→1，按式 (2) 该项损失取负极值 -1（即按"完美"计）；GT 空而预测非空则 DC→小值、该项损失趋 0（按"最差"计）。数值安全，但"全空=完美"的语义要与指标定义对齐。

## 实验反链

- 暂无（`experiments/` 目前为空）。
