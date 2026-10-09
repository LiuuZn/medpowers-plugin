# uloss —— 不确定性/难度加权分割损失（PGU + REH）

## 论文支撑（统一四问）

- **哪篇论文用了它**：Yanyan Zheng, Yabo Wu, Jie Chen, Xiaoguo Yang, Hao Zhang, Quanyong Yi, Jiantao Pu, Lei Wang（温州医科大学附属眼视光医院 / 伦敦国王学院 / 匹兹堡大学等），《Uncertainty- and hardness-weighted loss functions for medical image segmentation》，Engineering Applications of Artificial Intelligence 163(Pt.3):113118，2026（2025-11-10 在线），https://doi.org/10.1016/j.engappai.2025.113118 ，PMC 全文 https://pmc.ncbi.nlm.nih.gov/articles/PMC12691699/ 。本卡依据 PMC 全文（PMC12691699，NIHMS2122123）通读整理（2026-10-09 抓取全文 XML，公式逐条按 MathML 核对）。注意：**"uLoss" 一词不见于论文正文**，论文自命名 CeW1/CeW2/DiceW1/DiceW2（加权 CE/Dice），代码仓库名 uhLoss；PGU/REH 是论文对两种权重的命名。
- **目的**：CE/Dice 等常用损失对所有像素等权，不感知像素级预测不确定性（uncertainty）与难度（hardness），分割误差集中在目标边界。论文设计两种逐像素权重嵌入现有损失：PGU（概率引导不确定性——低不确定像素保底权重、高不确定像素按 |y−p| 区分并相对降权）与 REH（区域增强难度——直接放大 FP/FN 区域像素权重），不改网络与管线即提升边界精度。
- **效果**（平均 DIS，即 Dice 相似度；Swin-Unet 2D / V-Net 3D，四个数据集）：REFUGE 视盘/视杯 CeW1 平均 0.9223、最优组合 CeW2+DiceW2 视盘 0.9733（Dice loss 平均 0.9154，CE 0.9196）；RETA 视网膜血管 CeW1 0.8639 vs CE 0.8337 / Focal 0.8044 / Dice 0.8476 / CE+Dice 0.8564（ASD 1.42 vs CE 1.77、HD95 10.77 vs 14.46）；OCT 角膜层 CeW1 0.8974 vs CE 0.8693，且在 U-Net++（0.9055）/UNeXt（0.8957）/U-KAN（0.9019）上一致优于各自 CE/Dice 基线（配对 t 检验 p<0.05 为显著，REFUGE 上最优组合对 CeW1/CeW2/DiceW1/hDice 达 p<0.0002）；3D 左心房（ASC，V-Net）CeW1+DiceW1 0.9133±0.0174 vs CE 0.8926±0.0247 / CE+Dice 0.9109。
- **结论**：PGU 配 CE（CeW1）、REH 配 Dice（DiceW）各自最有效，二者机制正交可组合互补；跨 2D/3D、四数据集、四架构有效。作者自述局限：μ/η 最优值随任务变化大、调参困难；充分优化后部分加权损失仅与原版相当；权重只用了像素级概率差，未用符号距离场等其他差异信息（列为未来工作）。

## 类别标签

`[loss]`（可导损失；权重是 p 的函数、论文未写 detach，实现按公式字面让梯度流过权重，自动微分）

## 形式

概览框架（论文式(1)，加权平均）：L = Σf(y,p)·w / Σw。但实际两个损失用的是**不同的权重嵌入方式**（论文原话：针对对数与分数运算特性分别设计），如下。

记 y_ic∈{0,1} 为真值、p_ic∈[0,1] 为预测概率（i 像素、c 类），d = **|y_ic − p_ic|**（绝对概率差，原文为竖线围栏）。

PGU 权重（式(3)；τ 小阈值，实验取 0.005）：

```
w_ic = μ            , d ≤ τ   （低不确定：保底权重 μ）
w_ic = μ·d + p_ic   , d > τ   （高不确定：按差区分，μ>1 时 w 可低于 μ——相对降权）
```

REH 权重（式(4)；η 放大系数）：

```
w_ic = 1 + η·fp + fn = 1 + η·p_ic + y_ic − (η+1)·p_ic·y_ic
其中 fp = (1−y_ic)·p_ic，fn = (1−p_ic)·y_ic（像素级软计数）
```

（原文排版漏了 fp/fn 的括号，由式(4)右端恒等式唯一确定此读法；**η 只乘 fp**，不是 η·(fp+fn)。）w 取值 [1, η+2]。

加权 CE（式(5)，纯加权和、**无归一化**）与加权 Dice（式(6)，权重进分子与分母的 p 侧、y 侧不加权）：

```
CeW = −Σ_c Σ_i y_ic · log(p_ic) · w_ic
DiceW = 1 − (1/C) Σ_c [ 2·Σ_i y·p·w ] / [ Σ_i y + Σ_i p·w ]
```

合成 uLoss：`CeW_w + DiceW_w` 简单相加（Table 9 组合族 CeW1+DiceW1 / CeW2+DiceW2 等，无组合系数）。

## 适用

- 2D/3D 医学分割（论文场景：眼底视盘视杯、视网膜血管、OCT 角膜层、3D 左心房 MRI），误差集中在边界、FP/FN 成片的难目标。
- 需要跨架构稳定增益的正则项：论文验证 Swin-Unet / U-Net++ / UNeXt / U-KAN / V-Net 一致有效；对现有 CE/Dice 训练管线是纯损失级替换，不动数据与网络。
- 已有 CE+Dice 基线但边界指标（ASD/HD95）卡平台时，优先试 CeW1（PGU 配 CE）与 DiceW（REH 配 Dice）。

## 行为特征

- **PGU 与 focal 方向相反**：focal 放大难样本，PGU 在 μ>1 时把中等不确定像素的权重降到 μ 以下（w=μd+p<μ），"减少难像素对其他像素的影响"，同时用大 μ 放大高不确定像素之间的区分度；论文消融显示 PGU 对 CE 增益最大。
- **REH 直接放大 FP/FN**：易分类像素 w≈1，FP/FN 区域像素最高 η+2；对 CE 的增益略逊 PGU（论文解释：REH 给 FP/FN 区域内像素均匀加权、不区分不确定性）。η 只作用于 FP 侧，FN 侧恒为 1 倍。
- 与真实评价指标联动：DIS 即式(7) Dice，训练时应同步盯 DIS/ASD/HD95；论文各任务上加权版 ASD/HD95 系统性下降（边界误差减少）。
- **CeW 量级随图像大小与 μ 线性增长**（无归一化的加权和）：论文场景下 2D 32×32 输入自测单样本 CeW ~O(10²~10³)，与 ∈[0,1] 的 DiceW 相加后实际由 SGD 学习率吸收（论文 SGD lr=0.01、100 epoch 正常收敛）；换自适应优化器或改批量归约口径时注意量级。
- **DiceW 可为负**：式(6)分母只加权 p 侧，w≠1 时完美预测的 PGU-DiceW = 1−2μ/(1+μ)（μ=4 时 −0.6）；这是公式固有行为而非 bug，REH 在完美预测时为 0。

## 调参经验

- **τ 不敏感**：论文扫描 {1e-4, 1e-3, 5e-3, 1e-2, 1e-1}，前四档各任务差异可忽略，0.1 开始劣化；直接用 0.005。
- **μ（PGU）任务相关、常需较大值**：论文扫描 1–20；最优 CeW1 为 REFUGE 4 / RETA 16 / OCT 14 / ASC 4，DiceW1 为 1/4/7/4。论文解释：FP 区域的概率差 d 远小于 p，μ 太小则高不确定像素区分不开。建议从 4 起，边界误差不降则网格上探至 14–16。
- **η（REH）论文常用 2–3**：扫描 1–12；最优 CeW2 为 2/1/2/5、DiceW2 为 2/1/3/8（REFUGE/RETA/OCT/ASC）。血管等细结构偏小值（1–2），3D 左心房偏大值（5–8）。
- μ/η 无通用默认：论文归因于各数据集强度分布、目标形状尺寸、架构与损失组合的差异；实用起点 μ=4、η=2、τ=0.005，按任务小网格（μ∈{4,8,16}、η∈{1,2,4,8}）搜索。
- 组合方式：CeW1+DiceW1 与 CeW2+DiceW2 在四数据集都稳健；论文 Table 9 显示交叉组合（CeW1+DiceW2 等）也可，但无一致最优。
- 论文训练配置参考：SGD lr 0.01（poly 衰减）、momentum 0.9、weight decay 1e-4、100 epoch，2D batch 32 / 3D batch 8，单卡 RTX 3090——本仓 2×RTX 4090（23GB）预算宽裕；AMP 下软计数统计（Σy·p·w 等）保持 fp32（同 tversky-loss 卡建议）。

## 坑

- **很新且无官方代码**：2025-11-10 在线；官方仓库 https://github.com/wmuLei/uhLoss 截至 2026-10-09 只有一句 README（"PyTorch implementation will be available once the paper is accepted"，实际已接收仍未放码，单分支单 commit、无 tag/fork），独立复现极少。本仓实现是论文公式的直接转写，官方代码发布后应回来对拍复核（detach 口径、批归约细节尤甚）。
- **命名混乱**：检索时用 PGU / REH / uhLoss / CeW / DiceW；"uLoss" 是任务侧简称。
- **公式易读错三处**：d 是绝对值 |y−p|；fp/fn 原文漏括号（应为 (1−y)p、(1−p)y）；η 只乘 fp。
- **μ=0、η=0 不退化为普通 CE+Dice**：μ=0 时 w=p·𝟙(d>τ)、η=0 时 w=1+fn；要无权重基线请直接用 CE+Dice，不要用这组参数。
- CeW 无归一化的量级问题（见"行为特征"）；DiceW 的负值行为（同上）。
- C==1（sigmoid 单通道）入口的 CeW 按式(5)字面只有前景项 −y·log p，与论文 softmax 双通道（含背景的全 CE）数值口径不同；多类/二类建议一律用 C≥2 的 one-hot 入口。
- 多类任务某类整图缺失时，式(6)字面把该类比值计 0（拖低平均、相当于惩罚"缺类"）；论文是逐目标二分割场景无此问题，真正多类任务需自行决定空类处理。
- 权重是否 detach 论文未写明；本实现按字面让梯度流过权重，若复现对不上可试 w.detach()（权重当常数）。
- "跨架构一致有效（p<0.05）"是配对 t 检验、以 DIS 为指标的口径；ASD/HD95 并非都显著。

## 实验（EXP-*）反链

暂无（experiments/ 目前为空；后续实验启用本损失时在此处互加反链）。

## 参考实现

`assets/objectives/uloss.py`（同目录）：式(3)~(6)逐条对应，含 pgu_weight/reh_weight/cew_loss/dicew_loss/uloss/ULoss、朴素双精度参考实现与八项自测（forward+backward 有限性、极端 logits、退化路径、REH 难样本放大方向、PGU 降权方向、2D/3D 端到端），`python uloss.py` 全部通过。对拍结论：与论文公式的**独立**双精度 numpy 实现（/tmp/check_uloss_vs_paper.py，42 个形状×超参组合）最大绝对差 9.1e-13、最大相对差 2.6e-16（float64 机器精度级）；与官方代码对拍不可行（仓库无代码，见"坑"首条）。与论文/官方的不一致处共 6 条（log_softmax 替代 log(p)、Dice 分母 eps 兜底、批归约默认 mean、权重不 detach、官方代码缺失待复核、C==1 口径），逐条见 .py 文件头。
