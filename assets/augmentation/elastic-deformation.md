# 弹性形变（elastic-deformation）

类别：`[augmentation]`

## 论文支撑（统一四问）

**支撑论文 1：Simard, Steinkraus & Platt，ICDAR 2003（方法起源）**

- **哪篇论文用了它**：Simard, P.Y., Steinkraus, D., Platt, J.C.，ICDAR 2003，对象是 MNIST 手写数字。
- **目的**：模拟手写笔画的弹性变化，让分类器获得对小幅度非刚性形变的不变性。
- **效果**：确立"随机位移场 + 高斯平滑 + 弹性系数"的标准构造，成为后续一切弹性增强的原型（本次调研未核实到单独量化数字）。
- **结论**：平滑随机场 + 幅度系数即弹性形变，构造简单、通用。

**支撑论文 2：Ronneberger, Fischer & Brox，U-Net，MICCAI 2015（医疗引用源头）**

- **哪篇论文用了它**：Ronneberger 等，MICCAI 2015，arXiv:1505.04597：<https://arxiv.org/abs/1505.04597>。
- **目的**：在仅约 30 张显微图像上训练分割网络，靠 "excessive use of data augmentation"（以弹性形变为主）补足数据量。
- **效果**：大比分赢得 ISBI Cell Tracking 2015 竞赛。
- **结论**：极小数据下弹性形变曾是收益最大的增强之一；论文摘要页未给出 α/σ 具体取值。

**支撑论文 3：Rueckert 等，IEEE TMI 1999（3D 形式基础）**

- **哪篇论文用了它**：Rueckert, D. 等，IEEE Transactions on Medical Imaging 1999;18(8):712-721（FFD 自由形变，原为 MR 配准）。
- **目的**：用 B 样条控制点网格参数化非刚性形变。
- **效果**：为 3D 弹性/B 样条形变增强提供数学形式（控制点网格 + 样条插值）。
- **结论**：3D 弹性增强普遍沿用 FFD 式构造。

**支撑论文 4：nnU-Net（弃用证据）**

- **哪篇论文用了它**：Isensee 等，Nature Methods 2021 / arXiv:1904.08128；代码一手核自 PyPI nnunet==1.7.0（`data_augmentation_moreDA.py`、`nnUNetTrainerV2.py`）与 nnunetv2 2.6.4。
- **目的**：检验弹性形变在其通用默认增强组合中的必要性。
- **效果**：v1.7.0 起默认关闭（`do_elastic=False`，备用参数 α(0,900)、σ(9,13)）；trainer 源码 docstring 明言 "we don't do elastic deformation anymore"；v2 彻底移除弹性参数。
- **结论**：最强通用流水线的实证方向是**弃用**；论文正文与代码默认值存在"时间差"，引用以代码为准（本目录 README 纠错名单点名此条）。

## 机制与动机

对图像与标签施加**同一个**平滑非线性位移场：生成随机位移场 → 高斯平滑（σ 控制形变的空间频率）→ 按弹性系数 α 缩放幅度 → 重采样。模拟的真实变异源是软组织非刚性形变：器官蠕动、充盈程度变化、摆位/姿态差异、术中移位。医疗场景的高光时刻是 U-Net 在约 30 张图上的极端小数据——此后它长期被当作医疗增强标配，直到 nnU-Net 在大批量通用基准上默认关闭。弹性形变增强的收益高度依赖任务中"非刚性形变变异"是否真实存在。

## 适用（什么时候用）

- **可能有收益**：目标本身非刚性且形态变异大（肠道、胃、心脏、术中组织）；极小数据集（U-Net 式约 30 例量级）需要极限增强补给变异。
- **慎用/默认不用**：多数解剖目标（脑内结构、椎体、腺体）在常规采集间并无大幅非刚性形变，弹性带来的形变分布是"编造的变异"——nnU-Net 在通用基准上默认关闭即是表态。
- **dummy 2D 场景**：nnU-Net 仅在 dummy 2D 分支保留收窄后的 α(0,200)。

## 预期收益

- U-Net：约 30 张图 + 以弹性为主的极限增强，大比分赢 ISBI Cell Tracking 2015（流水级证据，无单独消融数字）。
- 反向证据：nnU-Net v1.7.0 默认禁用、v2 彻底移除——通用分割基准上不再视为默认有益。
- 两者之间无系统的医学消融数字（本次调研未核实到）。

## 风险与坑

- **拓扑破坏**：幅度过大会让细小结构错位、血管/导管"断开"、标签拓扑改变；幅度应控制在目标结构自身尺度以下。
- **标注必须同场形变**：图像形变而标签不动（或反之）直接制造错位噪声；标签插值按惯例用最近邻、图像用灰度插值，通道间用同一位移场。
- **引用陷阱（本目录 README 点名）**：nnU-Net 论文正文长期保留弹性形变叙述，代码 v1.7.0 起默认禁用、v2 彻底移除——写实验记录或论文时以代码核实值为准。
- **α/σ 无权威默认**：U-Net 未给取值；nnU-Net 的备用值是关掉之前的旧配置，不代表推荐值。

## 参数与经验值

- **nnU-Net v1 备用参数**（默认 `do_elastic=False`）：α ∈ (0, 900)、σ ∈ (9, 13)。
- **dummy 2D 分支**：α 收窄为 (0, 200)。
- **经验原则**（定性）：α 以不超过目标结构自身尺度为界；σ 控制形变平滑尺度——过小接近噪声、过大退化为整体平移。
- **MONAI**：有 RandElasticDeform 等现成变换可用。

## 实现参考

- nnU-Net v1：nnunet==1.7.0 `data_augmentation_moreDA.py` 的 `do_elastic` 开关与参数（默认关闭）。
- nnU-Net v2：弹性参数已彻底移除，勿在 v2 配置中寻找此开关。
- MONAI：RandElasticDeform 等弹性变换。
- FFD/B 样条构造的原始出处：Rueckert 1999（IEEE TMI 18(8):712-721）。

## 相关卡

- [nnunet-default-augmentation.md](nnunet-default-augmentation.md) — 弃用它的组合级默认
- [anisotropy-aware-augmentation.md](anisotropy-aware-augmentation.md) — dummy 2D 下 α 收窄规则
- [misalignment-augmentation.md](misalignment-augmentation.md) — 图像-标签错位类的另一族形变模拟

## 实验反链

暂无。
