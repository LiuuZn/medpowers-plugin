# assets/CODE_STYLE.md — 资产代码统一规范

本规范约束 `assets/` 下所有**含代码的条目**（`<类别>/<slug>.py`）。资产卡（`.md`）的字段格式见各目录 README（models / objectives / preprocessing）；本文件只管代码。目标是让每份参考实现**自包含、可自测、可复核**：任何人拿到仓库后，用一条命令即可在 CPU 上验证该条目实现的数学正确性，不需要 GPU、不需要下载数据。

自测统一解释器（本机）：`/ssd/lx/totalsegmentator/.venv/bin/python assets/<类别>/<slug>.py`。

## 1. 语言与依赖

- **Python ≥3.11 语法基线**（当前自测解释器：`/ssd/lx/totalsegmentator/.venv`，Python 3.11.14；代码不得使用高于该版本的语法特性；类型标注直接用 PEP 604 写法 `int | None`、内置泛型 `list[int]`）。
- **依赖仅 numpy、torch，必要时 scipy**（如 `scipy.ndimage` 的 EDT / 形态学操作）。不引入任何其他第三方运行依赖；标准库不限。
- **自测必须 CPU 可跑、几秒内完成**：
  - 禁止要求 GPU / CUDA，禁止下载数据集或联网；
  - 测试张量用小尺寸（2D ≤ 64×64、3D ≤ 32³、通道 ≤ 8；网络测试配置缩小通道数，如 `base_filters=8` / `channels=(4,8,16)`）；
  - 唯一例外：`try: import 第三方库 ... except ImportError: print("SKIP ...")` 式的**可选**对拍（如与 torchvision 官方实现对拍）——缺席时必须跳过且不影响正确性结论；条目正确性的主证据是文件内的朴素参考实现对拍（见 §7）。

## 2. 自包含

- 每个文件**不 import 本仓库其他实现文件**，也不依赖 `assets/` 之外的仓库代码；条目间需要互引时，在注释或资产卡里写路径（如 `../objectives/dice-loss.md`），不做代码级依赖。
- 一个文件一个条目：公共 API 都围绕同一论文/方法；文件内的小工具函数（`_` 前缀）允许私有复用。

## 3. 文件头 docstring（固定顺序）

文件第一个 docstring 依次写四块，全部**简体中文**：

1. **条目名与类别标签**：首行格式 `<slug> — <一句话中文名> [<标签>]`。标签按目录：`assets/models/` 用 `[model]`；`assets/objectives/` 用 `[loss]`（可导损失）/ `[objective]`（目标函数）/ `[metric]`（不可导评价指标）；`assets/preprocessing/` 用 `[preprocessing]`。
2. **论文支撑统一四问**（口径同资产卡，每项一两句话，写简单，附论文链接并注明所读版本）：
   - 哪篇论文用了它（作者/团队 + 简称 + 年份 + 链接）；
   - 目的：论文用它解决什么问题；
   - 效果：取得什么提升（有数字写数字）；
   - 结论：论文下的结论 / 适用边界。
3. **实现说明**：
   - 关键公式与论文式子编号的对应（论文无编号公式时按节/页注明出处）；
   - 参考了哪些开源实现（仓库名 + 文件路径；本地落点写绝对路径）；
   - 与参考实现不一致处，逐条编号并写明**以论文为准**的理由（或注明是本实现的有意扩展）。
4. **布局与数值稳定**：张量/数组布局约定（见 §5）；eps / 钳制清单及取值依据（见 §6）。可在文件头集中说明，函数 docstring 引用之。

## 4. 命名与 `__all__`

- 模块名、函数、变量：`snake_case`；类：`PascalCase`；仅文件内部使用的工具一律加 `_` 前缀。
- 公开 API（供外部使用的函数与类）全部写入 `__all__`，置于 import 之后、首个定义之前。
- slug（文件名）用方法/模型的通用名：全小写、连字符分隔（如 `dice-loss`、`attention-unet`、`foreground-crop`）。

## 5. 类型标注与 docstring、布局

- **类型标注**：公开函数与类方法的参数、返回值全部标注。torch 张量标 `torch.Tensor`，numpy 数组标 `np.ndarray`，可选用 `X | None`，序列参数写明元素类型（如 `list[list[int]]`）。
- 文件顶部加 `from __future__ import annotations`，使标注惰性求值——延迟 import 的名字（如在函数内才 `import torch` 的文件里标 `torch.Tensor`）也能安全书写。
- 公开函数/类一律**简体中文 docstring**，必须写明：
  - **张量/数组布局**与各参数含义。布局记法统一：
    - 模型/损失：2D = `(B, C, H, W)`，3D = `(B, C, D, H, W)`（B 批、C 通道；logits 注明未过 softmax）；真值/权重图相应为 `(B, H, W)` / `(B, D, H, W)`；
    - 预处理/评价：numpy 数组，写明布局（如 3D 体数据 `(D, H, W)`、2D `(H, W)`、多通道加前导通道轴 `(C, D, H, W)`；spacing 等元数据写明与轴的对应顺序）；
  - 返回值的形状与取值范围（如 ∈ [0,1]、可为负的原因）；
  - 数值稳定相关参数（eps、smooth 等）的含义、默认值与依据。

## 6. 数值稳定

零除、log、exp、幂运算处**必须**加 eps 或钳制，注释写明**取值依据**（相对正常输入量级有多小、防哪种退化情形）：

- **分母保护**：`denom.clamp_min(1e-12)` 或 `(std + eps)`；eps 取值要远小于正常分母量级（常用：1e-5 训练平滑、1e-7～1e-8 防零除、1e-12 兜底），在常量定义处或文件头集中注释；
- **log / exp**：优先 `F.log_softmax`、`F.binary_cross_entropy_with_logits` 等 log-sum-exp 内部稳定实现，不写裸 `log(p)` / `exp(x)`；确需手写时先减最大值并注明；
- **幂运算**：底数先钳到定义域（如 `(1 - p_t).clamp(min=0.0).pow(gamma)`），注明对合规输入是无操作、防的是越界标签的负底数幂；
- **退化输入**：全零 / 常数 / 极值 logits / 全空分割等路径必须显式处理（返回有限值或抛 ValueError），不静默 NaN；
- **自测必须断言**输出与梯度无 NaN/Inf（`torch.isfinite(...).all()` / `np.isfinite(...).all()`），覆盖上述退化路径。

## 7. 文件末自测块

每个文件末尾 `if __name__ == "__main__":`，**直接运行本文件即完成全部检查**（`python <slug>.py`，CPU、几秒）：

- **固定随机种子**（`torch.manual_seed(...)`；numpy 用 `np.random.seed(...)` 或 `np.random.default_rng(...)`）。
- 按条目类型的覆盖要求：
  - **损失/目标函数**：forward + backward，断言损失与梯度有限；与**朴素双精度参考实现**（严格按论文公式、不含任何稳定项，独立于主实现的稳定化路径）逐元素对拍；论文给了梯度闭式的一并对拍（float64）。
  - **模型**：forward + backward，断言输出/梯度有限，**打印参数量**。
  - **不可导指标**：典型 + 边界用例（含空分割等退化情形），断言形状/取值范围/有限性；可与暴力 O(n²) 实现对拍验证。
  - **预处理**：跑典型用例，断言输出**形状 / 数值范围**（及必要的保真性、退化行为）。
- **逐项 `print("PASS <检查点>")`**：检查点用中文短句说明验证了什么；文件**最后一行**打印 `PASS 全部自测通过`，便于 `python <slug>.py | grep -c PASS` 式的机器检查。

## 8. 文件与资产卡同名同目录

- 一条资产两个文件、slug 一致：`assets/<类别>/<slug>.py` 配 `assets/<类别>/<slug>.md`。
- 两者互链：`.md` 的"参考实现"字段指向同目录 `.py`；`.py` 文件头注明资产卡路径（如 `资产卡：assets/objectives/dice-loss.md`，或首行直接相邻写明）。
- 新增条目前先读对应目录 README 的收录边界与字段要求；代码本身的规范以本文件为准。

## 自测块骨架（示意）

```python
if __name__ == "__main__":
    torch.manual_seed(0)          # 固定随机种子
    np.random.seed(0)

    # 1) 与朴素双精度论文公式对拍（2D/3D 布局）……
    got = my_loss(logits64, target64)
    ref = _naive_reference(logits64.numpy(), target64.numpy())
    assert np.abs(got.numpy() - ref).max() < 1e-10
    print("PASS 与论文公式对拍（2D/3D）")

    # 2) forward + backward：损失与梯度有限 ……
    loss = my_loss(logits32, target32); loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits32.grad).all()
    print("PASS forward+backward 梯度有限")

    # N) 模型 forward+backward + 参数量 ……
    print(f"PASS 小模型端到端 参数量={n_params} loss={loss.item():.4f}")

    print("PASS 全部自测通过")
```
