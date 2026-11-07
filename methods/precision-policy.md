# 方法论：数据格式与计算精度（默认速度优先）

**定位**：任务无关，对所有任务仓库的训练与推理生效，是配置推荐（方法论③）与数据管理（`skills/data-mgmt`）的精度约束正本。一句话：**64 位批量数据零容忍；存储按位深够用；计算按硬件能力与数值敏感点选；默认偏速度。**

## 0. 铁律

1. **禁止 64 位批量数据**：影像、掩码、特征、缓存、DataLoader 张量、GPU 张量一律不得是 `float64` / `int64`。numpy 的默认 dtype **不追认**——`np.array([1,2])` 是 int64、`np.zeros(...)` 是 float64、`np.mean` 返回 float64、scipy/skimage 多数输出 float64；创建时显式 `dtype=`，外部库返回值先 cast 再进训练。
2. **唯一例外：框架强制的索引/标签类型**。PyTorch 的 `CrossEntropyLoss` target、`F.one_hot` 输入、`gather/index_select/scatter_add` 的 index 必须是 int64；`torch.sum`/`count_nonzero` 等归约也可能返回 int64 标量。它们只在 API 边界临时产生，**不得持久化、不得当作影像/掩码/特征的存储 dtype**——掩码落盘一律 uint8，collate 时再 `.long()`。
3. **标量元数据例外**：spacing/origin/direction、JSON 世界坐标、指标标量是 double（JSON number 本身即 IEEE double），量小、无批量内存意义；但不得把坐标/几何转成 float64 **数组**做批量运算。
4. **测试对拍例外**：单元测试里用 float64 做数值参考/梯度对拍是允许的（如 `assets/objectives/*.py` 的自测），但只限测试代码，**不得进入训练、推理或数据处理流程**；测试里的 int64 标签/索引同理。

## 1. 存储格式与 dtype（输入侧）

| 数据 | 默认 dtype | 说明 |
|---|---|---|
| DICOM 原始像素 | 保持源位深：`uint16`（多数）、`int16`/`uint8` | raw 层只读；先读 `BitsStored`/`PixelRepresentation`/`RescaleSlope`/`RescaleIntercept` |
| CT/MR 强度（processed 正本） | `int16` / `uint16`；需要浮点用 `float32` | **NRRD/NIfTI/ITK 没有 half 浮点类型**，磁盘正本不能存 fp16 |
| 训练 cache（`.npy`） | 强度 `float16`，掩码 `uint8` | 显式 dtype；DataLoader 直读；fp16 的主力落点 |
| 分割掩码 | `uint8` | 类别索引（0=背景）；不存 one-hot / float 掩码 |
| 2D PNG | 8-bit（常规）；需保留原始灰度用 16-bit（`I;16`/uint16） | 已归一化到 [0,1] 的图存 8-bit 有量化损失，要无损就 16-bit 定点或 npy fp16/float32 |
| 检测框/平面/关键点 | 世界坐标 JSON（RAS mm） | double 标量可接受；loader 转 float32 张量 |
| 概率图/软标签（若落盘） | `float16` 或 uint8 量化 | fp16 注意小概率下溢与 eps；训练目标仍建议在 fp32 计算 |
| 模型/checkpoint | 权重与优化器状态 `float32` | 训练主权重必须 fp32；推理可另存 bf16/fp16 |

外部输入若以 int64 到达（某些标注器/索引文件、numpy 默认整数），在审计确认取值范围后降到 `uint8`/`uint16`/`int16` 存储；只有框架 API 当场需要的 int64 才临时保留，用完即丢。

**CT/MR 的 fp16 细节（重要）**：

- fp16 的**取值范围**（±65504）覆盖 CT 的 -1024…+4095 HU，也覆盖 MR 归一化后的常见范围——这是"16 位浮点够用"的含义；
- 但 fp16 只有 11 位有效位，**整数精确表示范围是 ±2048**；|x| 在 2048–4096 之间步长 2，4096–8192 之间步长 4。也就是说 HU 存成 fp16 后，骨/钙化等 >2048 HU 的值会丢 1–2 位；
- 因此：**原始 HU 的逐值精确操作（阈值分割、钙化评分、按 HU 测量）必须在 int16 或 float32 上做**；窗宽窗位映射到 [-0.5, 0.5]（或归一化）之后再转 fp16 是安全的；
- 若整体审计发现取值范围超出 ±65504 或大量 NaN/Inf（部分 MR、PET 融合图），先缩放/截断再转 fp16，或保持 float32。

## 2. 转换前的整体判断（强制步骤）

任何"把数据改成 fp16/uint8"的动作之前先跑一遍审计，把结论写进实验记录或数据集说明：

1. **来源与位深**：DICOM 的 `BitsStored`、`PixelRepresentation`、`RescaleSlope/Intercept`（slope≠1 时物理值可能超出 int16）；NRRD/PNG 的 header type。
2. **数值分布**：min/max、NaN/Inf 计数、极端离群值。
3. **整数性**：是否全为整数值；是否存在必须逐值精确的语义（阈值、测量、评分）。
4. **fp16 适配性**：|x|≤2048 精确；2048–65504 步长变大；>65504 溢出；|x| 在 6e-5 以下进入 subnormal、约 6e-8 以下下溢为 0。
5. **用途与收益**：存储/IO/显存各占多少；patch 采样有重复读放大时收益最大；只读一次就丢的中间结果不值得做有损转换。

```python
# 批量数据精度审计（numpy；在 cast 之前跑；标量结果是 Python float，不构成批量 float64）
import numpy as np

def audit(a: np.ndarray, name: str = "array") -> dict:
    a = np.asarray(a)
    is_float = a.dtype.kind == "f"
    rep = {
        "name": name,
        "dtype": str(a.dtype),
        "shape": tuple(a.shape),
        "bytes": int(a.nbytes),
        "min": float(a.min()),
        "max": float(a.max()),
        "nan": int(np.isnan(a).sum()) if is_float else 0,
        "inf": int(np.isinf(a).sum()) if is_float else 0,
        "all_integer": bool(np.all(a == np.floor(a))) if is_float else True,
        "fp16_exact": bool(np.max(np.abs(a)) <= 2048) if a.size else True,
        "fp16_range_ok": bool(np.max(np.abs(a)) <= 65504) if a.size else True,
    }
    return rep
```

**判断规则**：`fp16_range_ok` 为假 → 先缩放/截断；`fp16_exact` 为假且任务依赖逐值精确 → 保持 int16/float32；否则转 fp16。

## 3. 训练计算精度（默认 AMP + bf16）

- **默认**：`torch.autocast(device_type="cuda", dtype=torch.bfloat16)` + fp32 主权重/优化器状态。**bf16 优先**——与 fp32 同指数范围，不需要 GradScaler，溢出风险小。
- **bf16 不可用**（compute capability < 8.0，如 Turing）：fp16 autocast + `GradScaler`（动态 loss scale）；或 fp32 + TF32。
- **TF32**（Ampere/Ada 及以上）：允许用于 fp32 的 matmul/conv；`torch.backends.cuda.matmul.allow_tf32 = True`、`torch.backends.cudnn.allow_tf32 = True`（或 `torch.set_float32_matmul_precision("high")`）。速度收益明显，数值影响通常小于 fp16，但要对照验收指标确认。
- **数值敏感点必须 fp32**：损失函数（Dice/CE 的分子分母、smooth/eps）、指标（Dice/HD95/ASSD）、归一化统计、softmax/sigmoid/log、大幅归约。
- **fp16 归约会溢出**：对 512³ 级体数据直接 `sum()`（fp16）会超过 65504；先 `x.float()` 再归约，或分块累加。`eps`/`smooth` 用 Python float 或 fp32 tensor，不随 autocast 降精度。
- **不要手动把整个模型 `.half()`** 代替 autocast：混合精度由 autocast 按算子选择，手动全 half 会让 softmax/loss 等敏感算子也降精度。
- **FP8 训练**：不是默认选项，仅在框架支持（Transformer Engine / torchao / 专用 kernel）且精度验收通过时使用；主权重与优化器状态仍 fp32。
- **精度变更是一族配置**：与损失/超参/预处理分开做对照轮，避免归因失效（见 [next-config-recommendation.md](next-config-recommendation.md)）。

## 4. 推理计算精度（默认 bf16，极速 FP8 / MXFP8）

推理默认**也偏速度**，但必须"先探测环境、再选格式、最后做精度验收"：

1. **探测**：GPU compute capability、torch/推理引擎版本、bf16/fp16/fp8 算子是否真的可用——`skills/init-env` 的 `probe-precision` 写入 `runtime/env.json` 的 `compute_precision`。
2. **常规推理**：`bf16`（Ampere/Ada/Hopper 首选）；不支持 bf16 用 `fp16`；对数值极敏感的小任务用 `fp32`。
3. **要求推理极快**：`FP8 E4M3`（权重+激活，per-tensor/per-channel scale，静态量化需校准集）；优先走推理引擎（TensorRT / ONNX Runtime / 专用 kernel），PyTorch 路径以 `torch._scaled_mm` 实测可用为准。
4. **极致速度**：`MXFP8`（E8M0 块缩放 + E4M3/E5M2）——需要 **Blackwell（sm_100+）** 与显式支持 MX 格式的运行时；`probe-precision` 的 `mxfp8` 标志只表示"硬件 + torch 具备 E8M0 dtype"，真正启用前必须用目标推理引擎跑通一次 MXFP8 kernel。**Ada（sm_89，含 RTX 4090）不支持 MXFP8**。
5. **精度护栏**：归一化、softmax/sigmoid、最终 logits、argmax 保持 bf16/fp32；只量化 conv/linear 等主体算子。FP8 上线前，在与 bf16/fp32 基线相同的验证集上比对 Dice/HD95/ASSD 与逐例最大偏差，容差预先写进实验记录，并测延迟（warmup 后取中位数）。
6. **RTX 4090 现实**：Ada 有 FP8 tensor core（sm_89），但 PyTorch 对 FP8 的支持面窄、算子覆盖不全；实践中最可靠的 FP8 推理路径是 TensorRT。MXFP8 不可用。

## 5. 默认值速查

| 场景 | 默认 | 备选 | 硬约束 |
|---|---|---|---|
| raw 存储 | 源格式（DICOM uint16/int16） | — | 无损、只读 |
| processed 正本 | int16/uint16（CT/MR）、8/16-bit PNG | float32（NRRD/NIfTI） | 不用 float64；不把 fp16 写进 NRRD/NIfTI |
| cache（`.npy`） | 强度 fp16、掩码 uint8 | int16 | 显式 dtype；DataLoader 直读 |
| 训练 | AMP bf16 | fp16+GradScaler；fp32+TF32 | 主权重/损失/指标 fp32 |
| 推理（常规） | bf16；FP8（可用且验收后） | bf16 / fp16 / fp32 | 与 bf16 基线比指标 |
| 推理（极速） | FP8 E4M3 | MXFP8（Blackwell） | 校准 + 精度验收 + 延迟实测 |

## 6. 记录与验收

- `skills/init-env` 在 `runtime/env.json` 写入 `compute_precision`：探测到的能力（bf16/fp16/TF32/FP8/MXFP8）与推荐默认，是训练/推理选精度的事实源。
- 每条实验记录的「训练配置」必须写明精度（AMP dtype、TF32 开关）；推理/评测实验写明推理精度与引擎。
- 切换精度（尤其 FP8）必须作为独立对照：同一数据、同一 checkpoint、只改精度，记录指标差与延迟；容差未预先定义的结果不算验收通过。
- "因为更快所以选 FP8"不够——速度收益与精度代价都要有程序化数值。

## 7. 常见坑

- **numpy/torch 默认 dtype**：`np.array` 整数→int64、`np.zeros`→float64、`np.mean`→float64；`torch.tensor(np_arr)` 原样继承 dtype。创建即显式 dtype，外部库返回值先 cast 再进训练。
- **SimpleITK**：`GetArrayFromImage` 返回源的 int16/uint16/float32；`GetImageFromArray` **不接受 float16**（ITK 无 half 像素类型），需要先转 float32——这也是"磁盘正本不用 fp16"的直接原因。
- **PIL 16-bit**：模式是 `I;16`/`I`，不是 `L`；`np.array(img)` 的 dtype 要显式确认。
- **fp16 存 HU**：>2048 不精确；阈值/钙化评分在 int16/fp32 上做。
- **fp16 归约**：大体积 `sum` 溢出 65504；先 `float()`。
- **GradScaler 与 bf16 混用**：bf16 不需要 scaler；混用无收益且增加复杂度。
- **FP8 缩放 API 版本差异**：`torch._scaled_mm` 的参数与 B 矩阵布局随版本变化；写死一种签名会在他机失败，必须实测并回退。
- **OOM 归因**：显存不够先查是否某处仍是 float64/int64（一个 float64 张量 = 两个 fp32 的量），再考虑降 batch/换精度。
- **精度与泛化的混淆**：fp16/bf16 训练通常不改变收敛行为；若指标变化明显，先排查 loss/metric 是否在低精度下计算，而不是归因于"精度正则化"。

## 8. 与其他文档

- `skills/data-mgmt`：存储格式与 dtype 的落地规范（本文件给原则，data-mgmt 给目录/格式）。
- `skills/init-env`：`probe-precision` 探测硬件能力并写入 `runtime/env.json`。
- [next-config-recommendation.md](next-config-recommendation.md)：精度变更作为独立配置族做对照。
- `experiments/TEMPLATE.md`：训练配置栏必须写精度。
