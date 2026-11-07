# EXP-<NNN>-<slug>

- **status**: planned
- **model**:
- **planned_at**:
- **started_at**:
- **ended_at**:
- **machine**:
- **gpus**:

<!-- model 为 Studio 任务 code（= 模型标识），`exp-mgmt new --task <task_code>` 写入；
     同一仓库训练多个模型时用它区分。machine/gpus 为可选结构化字段：训练进程经
     medpowers.training_session 在 start 时自动写入（machine = notify.json 的
     machine_name，gpus 如 0,1）；旧记录无这些行属正常，set_field 会自动补插 -->

## 任务与数据【规划时填】

- 任务项目：实际执行训练的仓库与任务（如 `/ssd/lx/BC3-seg`，abdomen 2D 分割）
- 数据：数据集、规模、模态、train/val 划分
- 验收口径：strict 指标的精确定义与容差（如「预测片=标注片」或「±1 片」）、目标值；与数据集 annotation 契约一致

## 配置（本轮决策）【规划时填】

- 基础模型：<主选>（选择理由，对照 methods/base-model-selection.md 的任务画像：____；备选：____）
- 预处理：引用 assets/preprocessing 条目 + 关键参数
- 损失/目标：损失函数及权重/组合；监控的不可导指标（引用 assets/objectives 条目）
- 训练配置：lr / 调度 / batch / epochs / AMP（精度：bf16/fp16、TF32 开关，按 methods/precision-policy.md 与 runtime/env.json 的 compute_precision）；GPU 与 lease slug（start 后补记）
- 推理配置（推理/评测实验填）：精度（bf16/fp8/mxfp8）、推理引擎（PyTorch/TensorRT/ONNX Runtime）、量化校准来源；与基线精度对照的指标差与延迟

## 结果【结束时补】

- 最终与最佳指标（train / val）
- 曲线特征：收敛速度、平台、震荡、train-val 间距、各损失分量行为、指标-loss 是否背离
- 若 interrupted：中断原因、进行到的 epoch、已观察到的部分
- （训练进程经 medpowers 包收尾时会在本节头部自动追加「训练进程自动落账」事实块：时长/指标 JSON/说明——人工与智能体在块后继续填写分析）

## 诊断（对照 methods/training-state-diagnosis.md）【结束时补】

- 状态判断：____（证据：____，置信度：高/中/低）

## 下一轮决策（对照 methods/next-config-recommendation.md）【结束时补】

- 变更清单：改哪族、从什么改成什么、理由（回溯诊断证据）
- 预期信号：若下一轮出现 ____ 则变更有效

## 产物【结束时补】

- checkpoint / 日志 / 曲线图在任务仓库中的路径（训练进程可经 session 句柄 `s.artifacts.append(...)` 随时登记，收尾自动入账）

## Studio 数据与评测来源【按需】

- 任务/模型：Studio task code（与本记录 `model` 字段一致）
- 数据：Studio dataset/task、导出包或数据集版本、approved 病例数、annotation 契约是否完整
- 评测：Studio eval_id / 外部评测报告路径 / report_sha256 / GT 状态（默认 approved）
- 备注：Studio 只做可视化镜像；本记录的正文与诊断仍是权威。不要把 token、webhook 等密钥写进本节。

## 取消理由【仅 cancelled 填】
