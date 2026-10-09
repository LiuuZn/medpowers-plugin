# EXP-YYYYMMDD-<序号>-<slug>

> 复制本模板到任务仓库的记录目录，命名 `EXP-YYYYMMDD-<序号>-<slug>.md`。记录存放位置由任务仓库自选（默认 `<工作区>/experiments/`），INDEX.md 由任务仓库维护；即使放在本仓库路径下也绝不纳入本仓库 git。
> **启动实验时**：建记录，填【启动时填】两节，`status: running`，同步在任务仓库的 INDEX.md 追加一行。
> **实验结束时**：补全其余各节，`status` 改 `done`（正常完成）或 `interrupted`（中断，注明原因），同步更新 INDEX.md 对应行。

- **status**: running

## 任务与数据【启动时填】

- 任务项目：实际执行训练的仓库与任务（如 `/ssd/lx/BC3-seg`，abdomen 2D 分割）
- 数据：数据集、规模、模态、train/val 划分
- 验收口径：strict 指标的精确定义与容差（如「预测片=标注片」或「±1 片」）、目标值；与数据集 annotation 契约一致

## 配置（本轮决策）【启动时填】

- 基础模型：<主选>（选择理由，对照 methods/base-model-selection.md 的任务画像：____；备选：____）
- 预处理：引用 assets/preprocessing 条目 + 关键参数
- 损失/目标：损失函数及权重/组合；监控的不可导指标（引用 assets/objectives 条目）
- 训练配置：lr / 调度 / batch / epochs / AMP；使用的 GPU 与 lease slug

## 结果【结束时补】

- 最终与最佳指标（train / val）
- 曲线特征：收敛速度、平台、震荡、train-val 间距、各损失分量行为、指标-loss 是否背离
- 若 interrupted：中断原因、进行到的 epoch、已观察到的部分

## 诊断（对照 methods/training-state-diagnosis.md）【结束时补】

- 状态判断：____（证据：____，置信度：高/中/低）

## 下一轮决策（对照 methods/next-config-recommendation.md）【结束时补】

- 变更清单：改哪族、从什么改成什么、理由（回溯诊断证据）
- 预期信号：若下一轮出现 ____ 则变更有效

## 产物【结束时补】

- checkpoint / 日志 / 曲线图在任务仓库中的路径
