---
name: notify
description: 训练通知组件（实现在 medpowers Python 包，本技能负责配置与兜底）：训练开始/结束/异常中断的信号由训练进程内嵌包发送——全接管正路 `training_session`（start 门禁+lease+通知+finish 落账一体），纯通知场景 `training_lifecycle`；经 webhook（至少支持飞书机器人，可选加签校验）推送，内容含机器名称、GPU 编号、实验编号与操作说明；智能体仅保留两个发送点——exp-mgmt cancel 之后发 cancel、清理 stale lease（SIGKILL/断电等进程无法自报的终止）时补发 interrupt 兜底。配置在 init-env 问卷完成（写入任务仓库 runtime/notify.json；用户不告知则组件不生效，发送自动跳过且不报错）。凡配置通知、确认训练入口已内嵌、取消实验、清理 stale lease 时使用。
---

# notify — 训练通知组件

把训练生命周期的关键时点推送到用户的 IM（飞书等）：**训练开始 / 任何原因中断 / 训练结束 / 取消**，让用户不必盯着智能体会话也能掌握训练动态。**start / finish / interrupt 三个信号由训练进程内嵌 medpowers 包发送**（见下），智能体不代发；仅取消与无法自报终止的兜底走智能体。配置在任务仓库 `<工作区>/runtime/notify.json`，**用户不告知机器名称或 webhook 时组件不生效**（发送打印跳过原因并 exit 0，属正常路径，不是错误）。

## 配置（init-env 问卷采集，本技能维护）

- 问卷随时可重配：`python3 skills/notify/notify.py set --machine-name <机器名称> --url <webhook地址> [--type feishu|generic] [--secret <飞书加签密钥>]`（脚本为 medpowers 包的薄启动器；包安装后也可 `medpowers notify set ...`）；
- 用户选择不配置：`python3 skills/notify/notify.py disable --reason "用户选择不配置"`（保留结构，`enabled=false`）；
- `status` 查看当前配置（URL 与密钥已脱敏）；`test` 发一条测试消息，配置后**必须跑一次并请用户在 IM 里确认收到**。

配置 schema（正本 `templates/runtime/notify.example.json`）：`enabled`、`machine_name`（这台机器的名称：既用于通知文案，也是实验记录与 lease 中机器指代的正名来源——多机共读的台账禁称"本机"，规则见 exp-mgmt 纪律 9）、`webhook{type,url,secret}`（type 目前 `feishu` / `generic`；`secret` 仅飞书加签校验用）。是否纳入任务仓库 git 由任务仓库决定——**webhook 地址含敏感 token**，入库前先确认仓库私有性。

## 发送主体：训练进程内嵌 medpowers 包（start / finish / interrupt 的正路）

包由 init-env 以 path 依赖装进任务仓库训练环境（`uv add <medpowers 副本根>`，版本与插件一致）。训练入口包一层即可：

```python
from medpowers import training_session

# 全接管正路：start 门禁 → 写自己的 lease（真实 pid）→ start 通知 → 训练
# → finish 落账（指标/产物/时长自动入账）→ 删 lease → finish/interrupt 通知
with training_session("EXP-001", gpus=[0, 1], task="bc3seg-unet3d",
                      note="nnUNet 3d_fullres", log_path="runtime/logs/exp001.log") as s:
    s.artifacts.append("runs/exp001/best.pth")   # 训练中随时登记产物
    main_train()
```

- **两级纪律**：记录门禁（start 门禁失败训练不得开始）是失败即停；通知是尽力而为（绝不影响训练与收尾）——详见 medpowers/session.py 与 AGENTS.md；
- 纯通知场景（实验记录不经包管理时）用 `training_lifecycle`：进入发 `start`、正常走完（含 `sys.exit(0)`）发 `finish`、任何异常/SIGTERM/SIGINT 发 `interrupt`（带原因）后训练行为零改变；训练代码自行接管信号时 `install_signals=False`，在其处理器里调用 `send_notification`；
- runtime 解析：`$MEDPOWERS_RUNTIME` > 从 cwd 向上最近含 `.git` 祖先下的 `runtime/`——**训练进程的 cwd 须在任务仓库工作区内**（或启动时设 `MEDPOWERS_RUNTIME`）；
- **能力边界（诚实声明）**：SIGKILL、OOM-killer、断电、机器死机这类进程无法自报的终止，内嵌钩子覆盖不到——由智能体清理 stale lease 时补发 `interrupt` 兜底（templates/runtime/README.md）。

## 智能体侧仅存的两个发送点（CLI）

| 时点 | 命令 |
|---|---|
| 取消实验（exp-mgmt `cancel` 之后；训练进程不参与取消） | `python3 skills/notify/notify.py send --event cancel --experiment EXP-001 --note "<取消理由>"` |
| stale lease 清理兜底（进程已不存在——SIGKILL/断电等无法自报的终止） | `... send --event interrupt --gpus 0,1 --experiment EXP-001 --note "stale lease 清理：进程已不存在"` |

`--event` 也接受自定义文本（如 `--event "数据准备完成"`，按信息卡发送）；`--style text` 可退回纯文本消息。

## 进程内 API（训练代码内嵌用，通知路径绝不抛异常）

```python
from medpowers import send_notification, training_lifecycle

send_notification("interrupt", gpus=[0], experiment="EXP-002", note="CUDA OOM")
# 返回 {"sent": bool, ...}；任何故障折叠为 {"sent": False, "skipped"/"error"}——
# training_session / training_lifecycle / send_notification 的通知路径均如此
```

## 纪律（强制）

1. **尽力而为，绝不影响训练**：通知路径全程旁路——绝不抛异常、单次请求 10 秒超时、失败重试 1 次后放弃；interrupt 后异常照常上抛、信号照常按默认语义终止。智能体侧的 send 失败同样重试 1 次后放弃（send exit 1 提示），**不得用 `&&` 把 send 与 exp-mgmt 动作串联**，更不得因通知失败回滚记录状态。
2. **未配置不是错误**：用户未配置 webhook 时发送打印"跳过"并 exit 0。不要因此重试、报错或反复向用户追问 webhook——那是 init-env 问卷的职责，问过一次即可。
3. **内容三要素**：机器名称（配置带入，通知标题与正文必有）、GPU 编号（`gpus=`，训练类通知必给）、操作信息（事件 + `experiment` + `note`）；**中断通知必须写明原因**。
4. **敏感信息不落记录**：webhook URL 含 token、`secret` 是签名密钥；status/test 的输出已脱敏，不要把完整 URL 或密钥写进会话日志、实验记录或提交信息。
