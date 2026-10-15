---
name: notify
description: 训练通知组件：经 webhook（至少支持飞书机器人，可选加签校验）推送训练开始、任何原因中断、训练结束/取消的提醒，内容含机器名称、GPU 编号、实验编号与操作说明。配置在 init-env 问卷完成（用户告知机器名称与 webhook 链接，写入任务仓库 runtime/notify.json；用户不告知则组件不生效，send 自动跳过且不报错）。凡启动训练、发现训练中断/崩溃/清理 stale lease、训练结束或取消实验时使用。
---

# notify — 训练通知组件

把训练生命周期的关键时点推送到用户的 IM（飞书等）：**训练开始 / 任何原因中断 / 训练结束（含取消）**，让用户不必盯着智能体会话也能掌握训练动态。配置在任务仓库 `<工作区>/runtime/notify.json`，**用户不告知机器名称或 webhook 时组件不生效**（send 打印跳过原因并 exit 0，属正常路径，不是错误）。

## 配置（init-env 问卷采集，本技能维护）

- 问卷随时可重配：`python3 skills/notify/notify.py set --machine-name <本机名称> --url <webhook地址> [--type feishu|generic] [--secret <飞书加签密钥>]`；
- 用户选择不配置：`python3 skills/notify/notify.py disable --reason "用户选择不配置"`（保留结构，`enabled=false`）；
- `status` 查看当前配置（URL 与密钥已脱敏）；`test` 发一条测试消息，配置后**必须跑一次并请用户在 IM 里确认收到**。

配置 schema（正本 `templates/runtime/notify.example.json`）：`enabled`、`machine_name`（通知中标识本机）、`webhook{type,url,secret}`（type 目前 `feishu` / `generic`；`secret` 仅飞书加签校验用）。是否纳入任务仓库 git 由任务仓库决定——**webhook 地址含敏感 token**，入库前先确认仓库私有性。

## 发送时机（与 exp-mgmt / lease 协议对齐，全部由智能体在动作完成后调用）

| 时点 | 命令 |
|---|---|
| 训练开始（过 start 门禁、lease 写入且训练后台分离启动成功后） | `python3 skills/notify/notify.py send --event start --gpus 0,1 --experiment EXP-001 --note "一句话说明"` |
| 任何原因中断（崩溃、OOM、手动终止、发现进程消失、stale lease 清理时补发） | `... send --event interrupt --gpus 0,1 --experiment EXP-001 --note "<中断原因>"` |
| 训练结束（exp-mgmt `finish --status done` 之后） | `... send --event finish --gpus 0,1 --experiment EXP-001` |
| 训练中断收尾（exp-mgmt `finish --status interrupted` 之后） | `... send --event interrupt ... --note "<原因与已产生的观察>"` |
| 取消实验（exp-mgmt `cancel` 之后） | `... send --event cancel --experiment EXP-001 --note "<取消理由>"` |

`--event` 也接受自定义文本（如 `--event "数据准备完成"`，按信息卡发送）；`--style text` 可退回纯文本消息。

## 函数级 API（任务仓库脚本内调用）

```python
sys.path.insert(0, "<mtskills>/skills/notify")
import notify
notify.send_notification("interrupt", gpus=[0], experiment="EXP-002", note="CUDA OOM")
# 返回 {"sent": bool, ...}；绝不抛异常——任何故障折叠为 {"sent": False, "skipped"/"error"}
```

## 纪律（强制）

1. **尽力而为，绝不阻断**：通知失败重试 1 次后放弃（send exit 1 提示）。不得因通知失败中止训练、回滚实验记录或 lease 动作；**不得用 `&&` 把 send 与训练启动/finish 串联**（通知挂了不能拖住主流程）。
2. **未配置不是错误**：用户未配置 webhook 时 send 打印"跳过"并 exit 0。不要因此重试、报错或反复向用户追问 webhook——那是 init-env 问卷的职责，问过一次即可。
3. **内容三要素**：机器名称（配置带入，通知标题与正文必有）、GPU 编号（`--gpus`，训练类通知必给）、操作信息（`--event` + `--experiment` + `--note`）；**中断通知必须写明原因**。
4. **敏感信息不落记录**：webhook URL 含 token、`secret` 是签名密钥；status/test 的输出已脱敏，不要把完整 URL 或密钥写进会话日志、实验记录或提交信息。
