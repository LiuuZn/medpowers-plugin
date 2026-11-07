---
name: studio
description: MedPowers Studio 对接：设备码授权、任务/模型就绪查询、实验与指标镜像推送、外部评测结果推送、标注复核申请。训练仓库需要了解"数据/标注能不能训、有多少 approved、模型最近评测如何"，或需要把训练实验/指标/评测结果同步到 Studio 可视化时使用。前提是用户已在 Studio 网页批准设备授权（token 按仓库发放、跟随用户，像 API key 一样保管）；Studio 是数据与标注的唯一事实源，训练仓库永远不改标注内容，只能申请复核；git 中的 EXP 记录是实验事实的权威，Studio 只是可视化镜像。
---

# studio — 训练仓库 ↔ MedPowers Studio

## 定位（先读）

- **数据与标注**：Studio 是唯一事实源。训练仓库只读；**永远不改标注内容**。发现标注可疑时，只能申请人工复核（`review_requested`），不能改像素/框/关键帧。
- **实验**：任务仓库 git 中的 `EXP-<NNN>` 记录是权威；Studio 是实验状态、训练指标、评测结果的**只读可视化镜像**。Studio 宕机不影响训练与 git 记录。
- **任务 = 模型**：Studio 的一个 Task 对应一个模型。一个训练仓库可对应多个 Task；本地用 `task_code` 作为模型标识（EXP 记录的 `model` 字段、`versions.json` 的模型键、`data.json` 的模型索引都用它）。
- **token**：按仓库粒度的用户 API key，像其他密钥一样保管；明文存 `<工作区>/runtime/studio.token`（0600）或 `MEDPOWERS_STUDIO_TOKEN`，绝不进 git/实验记录/日志。`repo_key` 是仓库标识，同一仓库多机可共用。

## 动作

| 动作 | 命令 | 何时用 |
|---|---|---|
| auth | `python3 skills/studio/studio.py auth --url http://10.0.0.20:8000 --repo-key <repo> [--allow-insecure-http] [--scopes ...]` | 首次接入/换机/换 token；打印用户码与网页地址，用户批准后自动落盘 |
| status | `... status` | 查看连接、用户、scope、能力 |
| tasks | `... tasks` | 列出任务（模型）与 approved/复核/契约摘要 |
| task | `... task <task_code>` | 单个模型就绪报告：病例数、标注状态、逐类病例、QC、标注契约 |
| exp | `... exp push --exp EXP-012 --task <task_code> [--status ...] [--config cfg.json]` | 手动推送/更新实验镜像（训练进程内嵌 `training_session` 时自动推送 running/finish） |
| metrics | `... metrics push --exp EXP-012 --file metrics.json` | 批量推送任意数值指标（`[{name,value,step,split,class?,tags?}]`） |
| eval | `... eval push --exp EXP-012 --file eval.json` | 推送外部评测结果（summary/per_case/source=external） |
| review | `... review request --annotation <id> --reason "..." [--eval <id>] [--evidence evidence.json]` | 发现可疑标注，申请人工复核；**不改标注内容** |
| sync-exp | `... sync-exp` | 重放离线队列（Studio 恢复后） |

所有命令都支持 `--runtime <目录>`（默认 `$MEDPOWERS_RUNTIME` 或 `<工作区根>/runtime`）。

## 授权

1. 用户已部署 Studio，并在网页登录。
2. 在训练仓库执行 `auth`；终端打印 `user_code` 与 `verification_uri`。
3. 用户在浏览器打开地址、确认仓库/scope、批准。
4. CLI 轮询拿到 token，写 `runtime/studio.json` + `runtime/studio.token`（0600）。
5. `status` 验证 `auth_method=token`、`repo_key`、scope 与能力。

scope 建议：`studio:read studio:data studio:experiment studio:evaluate`；只有明确需要"提醒人工复核"时才加 `studio:review-request`。token 泄漏 = 用户凭证泄漏：立即在 Studio「用户/令牌」撤销并重新授权。

## 训练集成（正路）

训练入口用 `training_session(..., studio_task="<task_code>")`。训练进程会：

- 进入：start 门禁 + lease + 通知后，把实验 `running` 镜像到 Studio；
- 训练中：`s.report_metrics([{...}, ...])` 批量推送指标（内部 200 点批量，失败进 `runtime/studio-queue/`）；
- 退出：flush 指标，推送 `done/interrupted`；Studio 不可用只警告，不影响训练。

```python
from medpowers import training_session

with training_session("EXP-012", task="spine-keyframe-unet",
                      gpus=[0], studio_task="spine_keyframe") as s:
    s.report_metrics([{"name": "train/loss", "value": loss, "step": step,
                       "split": "train", "tags": {"fold": "0"}}])
    s.artifacts.append("runs/exp012/best.pt")
    main_train()
```

未配置 `studio.json` 时 `studio_task` 自动跳过，不报错、不追问。

## 指标与评测格式

指标点（数值型，name 自由命名，建议 `train/loss`、`val/dice/<class>`）：

```json
{"name": "val/dice/1", "value": 0.9123, "step": 120, "epoch": 30,
 "split": "val", "class": "1", "tags": {"fold": "0"}, "source": "train"}
```

评测结果（外部推送，不冒充 Studio 原生评测）：

```json
{
  "eval_id": "local-20261107-001",
  "source": "external",
  "task_code": "spine_keyframe",
  "gt_statuses": ["approved"],
  "dataset_version": "dv-20261107",
  "summary": {"mean_dice": 0.9, "hd95": 2.1},
  "per_case": [{"case_uid": "...", "dice": 0.88}],
  "report_sha256": "..."
}
```

幂等：同 `(exp, name, split, class, step)` 覆盖；同 `(experiment, eval_id)` 覆盖。

## 数据就绪与实验规划

- 规划 EXP 前先 `task <task_code>`：`approved` 是唯一训练金标准；`review_requested` 已移出 approved 池；`annotation_contract_complete=false` 时先让用户补契约（`produced_by/rule/tolerance/verified`）。
- `qc_issues` 非空先修数据或标注，不要带病训练。
- 本轮 P0 **不提供数据导出/下载**；训练数据仍按 data-mgmt 本地管理。导出包与 Studio 原生评测后续版本再接。

## 复核申请纪律

- 只对 `approved` 标注申请；提交后状态变 `review_requested`，内容/revision 不变，但立即移出 approved 池（不参与后续导出/评测金标准），直到复核通过。
- 必须写 `reason`；尽量附 `eval_id` 与 `evidence`（指标、切片号、最差例）。
- 不能提交/审核/撤回标注；这些仍是网页端人工动作。
- 批量翻转请先与用户确认，避免把训练数据池一夜清空。

## 离线与失败

- 推送尽力而为：失败写 `runtime/studio-queue/<repo_key>/<exp>.jsonl`，下次 `sync-exp` 或训练进程下次运行补推；服务端幂等，不会重复。
- 读接口失败直接报错，不使用旧缓存冒充最新；训练已开始则不受 Studio 影响。
- HTTP 仅限内网受控：`studio.json` 必须显式 `allow_insecure_http=true`；有 TLS 时配置 `ca_bundle`/`verify_tls`。

## 相关

- Studio 端契约：`docs/studio-integration-contract.md`（本仓库）、Studio `README.md`/`AGENTS.md`。
- 训练记录：`skills/exp-mgmt/SKILL.md`（`new --task`、`model` 字段、Studio 来源块）。
- 数据格式：`skills/data-mgmt/SKILL.md`（`keyframe3d` 为统一任务类型；`plane3d` 为历史别名）。
- 通知：`skills/notify/SKILL.md`（与 Studio 推送同属"尽力而为"）。
