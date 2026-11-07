# MedPowers ↔ MedPowers Studio 对接契约（P0）

> 版本：2026-11-07  
> 状态：P0 冻结；导出包 / Studio 原生评测 / 模型注册表为后续版本。

## 1. 职责边界

| 事项 | 唯一事实源 | 说明 |
|---|---|---|
| 影像、病例、金标准标注、标注版本 | **Studio** | 训练仓库只读；永不改标注内容 |
| 训练副本、预处理 recipe、cache | 任务仓库 `data_root` | 从 Studio 派生时登记 provenance |
| 实验编号、配置、诊断、下一轮决策 | **任务仓库 git** | Studio 只做只读可视化镜像 |
| 严格指标评测 | 后续 Studio 原生评测 | P0 只接收训练仓库推送的外部评测结果 |

- **任务 = 模型**：Studio 一个 Task 对应一个模型；一个训练仓库可对应多个 Task。`task_code` 用于 EXP 的 `model` 字段、`versions.json` 的 `models` 键、`data.json` 的 models 索引。
- Studio 不托管 checkpoint / 模型权重；`versions.json` 仍在训练仓库。

## 2. 授权

- 设备码授权（RFC 8628）：`POST /api/auth/device/code` → 用户网页批准 → `POST /api/auth/device/token`。
- token 按仓库粒度（`repo_key`）、跟随用户；像 API key 一样保管。
- 插件存储：`<工作区>/runtime/studio.json`（非敏感配置）+ `<工作区>/runtime/studio.token`（0600）或 `MEDPOWERS_STUDIO_TOKEN`。
- 明文 HTTP 仅限内网受控，必须显式 `allow_insecure_http`；有 TLS 时配置 `ca_bundle`/`verify_tls`。

### Scope

| scope | 允许 | 禁止 |
|---|---|---|
| `studio:read` | 元数据、就绪报告、实验/指标/评测/复核查询 | 任何写 |
| `studio:data` | 二进制下载（后续导出包、切片、掩码） | 元数据写 |
| `studio:experiment` | upsert 实验、批量指标 | 标注、数据集/任务/用户管理 |
| `studio:evaluate` | 推送外部评测结果；后续原生评测 | 同上 |
| `studio:review-request` | approved → review_requested | 改标注内容、审核/撤回 |
| `studio:write` | 通用写（仅网页/管理端或极可信自动化） | 训练仓库不申请 |

- **API token 不继承用户角色**：`/api/users`、数据集/任务创建、导出/评估创建、标注保存/提交/审核/撤回只允许 JWT。
- token 的 `repo_key` 为准；请求体传其他仓库 403。

## 3. 标注复核

状态机新增：

```text
approved --review_request--> review_requested --通过--> approved
                                             --驳回--> in_progress（标注员修改后重新提交）
```

- 只改 `status`，不改 `data`/`mask_path`/`version`/revision。
- `review_requested` **不计入 approved**，不进入导出/评测金标准池。
- 申请必须带 `reason`；可选 `evidence`/`eval_id`；写 `AnnotationReviewRequest` 审计（token/repo_key/user/time/结论）。
- 训练仓库没有标注内容写权限。

## 4. 实验与指标

### 4.1 Experiment

- 键：`(repo_key, exp_id)`；`exp_id` 格式 `EXP-<NNN>`；`task_code` 必须存在。
- 状态：`planned/running/done/interrupted/cancelled`。
- 字段：`title`、`machine`、`git_commit`、`config`、`notes`、`record_path`、`record_sha256`、`started_at`、`ended_at`。
- git 记录是权威；Studio 只读展示，不提供正文编辑。

### 4.2 MetricPoint

```json
{"name":"val/dice/1","value":0.9123,"step":120,"epoch":30,
 "split":"val","class":"1","tags":{"fold":"0"},"source":"train","extra":{}}
```

- 任意数值指标名；`class` 在 JSON 中为键名，内部字段 `class_label`。
- 幂等键 `(experiment,name,split,class,step)`；重推覆盖。
- 单批 ≤ 5000 点；长曲线由 Studio 保留/降采样。

### 4.3 外部评测

```json
{"eval_id":"local-20261107-001","source":"external","task_code":"spine_keyframe",
 "gt_statuses":["approved"],"dataset_version":"dv-20261107",
 "summary":{"mean_dice":0.9,"hd95":2.1},
 "per_case":[{"case_uid":"...","dice":0.88}],"report_sha256":"..."}
```

- `source=external` 与后续 Studio 原生评测区分；不得冒充金标准评测。
- 幂等键 `(experiment,eval_id)`。

## 5. P0 API

| 方法 | 路径 | scope |
|---|---|---|
| GET | `/api/studio/meta` | read |
| GET | `/api/studio/tasks` | read |
| GET | `/api/studio/tasks/{code}/readiness` | read |
| GET | `/api/studio/changes?since=` | read |
| POST | `/api/studio/experiments` | experiment |
| PATCH | `/api/studio/experiments/{exp_id}` | experiment |
| GET | `/api/studio/experiments/{exp_id}` | read |
| POST | `/api/studio/experiments/{exp_id}/metrics:batch` | experiment |
| GET | `/api/studio/experiments/{exp_id}/metrics` | read |
| POST | `/api/studio/experiments/{exp_id}/evaluations` | evaluate |
| GET | `/api/studio/experiments/{exp_id}/evaluations` | read |
| GET | `/api/studio/tasks/{code}/experiments` | read |
| POST | `/api/annotations/{id}/request-review` | review-request |
| GET | `/api/studio/review-requests?status=open` | read |

错误：`{code, message/detail, request_id}`；401（凭证无效）/403（scope 或 JWT-only）/404/409/422/429。

## 6. 插件命令与训练集成

```bash
medpowers studio auth --url http://<host>:8000 --repo-key <repo> [--allow-insecure-http]
medpowers studio status
medpowers studio tasks
medpowers studio task <task_code>
medpowers studio exp push --exp EXP-012 --task <task_code>
medpowers studio metrics push --exp EXP-012 --file metrics.json
medpowers studio eval push --exp EXP-012 --file eval.json
medpowers studio review request --annotation <id> --reason "..." [--eval <id>]
medpowers studio sync-exp
```

训练入口：

```python
with training_session("EXP-012", task="...", gpus=[0],
                      studio_task="<task_code>") as s:
    s.report_metrics([...])
```

- 推送尽力而为：失败进 `runtime/studio-queue/`，恢复后补推；不影响训练与 git 台账。
- 未配置 `studio.json` 时自动跳过。

## 7. P0 验收

1. 设备码授权后 `repo_key` 与 scope 正确；token 不能调用管理端/写标注。
2. `task <code>` 返回 approved/review_requested 计数、逐类病例、QC、标注契约完整性。
3. `training_session(..., studio_task=...)` 推送 running/done；指标幂等；外部评测可查。
4. `review request` 后 approved 计数减少、复核队列可见；通过后恢复。
5. Studio 不可用时训练照常、队列补推；HTTP 必须显式确认。
6. `pytest`（Studio）与 `python3 -m unittest discover -s medpowers/tests -t .` 全绿。
