# MedPowers ↔ MedPowers Studio 对接契约（P0 + P1 数据集接入）

> 版本：2026-11-08（增补 P1 数据集接入，见 §8；P0 部分维持 2026-11-07 冻结）  
> 状态：导出包 / Studio 原生评测 / 模型注册表为后续版本。

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
| `studio:experiment` | upsert 实验、批量指标 | 标注、任务/用户管理 |
| `studio:evaluate` | 推送外部评测结果；后续原生评测 | 同上 |
| `studio:review-request` | approved → review_requested | 改标注内容、审核/撤回 |
| `studio:dataset-manage`（P1） | 数据集接入：新建数据集、导入病例、标记就绪、新建扫描流程 | 删除/归档数据集、改标注、任务管理 |
| `studio:write` | 通用写（仅网页/管理端或极可信自动化） | 训练仓库不申请 |

- **API token 不继承用户角色**：`/api/users`、任务创建、导出/评估创建、标注保存/提交/审核/撤回只允许 JWT。
- **数据集接入例外（P1，§8）**：持有 `studio:dataset-manage` 且**所属账号为管理员**的 token 可新建数据集/导入/标记就绪/新建扫描流程；仍不可删除或归档。
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

## 8. P1 数据集接入（2026-11-08 增补）

> 对应 Studio 端提交 `38edf00`（允许 API 令牌管理数据集并增加接入审计）。**只增不删**：
> 没有删除/归档命令；删除、归档、任务管理、标注全部仍是网页端管理员动作。

### 8.1 授权

- scope：`studio:dataset-manage`，且**令牌所属账号必须是管理员**（`require_admin_or_dataset_manage`）。
- 网页管理员 JWT 本就允许；CLI/自动化走 token 时在设备码授权时勾选该 scope。
- 完成建库后建议重新授权撤掉该 scope（最小权限）。

### 8.2 API

| 方法 | 路径 | scope | 说明 |
|---|---|---|---|
| GET | `/api/scan-protocols` | read | 列出扫描流程（数据集必须绑定流程） |
| POST | `/api/scan-protocols` | dataset-manage | 新建扫描流程 |
| GET | `/api/datasets` | read | 列出数据集（modality/status/search 过滤） |
| POST | `/api/datasets` | dataset-manage | 新建数据集（`code` 唯一，必填 `modality` + `scan_protocol_id`，初始 `draft`） |
| GET | `/api/datasets/{id}` | read | 数据集详情（含病例/患者/检查计数） |
| PATCH | `/api/datasets/{id}` | dataset-manage | 改名称/描述/流程 |
| POST | `/api/datasets/{id}/import-path` | dataset-manage | 服务端本地路径导入（**同步**，路径须容器可见） |
| POST | `/api/datasets/{id}/import-upload` | dataset-manage | ZIP 上传导入（**同步**，multipart，≤ `MEDPOWERS_MAX_UPLOAD_MB` 默认 4GB） |
| POST | `/api/datasets/{id}/import-jobs` | dataset-manage | 后台路径导入任务（`acknowledge_source_discarded` 必须为 true；令牌并发 ≤3，超限 429） |
| GET | `/api/datasets/{id}/import-jobs[/{job_id}]` | read | 查询导入任务；状态 queued→running→succeeded/failed |
| POST | `/api/datasets/{id}/ready` | dataset-manage | 标记就绪（`case_count>0` 才允许） |

错误语义同 P0（401/403/404/409/422/429）。导入是**复制**：数据进 Studio 存储，源数据不动；
`acknowledge_source_discarded` 表示"已自行备份原始数据"的用户确认。

### 8.3 数据形态与层级

服务端自动发现：DICOM 序列目录（建 Patient→Study→Series 层级，patient_key 由 PatientID 派生）、
NRRD/NIfTI 体数据（文件名=case_uid）、2D 图像、表格文件。导入报告：
`{imported, skipped, converted, stored_bytes, cases, errors, warnings}`。

### 8.4 部署约定（工作区）

Studio 容器把宿主机 `<实例>/imports/` 只读挂载为容器 `/imports`（`MP_IMPORT_DIR`，compose 默认 `./imports`）。
CLI `--server-path` 只能传容器内可见路径；工作区工作流是先把数据放进实例 `imports/` 目录再传 `/imports/<子目录>`。

### 8.5 插件命令

```bash
medpowers studio datasets [--modality MR] [--status draft] [--search ...]
medpowers studio protocols
medpowers studio protocol create --code mr_routine --name "MR 常规" --modality MR
medpowers studio dataset create --code mr_knee --name "膝关节 MR" --modality MR --protocol mr_routine
medpowers studio dataset import --dataset mr_knee --upload <本地路径> [--confirm-backup]
medpowers studio dataset import --dataset mr_knee --server-path /imports/knee --async --wait --confirm-backup
medpowers studio dataset jobs --dataset mr_knee [--job <id> --wait]
medpowers studio dataset show --dataset mr_knee
medpowers studio dataset ready --dataset mr_knee
```

### 8.6 P1 验收

1. 无 `studio:dataset-manage` 的 token 调用数据集接入端点 403；非管理员账号的 token 同样 403。
2. 建数据集 → 上传/路径导入 → 报告正确 → 标记就绪；`dataset show` 计数一致。
3. 后台任务并发第 4 个时 429；任务失败时 `error`/`report.errors` 可见。
4. token 始终无法：删除/归档数据集、修改标注、创建任务、用户管理。
