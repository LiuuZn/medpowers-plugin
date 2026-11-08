---
name: studio
description: MedPowers Studio 对接：设备码授权、任务/模型就绪查询、实验与指标镜像推送、外部评测结果推送、标注复核申请、数据集接入（把本地 DICOM/NRRD/NIfTI 影像推上 Studio 建数据集、导入病例、标记就绪）。训练仓库需要了解"数据/标注能不能训、有多少 approved、模型最近评测如何"，需要把训练实验/指标/评测结果同步到 Studio 可视化，或需要把本地下载的影像数据（如 MR/CT DICOM）推送建库时使用。前提是用户已在 Studio 网页批准设备授权（token 按仓库发放、跟随用户，像 API key 一样保管）；数据集接入需额外 scope studio:dataset-manage 且令牌所属账号为管理员，只增不删；Studio 是数据与标注的唯一事实源，训练仓库永远不改标注内容，只能申请复核；git 中的 EXP 记录是实验事实的权威，Studio 只是可视化镜像。
---

# studio — 训练仓库 ↔ MedPowers Studio

## 定位（先读）

- **数据与标注**：Studio 是唯一事实源。训练仓库只读；**永远不改标注内容**。发现标注可疑时，只能申请人工复核（`review_requested`），不能改像素/框/关键帧。
- **数据集接入是"只增不删"**：允许把本地数据**推上** Studio（新建数据集、导入病例、标记就绪），**不允许**删除/归档数据集、改标注、管任务——这些仍是网页端管理员动作。导入是**复制**：数据进 Studio 存储，源数据不动；导入前确认已自行备份（`--confirm-backup`）。
- **实验**：任务仓库 git 中的 `EXP-<NNN>` 记录是权威；Studio 是实验状态、训练指标、评测结果的**只读可视化镜像**。Studio 宕机不影响训练与 git 记录。
- **任务 = 模型**：Studio 的一个 Task 对应一个模型。一个训练仓库可对应多个 Task；本地用 `task_code` 作为模型标识（EXP 记录的 `model` 字段、`versions.json` 的模型键、`data.json` 的模型索引都用它）。
- **token**：按仓库粒度的用户 API key，像其他密钥一样保管；明文存 `<工作区>/runtime/studio.token`（0600）或 `MEDPOWERS_STUDIO_TOKEN`，绝不进 git/实验记录/日志。`repo_key` 是仓库标识，同一仓库多机可共用。

## 动作

| 动作 | 命令 | 何时用 |
|---|---|---|
| auth | `python3 skills/studio/studio.py auth --url http://10.0.0.20:8000 --repo-key <repo> [--allow-insecure-http] [--scopes ...]` | 首次接入/换机/换 token；打印用户码与网页地址，用户批准后自动落盘 |
| status | `... status` | 查看连接、用户、scope、能力（含 `dataset_manage`） |
| tasks | `... tasks` | 列出任务（模型）与 approved/复核/契约摘要 |
| task | `... task <task_code>` | 单个模型就绪报告：病例数、标注状态、逐类病例、QC、标注契约 |
| exp | `... exp push --exp EXP-012 --task <task_code> [--status ...] [--config cfg.json]` | 手动推送/更新实验镜像（训练进程内嵌 `training_session` 时自动推送 running/finish） |
| metrics | `... metrics push --exp EXP-012 --file metrics.json` | 批量推送任意数值指标（`[{name,value,step,split,class?,tags?}]`） |
| eval | `... eval push --exp EXP-012 --file eval.json` | 推送外部评测结果（summary/per_case/source=external） |
| review | `... review request --annotation <id> --reason "..." [--eval <id>] [--evidence evidence.json]` | 发现可疑标注，申请人工复核；**不改标注内容** |
| sync-exp | `... sync-exp` | 重放离线队列（Studio 恢复后） |
| datasets | `... datasets [--modality MR] [--status draft] [--search 关键词]` | 列出数据集（id/code/模态/状态/病例数） |
| protocols | `... protocols` | 列出扫描流程（建数据集必须绑定一个流程） |
| protocol create | `... protocol create --code mr_routine --name "MR 常规" --modality MR` | 现有流程不适用时新建（也可网页端建） |
| dataset create | `... dataset create --code mr_knee --name "膝关节 MR" --modality MR --protocol mr_routine` | 新建数据集（status=draft；code 唯一） |
| dataset import | `... dataset import --dataset mr_knee --upload <本地路径>` 或 `--server-path <容器路径>` | 导入病例；详见「数据集接入」 |
| dataset jobs | `... dataset jobs --dataset mr_knee [--job <id> --wait]` | 查看/等待后台导入任务 |
| dataset ready | `... dataset ready --dataset mr_knee` | 病例导入齐全、报告核对无误后标记就绪 |

所有命令都支持 `--runtime <目录>`（默认 `$MEDPOWERS_RUNTIME` 或 `<工作区根>/runtime`）。

## 授权

1. 用户已部署 Studio，并在网页登录。
2. 在训练仓库执行 `auth`；终端打印 `user_code` 与 `verification_uri`。
3. 用户在浏览器打开地址、确认仓库/scope、批准。
4. CLI 轮询拿到 token，写 `runtime/studio.json` + `runtime/studio.token`（0600）。
5. `status` 验证 `auth_method=token`、`repo_key`、scope 与能力。

scope 建议：`studio:read studio:data studio:experiment studio:evaluate`；只有明确需要"提醒人工复核"时才加 `studio:review-request`；只有明确需要"推送数据建库"时才加 `studio:dataset-manage`（见下节）。token 泄漏 = 用户凭证泄漏：立即在 Studio「用户/令牌」撤销并重新授权。

## 数据集接入（本地数据 → Studio 建库）

把下载/整理好的影像数据（如 MR DICOM 目录）推上 Studio：**建数据集 → 导入病例 → 核对报告 → 标记就绪**。

**前提与授权**：scope `studio:dataset-manage`，且**令牌所属账号必须是管理员**（数据集接入是只增不删的写操作）。缺 scope/非管理员时命令报 403，按提示重新授权：

```bash
python3 skills/studio/studio.py auth --url http://10.0.0.20:8000 --repo-key <repo> \
  --scopes studio:read studio:data studio:experiment studio:evaluate studio:dataset-manage \
  --allow-insecure-http
```

`status` 的能力行出现 `dataset_manage` 即就绪。建库完成后建议尽快重新 auth 撤掉该 scope（最小权限）。

**流程**（每步的输出先看再走下一步）：

```bash
python3 skills/studio/studio.py protocols                       # 1. 有无适用流程？没有先 protocol create
python3 skills/studio/studio.py protocol create --code mr_routine --name "MR 常规" --modality MR
python3 skills/studio/studio.py dataset create --code mr_knee --name "膝关节 MR" \
  --modality MR --protocol mr_routine                           # 2. 建库（draft）
python3 skills/studio/studio.py dataset import --dataset mr_knee --upload /data/knee_mr/
python3 skills/studio/studio.py dataset show --dataset mr_knee  # 4. 核对病例数与导入错误
python3 skills/studio/studio.py dataset ready --dataset mr_knee # 5. 全部导入无误后标记就绪
```

**两条导入通道**（`dataset import` 二选一）：

| 通道 | 适用 | 机制与限制 |
|---|---|---|
| `--upload <本地路径>` | 数据在**本机**、中等体量 | 目录/文件自动打 ZIP 上传（隐藏文件与 `__MACOSX` 自动跳过），≤ 服务端 `MEDPOWERS_MAX_UPLOAD_MB`（默认 4GB）；超限时拆分数据或改走 server-path |
| `--server-path <容器路径>` | 大体量、或数据已在**服务器**上 | 路径必须是 **Studio 容器内可见**的路径。工作区部署约定：宿主机 `<实例>/imports/` 挂载为容器 `/imports`（只读）——先把数据放进实例 `imports/` 目录，再传 `/imports/<子目录>`。默认同步执行；加 `--async` 走后台任务（推荐），`--wait` 轮询到完成，`--timeout` 秒数。API 令牌并发后台任务 ≤3（超限 429） |

**支持的数据形态**（服务端自动发现）：DICOM 序列目录（按 Patient/Study/Series 建层级）、NRRD/NIfTI 体数据（文件名=case）、2D 图像（DR）、表格文件（TABLE，CSV/XLSX/Parquet）。导入报告字段：`imported/skipped/converted/stored_bytes/cases/errors/warnings`——**errors 非空先修数据重导，不要带错标 ready**。

**纪律**：

- 只增不删：CLI 没有（也不该有）删除/归档数据集的命令；建错了请管理员在网页端处理。
- 不写标注内容；导入产生的病例默认无标注，标注在网页端人工进行。
- `--confirm-backup` 的语义是"已自行备份原始数据"（后台任务服务端强制要求）：导入是复制、源数据不动，但代表用户确认过数据安全，**先向用户确认再传**。
- 数据集 code 全局唯一（重复报 409）；`--dataset` 参数接受 id 或 code。
- 大体量数据（数百例/数十 GB）优先 `--server-path --async --wait`，避免同步请求超时。

**排障**：

| 现象 | 原因与处置 |
|---|---|
| 403（缺 `studio:dataset-manage`） | token 无该 scope 或账号非管理员 → 按上文重新 auth（需管理员账号在网页批准） |
| 429 | 后台导入任务并发超限（≤3/令牌）→ 等 `dataset jobs` 中的任务完成再提交 |
| 「上传文件不是有效的 ZIP」 | `--upload` 打包失败或传了损坏压缩包 → 检查源目录；已是 .zip 的文件会原样上传 |
| 400「路径不存在/不可见」 | `--server-path` 传了容器内看不到的路径 → 数据先放进实例 `imports/` 目录（容器 `/imports`），或改用 `--upload` |
| 上传超限 | ZIP 超过 `MEDPOWERS_MAX_UPLOAD_MB` → 拆分、调大服务端配置，或改 `--server-path` |
| 409 编码已存在 / 400 流程不适配模态 | 换 `--code`；或换/新建适用该模态的扫描流程（`protocols` 查看） |

**建库后的关联动作**：数据集在 Studio 网页端关联到任务（Task）、指派标注；本地训练副本按 data-mgmt 管理，`dataset.json` 的 `studio` provenance 记 `repo_key/dataset_code/dataset_version/pulled_at`。

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
- 本轮 **不提供数据导出/下载**（Studio → 训练仓库方向）；训练数据仍按 data-mgmt 本地管理。反向的**上传建库已支持**（见「数据集接入」）；导出包与 Studio 原生评测后续版本再接。

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
