---
name: init-env
description: 初始化 MedPowers 运行环境：第一件事是任务画像问卷——询问用户目标、医疗影像模态、2D/2.5D/3D 建模维度、分割/分类/配准等任务类型，用户必须全部答复后才能继续任何后续步骤；随后查询 GPU 静态信息（显存总量）、询问用户哪些卡可用（默认全部可用）、检查 PyTorch + CUDA 张量训练环境（默认 uv 虚拟环境，不可用时经用户同意用 uv 初始化整套环境，用户也可指定系统或其他解释器）、探测 bf16/fp16/TF32/FP8/MXFP8 计算精度能力、询问训练通知配置（机器名称与 webhook 链接，可不配置）、询问 MedPowers Studio 对接（地址/仓库标识/是否申请复核权限，可不配置），并把全部结果持久化到任务仓库工作区的 runtime/env.json、runtime/notify.json 与 runtime/studio.json。会话首次涉及训练、调优、数据管理或资源调度时使用。
---

# init-env — 初始化环境

只做初始化：GPU 只查**静态信息**（index/name/显存总量）。利用率、温度、进程等动态信息不在本技能范围，需要时实时 `nvidia-smi`。

**第 1 步是硬性门禁**：任务画像没有答复，后续步骤（GPU 清单、Python 环境、写入 env.json）一律不得开始——不知道任务是什么，选基础模型、推荐配置都无从谈起。

## 流程（含五次用户交互；Studio 对接可选）

### 第 1 步：任务画像问卷（必须问、用户必须答；无默认值）

用 AskUserQuestion 一次问 4 个问题：

1. **本次目标**（header「本次目标」）：这次要通过 MedPowers 达成什么？
   - 新任务完整回路——从基础模型选型开始，走"选型 → 训练 → 诊断 → 下一轮"
   - 已有任务调优——训练已在进行，要诊断训练状态并推荐下一轮配置
   - 暂不训练——先完成环境初始化（画像仍须按计划中的任务回答，不得留空）
2. **影像模态**（header「影像模态」，multiSelect）：数据是什么医疗影像？
   - CT / MRI / X 光（CR、DR）/ 超声；病理、内镜、眼底、多模态等用 Other 说明
3. **建模维度**（header「维度」）：影像按什么方式建模？
   - 2D——单张图 / 逐切片独立处理
   - 2.5D——2D 模型，相邻切片作为上下文输入
   - 3D——整个体数据一起处理
4. **任务类型**（header「任务类型」，multiSelect）：是什么任务？
   - 分割 / 分类 / 配准 / 检测定位；重建、生成、质量评估等用 Other 说明

规则：

- 与第 3 步"可用卡"不同，这里**没有默认回落**：用户不答复就停在原地继续追问，**不得替用户假设任务**。选"暂不训练"也必须回答其余三问（按计划中的任务答）。
- 答复原文记入 env.json 的 `task_profile`，是 `methods/base-model-selection.md` 任务画像（模态、2D/2.5D/3D、结构尺度……）的直接输入。

### 第 2 步：GPU 静态清单

```
python3 skills/init-env/init_env.py gpu-static
```

### 第 3 步：询问可用卡（必须问；答复与否都要记录）

用 AskUserQuestion 问用户："哪些 GPU 可用于本训练目录？"，并说明**默认所有卡都可用**。

- 用户答复 → `gpu_availability.source = "user"`，`answer` 记答复原文，按答复把每张卡的 `available` 标 true/false；
- 用户未答复/跳过 → `source = "default"`，`answer` 记"默认：所有卡可用"，全部卡 `available = true`。

### 第 4 步：Python 训练环境（只支持 PyTorch + CUDA）

1. 用户指定了解释器（系统的或某个虚拟环境的）→ `python3 skills/init-env/init_env.py check-python <路径>`；
2. 未指定 → 默认用 uv 虚拟环境（任务仓库工作区根 `.venv`）：直接 `check-python`（脚本会优先查工作区的 `.venv/bin/python`，不存在则回落到系统 python3 并在报告中注明）；
3. 检查不通过（torch 缺失 / CUDA 张量不可用）→ 询问用户："是否通过 uv 初始化一整套运行环境？"
   - 同意 → `init-uv [--python 3.11]`（创建 pyproject + `.venv` 并安装 torch；**默认走阿里 PyPI 镜像** `https://mirrors.aliyun.com/pypi/simple/`，同时写入 pyproject 的 `[tool.uv.index]`，后续 `uv add/sync` 也走同一镜像；换源用 `--index <URL>`；下载量数 GB，建议后台运行），完成后重新 `check-python`；
   - 拒绝 → 如实记录 `status: "not-ready"` 与原因；
4. 检查/初始化结果写入 env.json 的 `python_env`（`ready` / `not-ready`）；
5. **安装 medpowers 训练运行时客户端（python_env 为 ready 时）**——硬规则：**写入任务仓库 pyproject 的 medpowers 源必须跨机可解析**（pyproject/uv.lock 随 git 共享，他机 `uv sync` 按同一路径找包，路径失效即 import 失败）：
   - **先查现状**：读 pyproject 的 `[tool.uv.sources]`——已有 medpowers 源时先验证指向存在且 `uv run python -c "from medpowers import training_session"` 通过；**指向失效**（典型：他机写入的 `../mtskills` 兄弟目录路径或他机的插件缓存绝对路径在本机不存在）→ 按下条改为仓内相对源修复，并在汇报中提醒用户：**所有共用该仓库的机器都要做同样修复**；
   - **默认：仓内相对源**：任务仓库无 `.agents/medpowers` 时先按 INSTALL.md §5 装一份（嵌套 clone 须加 .gitignore；submodule 免），再 `cd <任务仓库工作区> && uv add ./.agents/medpowers`——写入 `path = ".agents/medpowers"`，每台机器各有一份副本、路径恒可解析，版本与插件一致（单机/多机通用）；
   - 备选（不愿放仓内副本的多机仓库）：git tag 依赖 `uv add "medpowers@git+https://gitlab.suoyin.info/medpowers/medpowers-plugin.git@v<插件版本>"`，锁 tag、跨机只需 GitLab 凭据；
   - **禁止提交进多机共享 pyproject**：兄弟目录相对路径（`../mtskills` 这类）与插件缓存绝对路径只对写入的机器有效；单机仓库确要用绝对路径源（仅插件模式、不放 .agents 副本）必须向用户说明转多机前需改源；
   - 装完冒烟：`uv run python -c "from medpowers import training_session; print('ok')"`；失败不阻断初始化，在汇报中注明并建议后续补装。

### 第 5 步：计算精度能力探测（python_env 为 ready 时；不询问用户）

```
python3 skills/init-env/init_env.py probe-precision [解释器]
```

把输出中的 `devices`/`dtypes`/`recommended` 连同探测时间写进 env.json 的 `compute_precision`，并加 `"policy": "methods/precision-policy.md"`。要点：

- 探测是**真实张量实测**（bf16/fp16 矩阵乘、FP8 `torch._scaled_mm` 逐个 API 变体试），不是只看 GPU 型号；
- 推荐值直接决定训练/推理默认：bf16 实测可用 → 训练 `bf16-amp`；否则 fp16+GradScaler 或 fp32+TF32；FP8 实测可用才允许 `inference_fast: fp8-e4m3`；
- MXFP8 仅 Blackwell（sm_100+）且 torch 具备 E8M0 dtype；该标志是能力探测，真正启用前还须用推理引擎跑一次 MXFP8 kernel。RTX 4090 是 Ada sm_89，**不支持 MXFP8**（FP8 可用，可靠路径是 TensorRT）；
- 多卡异构时 `recommended` 按最高算力卡给出；实际训练用哪张卡时按该卡能力复核。
- 探测失败不阻断初始化：`compute_precision.error` 记原因，汇报中注明，训练/推理回落到 fp32/fp16 并说明。

### 第 6 步：训练通知配置（必须问；用户可不配置，不配置则组件不生效）

用 AskUserQuestion 一次问 2 个问题：

1. **机器名称**（header「机器名称」）："这台机器叫什么名字？（用于训练通知、实验记录与 lease 中标识这台机器——多机协作时凭它区分是哪台机器在跑）"
   - 使用系统主机名（`<hostname>`，把实测值写在选项里）
   - 不配置（通知组件不生效）
   - 自定义名称在 Other 中填写
2. **Webhook**（header「Webhook」）："是否提供 webhook 链接以启用训练通知？（训练开始/中断/结束时推送到 IM，至少支持飞书机器人）"
   - 不提供（通知组件不生效）
   - 提供飞书机器人 webhook——URL 在 Other 中粘贴；机器人若开了**加签校验**，密钥一并给出

规则：

- 两问任一为"不配置"→ `python3 skills/notify/notify.py disable --reason "用户在 init-env 选择不配置"`。组件从此不生效：之后所有 send 动作自动跳过（exit 0），**不要在别处反复追问用户要 webhook**。
- 配置 → `python3 skills/notify/notify.py set --machine-name <名称> --url <URL> [--secret <密钥>]`，然后跑 `python3 skills/notify/notify.py test` 发测试消息，**请用户在 IM 里确认收到**才算配置完成。
- 通知配置存 `runtime/notify.json`（不在 env.json），schema 见 `templates/runtime/notify.example.json`；webhook URL 含敏感 token，提示用户注意任务仓库的私有性。
- 已有 `runtime/notify.json` 时，先 `notify.py status` 向用户展示现配置（已脱敏）：用户确认沿用则不动，要求修改才 set/disable——**不得静默覆盖**。

### 第 7 步：MedPowers Studio 对接（必须问；用户可不配置，不配置则组件不生效）

用 AskUserQuestion 一次问 3 个问题：

1. **是否接入 Studio**（header「Studio」）："是否把本训练仓库接入 MedPowers Studio（数据/标注查询、实验与指标镜像、评测结果推送）？"
   - 接入（提供 Studio 地址与仓库标识）
   - 暂不接入（不生成 `runtime/studio.json`，后续可随时执行 `studio auth`）
2. **Studio 地址与仓库标识**（header「Studio 地址」）：内网地址（如 `http://10.0.0.20:8000`）+ 仓库标识 `repo_key`（默认用任务仓库目录名，小写短横线）。同一仓库多机共用同一 `repo_key`。
3. **复核权限**（header「复核权限」）："是否允许训练仓库把可疑标注从「已通过」转为「待复核」提醒人工检查？"
   - 允许（scope 加 `studio:review-request`；仍不能改标注内容）
   - 不允许（默认只读 + 数据下载 + 实验/指标 + 评测）

规则：

- 接入 → 执行 `python3 skills/studio/studio.py auth --url <URL> --repo-key <repo> [--allow-insecure-http] [--scopes ...]`；把用户码与网页地址展示给用户，**等用户在浏览器批准**；成功后 `studio.py status` 验证 `repo_key`、scope、能力。
- **HTTP 显式确认**：Studio 仅内网受控部署时，明文 HTTP 会泄露 token；必须让用户明确同意后才加 `--allow-insecure-http`，并在汇报中提示风险。
- token 明文只写 `runtime/studio.token`（0600）或由用户提供 `MEDPOWERS_STUDIO_TOKEN`；**不得打印、不得写入 env.json/notify.json/实验记录/日志**。提示用户把 `studio.token` 排除在 git 之外。
- 已有 `runtime/studio.json` 时先 `studio.py status` 展示（脱敏）：用户确认沿用则不动，要求修改才重新 auth——**不得静默覆盖**。
- 用户选择暂不接入 → 不创建配置文件；之后所有 Studio 推送自动跳过（与 notify 同策略），不要反复追问。
- 建议顺便确认该仓库要训练哪些模型（= Studio task code），用于后续 `exp-mgmt new --task <task_code>` 与 `versions.json` 的 `models` 键；用户不确定可留待规划时再定。

### 第 8 步：保存与汇报

按下方 schema 组装完整内容（含 `task_profile`），管道交给 save 原子写入；save 会校验 `task_profile` 必填且非空，缺失即拒绝写入。最后向用户汇报：任务画像、可用卡、训练环境状态、训练/推理默认精度（来自 `compute_precision.recommended`，含 MXFP8 是否可用）、训练通知配置（已启用/未启用，已启用须确认测试消息送达）、Studio 对接状态（已接入/未接入，已接入须说明 repo_key、scope、模型/task 映射与 HTTP 风险）。

```
<组装好的 JSON> | python3 skills/init-env/init_env.py save
```

## env.json schema（save 会校验必填字段；saved_at 由 save 自动补）

```json
{
  "task_profile": {
    "goal": "用户对'本次目标'的答复（选项 + Other/备注补充）",
    "modality": ["CT"],
    "dimension": "2.5D",
    "task_type": ["分割"],
    "answers_raw": "四问答复原文汇总（可选，便于溯源）",
    "asked_at": "ISO8601"
  },
  "gpus": [
    {"index": 0, "name": "NVIDIA GeForce RTX 4090", "mem_total_mb": 23028, "available": true}
  ],
  "gpu_availability": {
    "source": "user 或 default",
    "answer": "用户答复原文，或 '默认：所有卡可用'",
    "asked_at": "ISO8601"
  },
  "python_env": {
    "type": "uv-venv / system / custom-venv",
    "interpreter": "/path/to/python",
    "python": "3.12.3",
    "torch": "2.x.x 或 null",
    "cuda_available": false,
    "cuda_tensor_ok": false,
    "status": "ready / not-ready",
    "note": "not-ready 的原因或其他说明"
  },
  "compute_precision": {
    "interpreter": "/path/to/python",
    "torch": "2.x.x",
    "cuda_available": true,
    "devices": [
      {"index": 0, "name": "NVIDIA GeForce RTX 4090", "capability": "8.9"}
    ],
    "dtypes": {
      "fp16": true, "bf16": true, "tf32": true,
      "fp8_e4m3": true, "fp8_e5m2": true, "mxfp8": false,
      "fp16_matmul_ok": true, "bf16_matmul_ok": true, "fp8_matmul_ok": true
    },
    "runtime": {},
    "recommended": {
      "train": "bf16-amp",
      "inference": "bf16",
      "inference_fast": "fp8-e4m3"
    },
    "policy": "methods/precision-policy.md",
    "checked_at": "ISO8601"
  }
}
```

## 约束

- 任务仓库工作区的 `runtime/env.json` 是训练环境与任务画像的**单一事实源**（是否纳入任务仓库的 git 由任务仓库自行决定）。
- 重新初始化覆盖 env.json 前，先读旧记录：若 `gpu_availability.source == "user"`，必须再次询问用户，不得静默沿用或丢失用户上次的答复；`task_profile` 同理——**每次初始化都必须重新问**（任务没变时用户可确认沿用原文），不得静默沿用。
- 未收集到任务画像时，save 的校验会拒绝写入 env.json——这是"不答画像不得完成初始化"的兜底。
- `python_env.status == "ready"` 时 `compute_precision` 必须有值；探测失败也要写 `compute_precision.error`，不得留空或假装可用。精度选择的正本是 `methods/precision-policy.md`。
- Studio 配置存 `runtime/studio.json`（schema 见 `templates/runtime/studio.example.json`），token 明文只存 `runtime/studio.token`（0600）或环境变量；Studio 是数据/标注唯一事实源，训练仓库永远不改标注内容，只能申请复核；git 中的 EXP 记录仍是实验事实权威，Studio 只做可视化镜像。
- 训练/长时间占卡前仍须按 runtime lease 协议**三确认**（协议模板 `templates/runtime/README.md`：env.json 可用性 + 无冲突 lease + `nvidia-smi` 实测）；启动训练时进程必须**后台分离**（`setsid`/`nohup`/tmux 脱离智能体会话，不得作为会话前台子进程，详见协议「训练进程必须后台分离运行」节）；env.json 只声明"哪些卡归本目录可用"，不反映实时占用。
