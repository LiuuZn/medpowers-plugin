# MTSkills — Medical Model Skill

## 仓库定位

git 管理的方法论仓库（全称 **Medical Model Skill**），服务对象是医疗场景深度学习**小模型**的训练效率与收敛精度。**本仓库不承载训练**：训练在任务仓库（`/ssd/lx` 下各子项目）中进行；本仓库提供四类任务无关的知识资产、一套"选模型 → 诊断训练状态 → 推荐下一轮配置"的方法论，以及实验记录规范（记录本身由任务仓库存放与管理，绝不纳入本仓库 git）。本仓库可经插件市场机制安装为用户级插件（见 README.md「插件安装」），也可 submodule / 嵌套 clone 进训练仓库。

## 目录结构

```
mtskills/
├── assets/                  # 任务无关的知识资产（按类型分四类）
│   ├── models/              # ① 模型架构及其论文（模型卡）
│   ├── preprocessing/       # ② 预处理方法（本仓库"数据处理"专指预处理）
│   ├── augmentation/        # ③ 数据增强方法（增强卡；训练时在线随机变换，与训练前一次性预处理区分；不配代码）
│   └── objectives/          # ④ 损失函数 / 目标函数 / 不可导评价指标
├── methods/                 # 方法论（按流程顺序）
│   ├── base-model-selection.md       # 第一步：如何选择基础模型
│   ├── training-state-diagnosis.md   # 第二步：结合损失/目标行为诊断训练状态
│   ├── label-ceiling-diagnosis.md    # 插步：模型平台期先测标注一致性上限
│   └── next-config-recommendation.md # 第三步：给出潜在的更好训练逻辑/配置
├── experiments/             # 实验记录规范：本仓库只跟踪 TEMPLATE.md 模板
│   └── TEMPLATE.md          # 单条实验记录模板（记录与 INDEX.md 归任务仓库，绝不进本仓库 git）
├── README.md                # 人类入口：插件安装与 git 安装（.agents/mtskills）、更新与发版纪律
├── skills/                  # 分发给训练仓库的技能（一技能一目录：SKILL.md + 脚本）
│   ├── init-env/            # ① 初始化环境：建任务仓库工作区的 runtime 目录 + 查询并持久化 GPU 状态
│   ├── data-mgmt/           # ② 数据管理：数据根目录与登记表（runtime/data.json）+ 分任务数据格式规范
│   ├── exp-mgmt/            # ③ 实验记录管理：全局唯一编号（EXP-001 起递增）+ planned→running→done/interrupted(+cancelled) 三时点状态机 + git 自动提交 + INDEX 自动生成 + 编号/lease 校验 + 旧格式迁移（new/start/finish/cancel/list/check/migrate）
│   ├── self-update/         # ④ 自更新：git 拉取最新主分支（只快进；runtime 外改动拒绝更新；仅服务 git 安装副本）
│   └── badcase-analysis/   # ⑤ bad case 定位范式+错误案例分析包：定位分层→溯源→重建→遴选→渲染→打包
├── hooks/                   # 插件 hook：SessionStart 注入方法论红线摘要（可视化红线/实验记录三时点（git 管理）/lease 三确认/训练后台分离）
├── templates/runtime/       # 任务仓库 <工作区>/runtime/ 的协议模板（README + 各 *.example.json，init-env 初始化时拷入）
├── .claude-plugin/          # 插件与市场清单（仓库即市场；plugin.json 的 version 是唯一事实源，发版纪律见 README.md）
│   ├── plugin.json
│   └── marketplace.json
├── .agents/skills/          # 本仓库自身的开发管理技能（ZCode 等智能体自动发现；不属于分发给训练仓库的技能包）
│   └── git-mgmt/            # 双远端 git 规范：origin=私有 GitLab 默认推送；github=公有发布须用户当次明确指示
└── runtime/                 # 源副本自身的运行时状态；布局与 lease 协议正本是 templates/runtime/README.md（安装副本/任务仓库的运行时状态在任务仓库 <工作区>/runtime/）
```


各资产/方法目录的 README 定义了条目格式与收录范围，新增条目前先读对应 README。技能放 `skills/<name>/`（一技能一目录：SKILL.md 用 frontmatter 的 name/description 定义触发条件与步骤，脚本放同目录），现有技能：**init-env**（初始化环境：**第一件事是任务画像问卷**——目标、医疗影像模态、2D/2.5D/3D 建模维度、分割/分类/配准等任务类型，用户必须全部答复才能继续后续步骤；然后查 GPU 静态清单、询问可用卡并记录、检查/按需 uv 初始化 PyTorch+CUDA 环境（默认走阿里 PyPI 镜像，`--index` 可换源），结果写入任务仓库工作区的 `runtime/env.json`）。**data-mgmt**（数据管理：数据统一放 `runtime/data.json` 的 `data_root` 下（默认任务仓库工作区下的 `<工作区>/data/<dataset>/`，用户可改）；三层存储 raw/processed/cache；3D 默认压缩 NRRD、2D 无损 PNG、分割标注 uint8、几何标注世界坐标 JSON；按 seg3d/seg2d/cls2d/cls3d/det3d/det2d/plane3d 规定格式，2.5D 按 3D 存，配准暂缓；scaffold/register/check/preview/du/set-root/import-planes 七个动作 + label_audit.py 标注审计（柱内面积剖面/排名检验/审片图）；dataset.json 含 `annotation` 标注契约字段（produced_by/rule/tolerance/verified，训练前必填））。**exp-mgmt**（实验记录管理：规划时分配任务仓库内全局唯一编号 EXP-001 起递增、建 planned 记录并 git 提交（提交即占号）；安排训练前必须过 start 编号门禁转 running 再写 GPU lease；finish 补全 done/interrupted；cancel 取消且编号不复用；list 从记录自动生成 INDEX.md；check 校验编号唯一/状态合法/INDEX 一致/lease 引用存在且 running；migrate 迁移旧格式 EXP-YYYYMMDD-序号-slug 记录（重编号+补齐时间字段+改写 versions.json 引用+悬空引用报告）；记录必须提交进任务仓库 git，绝不纳入本仓库 git）。**self-update**（自更新：通过 git 拉取本仓库最新主分支，更新脚本所在的当前副本——源副本或训练仓库 `.agents/mtskills` 的安装副本均可；只快进、runtime/ 外本地改动拒绝更新、runtime/ 内自动 stash→pop；插件安装模式的更新由各平台插件管理器完成，本技能只服务于 git 安装副本；安装方式见 README.md）。**badcase-analysis**（错误案例分析与标注质量工程，内含项目无关的 **bad case 定位范式**：
从 bad case 入手 → 定位问题层级（标注内容/标注管线/预处理/模型/评测）→ 具体问题具体分析；
八步闭环 = 单例取证假设判别 → 独立参照的绝对位置审计 → 全量扫描+富集分析 → 上游元数据
溯源 → 管线重建而非打补丁 → 双通道独立验证 → 保守分层治理 → 修复与重训批准点隔离；
读任务仓库的逐例预测与分桶统计，含「锚点直接放在标注切片上」的校验，按五类错误自动
遴选典型案例（每类 3-5 例），在原始影像上渲染标注与预测的对比图，组装「错误分析文档 +
案例原始 nrrd + 标注表」包并打 tar.gz（默认交付）；**开始分析前必须先查阅任务仓库
自身的分析要求**——不同仓库需要不同的分析角度、关注维度与判定口径，这些信息只在
任务仓库中（其 AGENTS.md/README/实验记录/既有分析包），有规定则融合/从其规定，
通用范式与质量纪律是底线；流程与质量纪律任务无关，逐例推理/渲染/溯源重建
工具在任务仓库（参考实现 spine_locator，含全量解剖审计与标注映射重建脚本）。
`skills/` 之外，本仓库自身的开发管理技能放 `.agents/skills/`（智能体自动发现路径）：**git-mgmt**（双远端布局——**origin=私有 GitLab，默认推送目标**；github=公有 GitHub 开源发布远端，仅当用户**当次明确指示**才允许推送，脚本层还需 `--publish` 双确认；未指明远端时一律推 origin，可以提醒用户"是否发布到 GitHub"但提醒不等于授权；含 conventional commits 中文提交规范、安装副本只拉不推、发布前泄露面扫描；status/push/add-github/check/validate 五个动作，validate 负责发布前插件清单与收录边界校验）。


## 医疗影像可视化红线（强制，对所有技能、方法与实验记录生效）

- **智能体不得把自己的影像视觉判断作为证据或决策依据。** 对医疗影像及其渲染图、切片截图"看图下结论"——判断解剖结构、确认标注或病灶位置、判定对错——一律无效；
- 智能体可以**生成**可视化产物（对比图、审片图等），但生成后必须**交由用户查看并反馈结论**；用户的审片意见是唯一的视觉证据来源；
- 对错判定只允许来自两类渠道：**程序化测量**（分割掩码统计、几何计算、指标数值）与**用户审片反馈**；
- 报告与实验记录中涉及影像的证据，只能写程序化信号与用户审片结论，不得写智能体的视觉印象（如"图中可见××"）。

## 核心方法论（一次训练调优的完整回路）

1. **选基础模型**（methods/base-model-selection.md）：根据任务画像（模态、2D/2.5D/3D、数据量级、目标结构尺度、显存预算；画像由 init-env 问卷采集，见任务仓库 `<工作区>/runtime/env.json` 的 `task_profile`）从 assets/models 中选出主选 + 备选，写明理由。
2. **训练并诊断**（methods/training-state-diagnosis.md）：按 `templates/runtime/README.md` 的 lease 协议（runtime 状态落任务仓库 `<工作区>/runtime/training/`）登记 GPU lease 后，在任务仓库**以后台分离方式启动训练**（`setsid`/`nohup`/tmux 做 OS 级分离——训练是长任务而智能体会话无法长期驻留，绝不能作为会话的前台子进程，见协议模板「训练进程必须后台分离运行」节）；依据训练/验证曲线，**特别是不同损失函数与目标函数下的行为差异**（含不可导指标与可导目标是否背离），判断当前训练状态（欠拟合/过拟合/lr 失当/损失失衡/收敛平台等）。
3. **推荐下一轮**（methods/next-config-recommendation.md）：将诊断结果映射为具体的训练逻辑/配置变更（变更优先级：损失与超参 → 预处理 → 模型），并给出验证变更是否生效的预期信号。
4. **记录实验（三时点）**：规划时经 exp-mgmt `new` 分配全局唯一编号建 `planned` 记录；安排训练经 `start` 门禁转 `running`；结束时 `finish` 补全（`done` / `interrupted`，取消用 `cancel`），规范见下；然后回到第 1 步循环。

> **前置契约**：训练开始前必须具备 (a) 数据集登记中的 `annotation` 标注契约；(b) 实验记录「验收口径」栏。模型进入平台期时先按 `methods/label-ceiling-diagnosis.md` 测标注一致性上限。具名优秀版本更新任务仓库 `<工作区>/runtime/versions.json`（baseline/sota/current）。

## 实验记录规范（强制，三时点，git 管理）

- **凡是经本技能方法论指导的实验，无论在哪个任务仓库执行、成功与否，都必须**有实验记录。记录由技能 **exp-mgmt**（`skills/exp-mgmt/`）管理：每个实验在**规划时**分配任务仓库内**全局唯一编号**（`EXP-<NNN>`，三位零填充从 001 递增，999 后进位四位），复制 `experiments/TEMPLATE.md` 模板建 `EXP-<NNN>-<slug>.md`，存放位置默认 `<工作区>/experiments/`（任务仓库可改）；INDEX.md 由 exp-mgmt 从记录自动生成，不得手工编辑。**记录必须提交进任务仓库 git**（exp-mgmt 每个时点自动提交，路径级 add，绝不动 `-A`）；即使文件放在本仓库路径下，也**绝不纳入本仓库 git**（本仓库只跟踪模板）。没有记录的实验结论不得作为后续调优依据。
- **三时点**：①规划——`new` 分配编号建 `planned` 记录（填任务与数据、本轮配置决策），提交即占号，即使训练还没排上；②安排训练——写 GPU lease 前**必须**先过 `start` 编号门禁（校验编号唯一与 planned 状态）转 `running`，lease 的 `experiment` 字段引用该编号；③结束——`finish` 补全结果/诊断/下一轮决策/产物，`status` 改 `done` 或 `interrupted`（注明原因与已产生的观察）。规划后决定不训用 `cancel`（`cancelled`，编号永久占用不复用，注明理由）。
- **编号纪律**：分配一律走 `new`（扫描取 max+1，提交即占号），禁止手工挑号；存在旧格式 `EXP-YYYYMMDD-<序号>-<slug>` 记录时 `new`/`start` 拒绝执行，先用 `migrate` 迁移（重编号 + 改名 + 补齐 v2 时间字段 + 改写 versions.json 引用 + 报告悬空引用）。`check` 兜底校验：编号唯一、状态与时间字段合法、INDEX 一致、lease 引用存在且 running。
- 记录须包含：任务与数据、模型选择理由、预处理与目标配置、结果曲线特征、诊断与证据、下一轮决策、产物（checkpoint/log）在任务仓库中的位置。

## 资源调度（文件系统，协议模板见 templates/runtime/README.md）

- 目的：给所有智能体提供"哪些任务正在训练、占用哪张卡多少显存"的上下文，避免撞卡。哪些卡归本训练目录可用，以任务仓库 `<工作区>/runtime/env.json`（初始化时用户答复的记录）为准。
- 协议三步：**训练前**读任务仓库 `<工作区>/runtime/training/` 全部 lease 并用 `nvidia-smi` 实测（快照不能替代实测），三确认（env.json 可用性 + 无冲突 lease + 实测空闲）都满足才启动；**启动时**训练进程必须后台分离启动（`setsid`/`nohup`/tmux 脱离智能体会话，会话终止不得连带杀掉训练；lease 的 pid 记分离后真实训练进程，日志重定向到文件），并原子写一个 lease JSON（任务、GPU、pid、起止时间、日志位置）到 `<工作区>/runtime/training/`；**结束时**立即删除自己的 lease。
- lease 是声明式参考，不是硬锁；真实占用以 `nvidia-smi` 为准。stale lease（进程已不存在）核对 pid 后可直接清理。只有训练和长时间占 GPU 的任务需要登记，短时推理不必。

## 资产收录边界

- assets/ 只收**任务无关**的内容：讲机制、适用场景、参数经验，不绑定具体数据集/器官/项目。任务特定的经验写进对应实验记录，不进资产卡。
- **每条资产必须有论文支撑**：用简体中文按"统一四问"注明——哪篇论文用了它、目的、效果、结论（每项一两句话，写简单；条目含代码时同样写进代码注释）。格式见各资产目录 README。无论文支撑的先放各 README 的"补录起点"，不算正式条目。
- 收录新资产时与实验记录互加反链，保证结论可溯源。

## 环境事实（未来 agent 必须知道）

- GPU：2× NVIDIA RTX 4090（每卡约 23 GB）。本仓库虽不训练，但推荐配置时必须按此预算约束任务仓库的 batch/模型规模（优先 AMP）。
- Python：系统 python3.12 无 torch。兄弟项目一律用 **uv**（pyproject.toml + uv.lock + 每项目 .venv）；训练环境同样默认 uv，由技能 init-env 检查并按需初始化在**任务仓库工作区**（pyproject + `.venv`，未指定解释器时），实际可用性以任务仓库 `<工作区>/runtime/env.json` 为准（技能 init-env 检查并维护）。
- **Git 布局**：本目录自 2026-10-05 起是**独立 git 仓库**（默认分支 `main`，repo-local 身份 `liuxin <liuxin@local>`）。双远端：**origin = 私有 GitLab**（日常开发默认推送目标）、**github = 公有 GitHub**（开源发布远端，须用户当次明确指示才可推送）——规则与脚本见 `.agents/skills/git-mgmt/`（本仓库自身的开发管理技能，开发本仓库的智能体自动识别；训练仓库安装副本只拉不推）。它物理上位于 `/ssd/lx` 大仓库的工作区内，但外层仓库不追踪本目录——**本目录内的一切 git 操作都作用于本仓库自身**。安装到训练仓库（submodule 或嵌套 clone 到 `.agents/mtskills`）与更新方法见 README.md；技能 **self-update** 负责拉取最新主分支。**插件分发通道**：仓库根的 `.claude-plugin/` 使本仓库自身即一个插件市场（插件名与市场名均为 mtskills），可经 ZCode / Claude Code / Codex 的插件市场机制安装为用户级插件（命令见 README.md「插件安装」），插件模式下的 runtime/experiments 状态同样外置到任务仓库工作区；无论哪种分发方式，推送 GitHub 发布都须用户当次明确指示。外层 `/ssd/lx` 大仓库仍混有大量无关改动，勿在其中 `git add -A`。
- 试验场与素材来源：`/ssd/lx` 下的 BC3-seg、totalsegmentator、spine_locator、eat-model 等任务项目，以及 `net_arch/`（架构）、`metrics/metrics.py`（指标）、既有预处理流水（norm → crop3D → resample 2D）——补录资产时从这里起步。
