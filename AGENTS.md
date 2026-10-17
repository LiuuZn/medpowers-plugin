# MedPowers — 医疗影像小模型训练方法论技能库

## 仓库定位

git 管理的方法论仓库（名称 **MedPowers**），服务对象是医疗场景深度学习**小模型**的训练效率与收敛精度。**本仓库不承载训练**：训练在任务仓库（`/ssd/lx` 下各子项目）中进行；本仓库提供四类任务无关的知识资产、一套"选模型 → 诊断训练状态 → 推荐下一轮配置"的方法论，以及实验记录规范（记录本身由任务仓库存放与管理，绝不纳入本仓库 git）。本仓库可经插件市场机制安装为用户级插件（见 README.md「插件安装」），也可 submodule / 嵌套 clone 进训练仓库。仓库根同时是一个 **Python 包源**（`pyproject.toml` + `medpowers/` 包：实验记录生命周期、GPU lease、训练通知的进程内实现）——插件缓存是整仓完整复制，包随插件分发，由技能主导装进任务仓库训练环境（`uv add <副本根>`，版本与插件一致，见技能 init-env）；训练框架 `from medpowers import training_session` 内嵌使用，实验记录的 start/finish 与三类训练通知由**训练进程自己**经包完成，而非智能体。

## 目录结构

```
medpowers/
├── assets/                  # 任务无关的知识资产（按类型分五类）
│   ├── models/              # ① 模型架构及其论文（模型卡）
│   ├── preprocessing/       # ② 预处理方法（本仓库"数据处理"专指预处理）
│   ├── augmentation/        # ③ 数据增强方法（增强卡；训练时在线随机变换，与训练前一次性预处理区分；不配代码）
│   ├── objectives/          # ④ 损失函数 / 目标函数 / 不可导评价指标
│   └── recipes/             # ⑤ 完整训练配方（优化器/LR 日程/损失组装/采样/AMP 的精确默认值与组装顺序，一手源码考证；把 ①-④ 的单项组装成可开工整体）
├── methods/                 # 方法论（按流程顺序）
│   ├── base-model-selection.md       # 第一步：如何选择基础模型
│   ├── training-state-diagnosis.md   # 第二步：结合损失/目标行为诊断训练状态
│   ├── label-ceiling-diagnosis.md    # 插步：模型平台期先测标注一致性上限
│   └── next-config-recommendation.md # 第三步：给出潜在的更好训练逻辑/配置
├── experiments/             # 实验记录规范：本仓库只跟踪 TEMPLATE.md 模板
│   └── TEMPLATE.md          # 单条实验记录模板（记录与 INDEX.md 归任务仓库，绝不进本仓库 git）
├── medpowers/               # Python 包：训练运行时客户端（实验记录生命周期 + GPU lease + 训练通知的进程内实现，随插件整仓分发）
│   ├── notify.py            # 通知（尽力而为，绝不影响训练）：training_lifecycle / send_notification
│   ├── records.py           # 实验记录核心 + start_experiment / finish_experiment 库 API（门禁语义：失败即停）
│   ├── lease.py             # GPU lease 原子写/读/删（机器正名 = notify.json 的 machine_name）
│   ├── session.py           # training_session 全接管会话：门禁→lease→start 通知→训练→finish 落账→删 lease→通知
│   ├── exp_cli.py           # agent 侧实验记录 CLI（new/cancel/list/check/migrate；start/finish 遗留兜底）
│   ├── cli.py / __main__.py # 统一 CLI 入口（medpowers notify … / medpowers exp …）
│   └── tests/               # 包测试（notify/records/lease/session；python3 -m unittest discover -s medpowers/tests -t .）
├── pyproject.toml           # 包定义：hatchling，版本动态读 .claude-plugin/plugin.json（单一事实源）；任务仓库经 uv add <副本根> 安装
├── README.md                # 人类入口：插件安装与 git 安装（.agents/medpowers）、更新与发版纪律
├── INSTALL.md               # 安装手册（AI Agent 执行版）：两种安装模式的事实源，README「通过 AI Agent 安装」提示词指向它
├── skills/                  # 分发给训练仓库的技能（一技能一目录：SKILL.md + 脚本）
│   ├── init-env/            # ① 初始化环境：建任务仓库工作区的 runtime 目录 + 查询并持久化 GPU 状态
│   ├── data-mgmt/           # ② 数据管理：数据根目录与登记表（runtime/data.json）+ 分任务数据格式规范
│   ├── exp-mgmt/            # ③ 实验记录管理（实现在 medpowers 包，本目录脚本为薄启动器）：全局唯一编号（EXP-001 起递增）+ planned→running→done/interrupted(+cancelled) 三时点状态机 + git 自动提交与推送（拉取→确认→提交→推送，多机防撞号防漏记）+ INDEX 自动生成 + 编号/lease 校验 + 旧格式迁移（new/cancel/list/check/migrate 为常备动作；start/finish 为遗留兜底——正路是训练框架经 medpowers.training_session 在训练进程内完成）
│   ├── self-update/         # ④ 自更新：git 拉取最新主分支（只快进；runtime 外改动拒绝更新；仅服务 git 安装副本）
│   ├── badcase-analysis/   # ⑤ bad case 定位范式+错误案例分析包：定位分层→溯源→重建→遴选→渲染→打包
│   ├── seg-error-topology/ # ⑥ 分割误差空间分布画像：FP/FN 按 GT 归一化相对坐标沿各轴分箱（端部/偏向/外溢/局部漏检率）
│   └── notify/             # ⑦ 训练通知组件（实现在 medpowers 包，本目录脚本为薄启动器）：start/finish/interrupt 信号由训练代码内嵌发送（training_session / training_lifecycle），经 webhook（至少飞书，可选加签）推送；智能体仅发 cancel 与 stale lease 兜底；init-env 问卷配置，用户不告知则不生效
├── hooks/                   # 插件 hook：SessionStart 注入方法论红线摘要（可视化红线/实验记录三时点（git 管理）/lease 三确认/训练后台分离）+ 自查插件下载日期（安装根目录 mtime），超一周未更新时注入"每次回复末尾友好提醒检查更新"的会话级指令（尽力而为，任何取值失败静默跳过）
├── templates/runtime/       # 任务仓库 <工作区>/runtime/ 的协议模板（README + 各 *.example.json，init-env 初始化时拷入）
├── .claude-plugin/          # 插件与市场清单（仓库即市场；plugin.json 的 version 是唯一事实源，发版纪律见 README.md）
│   ├── plugin.json
│   └── marketplace.json
├── .agents/skills/          # 本仓库自身的开发管理技能（ZCode 等智能体自动发现；不属于分发给训练仓库的技能包）
│   └── git-mgmt/            # 双远端 git 规范：origin=私有 GitLab 默认推送；github=公有发布须用户当次明确指示
└── runtime/                 # 源副本自身的运行时状态；布局与 lease 协议正本是 templates/runtime/README.md（安装副本/任务仓库的运行时状态在任务仓库 <工作区>/runtime/）
```


各资产/方法目录的 README 定义了条目格式与收录范围，新增条目前先读对应 README。技能放 `skills/<name>/`（一技能一目录：SKILL.md 用 frontmatter 的 name/description 定义触发条件与步骤，脚本放同目录），现有技能：**init-env**（初始化环境：**第一件事是任务画像问卷**——目标、医疗影像模态、2D/2.5D/3D 建模维度、分割/分类/配准等任务类型，用户必须全部答复才能继续后续步骤；然后查 GPU 静态清单、询问可用卡并记录、检查/按需 uv 初始化 PyTorch+CUDA 环境（默认走阿里 PyPI 镜像，`--index` 可换源），结果写入任务仓库工作区的 `runtime/env.json`）。**data-mgmt**（数据管理：数据统一放 `runtime/data.json` 的 `data_root` 下（默认任务仓库工作区下的 `<工作区>/data/<dataset>/`，用户可改）；三层存储 raw/processed/cache；3D 默认压缩 NRRD、2D 无损 PNG、分割标注 uint8、几何标注世界坐标 JSON；按 seg3d/seg2d/cls2d/cls3d/det3d/det2d/plane3d 规定格式，2.5D 按 3D 存，配准暂缓；scaffold/register/check/preview/du/set-root/import-planes 七个动作 + label_audit.py 标注审计（柱内面积剖面/排名检验/审片图）；dataset.json 含 `annotation` 标注契约字段（produced_by/rule/tolerance/verified，训练前必填））。**exp-mgmt**（实验记录管理：规划时分配任务仓库内全局唯一编号 EXP-001 起递增、建 planned 记录并 git 提交（提交即占号）；安排训练前必须过 start 编号门禁转 running 再写 GPU lease；finish 补全 done/interrupted；cancel 取消且编号不复用；全部变更动作强制多机同步——拉取远端→在最新台账上确认→提交后立即推送，推送成功才算动作完成，防多机重复/漏记实验；list 从记录自动生成 INDEX.md；check 校验编号唯一/状态合法/INDEX 一致/lease 引用存在且 running；migrate 迁移旧格式 EXP-YYYYMMDD-序号-slug 记录（重编号+补齐时间字段+改写 versions.json 引用+悬空引用报告）；记录必须提交进任务仓库 git，绝不纳入本仓库 git）。**self-update**（自更新：通过 git 拉取本仓库最新主分支，更新脚本所在的当前副本——源副本或训练仓库 `.agents/medpowers` 的安装副本均可；只快进、runtime/ 外本地改动拒绝更新、runtime/ 内自动 stash→pop；插件安装模式的更新由各平台插件管理器完成，本技能只服务于 git 安装副本；安装方式见 README.md）。**badcase-analysis**（错误案例分析与标注质量工程，内含项目无关的 **bad case 定位范式**：
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
**seg-error-topology**（分割误差空间分布画像：把过分割/欠分割体素按 **GT 归一化
相对坐标**沿各物理轴分箱——r∈[-1,1] 为 GT 沿轴范围归一、|r|>1 为轴向外溢——
输出逐例平均与汇总两口径的箱画像、局部漏检率剖面、μ_r（FP/FN 质量加权平均相对
坐标）分布与 dice 相关；回答"误差集中在器官的头/足/背/腹/左/右哪个部位、端部
还是中心带、端部外溢多少"，结论直接喂下一轮配置；前置=对齐 pred/gt 掩码对 +
spacing + 任务仓库几何核验的轴方向语义；纯掩码程序化测量，不涉影像判读；已知
陷阱含**稀疏层面式标注**（Slicer 描层 GT 与稠密预测逐体素比对会把层间体积全部
误计为 FP，先核验轴向占位密度，稀疏时改贴层/层间/越界三口径）。脚本
`skills/seg-error-topology/scripts/error_topography.py`（CLI 配对清单 + 函数级 API
供任务仓库在线推理包装 import）。参考实现 thymic_seg：val 42 例画像
`src/error_topography_val.py`、稀疏标注三口径 `src/error_topography_evalp1.py`）。
**notify**（训练通知组件，实现在 medpowers 包 `medpowers/notify.py`：经 webhook 推送训练生命周期
提醒——**训练开始、任何原因中断、结束/取消**，至少支持飞书机器人（可选加签校验），另有
generic 类型 POST 扁平 JSON；内容含机器名称、GPU 编号、实验编号与操作说明；**start/finish/
interrupt 三个信号由训练进程内嵌包发送**——全接管正路是 `medpowers.training_session`（记录
门禁+lease+通知一体），纯通知场景用 `training_lifecycle` 上下文管理器包住训练主流程（进入发
start、正常走完发 finish、任何异常/SIGTERM/SIGINT 发 interrupt 带原因后训练行为零改变：异常
照常上抛、信号按默认语义终止）；包随插件分发、由 init-env 装进任务仓库训练环境，技能脚本
`skills/notify/notify.py` 是薄启动器；智能体在准备训练入口时确认这层包裹存在、**不代发**这三
个信号；SIGKILL/OOM-killer/断电等进程无法自报的终止由智能体清理 stale lease 时补发 interrupt
兜底，实验取消（训练进程不参与）由智能体在 exp-mgmt cancel 后发 cancel——仅这两处走 CLI
send；配置由 init-env 问卷采集（用户告知机器名称与 webhook 链接，写入
`<工作区>/runtime/notify.json`），**用户不告知则组件不生效**（发送自动跳过 exit 0，不报错
不追问）；set/disable/status/send/test 五动作 + `training_session`/`training_lifecycle`/
`send_notification` 进程内 API（通知路径绝不抛异常）；**两级纪律**——通知是尽力而为（失败
重试 1 次后放弃、短超时，绝不影响训练），而记录门禁（start_experiment）是失败即停；webhook
URL 含敏感 token，不写入实验记录与提交信息）。
`skills/` 之外，本仓库自身的开发管理技能放 `.agents/skills/`（智能体自动发现路径）：**git-mgmt**（双远端布局——**origin=私有 GitLab，默认推送目标**；github=公有 GitHub 开源发布远端，仅当用户**当次明确指示**才允许推送，脚本层还需 `--publish` 双确认；未指明远端时一律推 origin，可以提醒用户"是否发布到 GitHub"但提醒不等于授权；含 conventional commits 中文提交规范、安装副本只拉不推、发布前泄露面扫描；status/push/add-github/check/validate 五个动作，validate 负责发布前插件清单与收录边界校验）。


## 医疗影像可视化红线（强制，对所有技能、方法与实验记录生效）

- **智能体不得把自己的影像视觉判断作为证据或决策依据。** 对医疗影像及其渲染图、切片截图"看图下结论"——判断解剖结构、确认标注或病灶位置、判定对错——一律无效；
- 智能体可以**生成**可视化产物（对比图、审片图等），但生成后必须**交由用户查看并反馈结论**；用户的审片意见是唯一的视觉证据来源；
- 对错判定只允许来自两类渠道：**程序化测量**（分割掩码统计、几何计算、指标数值）与**用户审片反馈**；
- 报告与实验记录中涉及影像的证据，只能写程序化信号与用户审片结论，不得写智能体的视觉印象（如"图中可见××"）。

## 核心方法论（一次训练调优的完整回路）

1. **选基础模型**（methods/base-model-selection.md）：根据任务画像（模态、2D/2.5D/3D、数据量级、目标结构尺度、显存预算；画像由 init-env 问卷采集，见任务仓库 `<工作区>/runtime/env.json` 的 `task_profile`）从 assets/models 中选出主选 + 备选，写明理由。
2. **训练并诊断**（methods/training-state-diagnosis.md）：智能体做训练前三确认（见下「资源调度」），确认训练入口已内嵌 `medpowers.training_session`（全接管：训练进程自己过 start 编号门禁→写 GPU lease→发开始通知→训练→finish 落账→删 lease→发结束/中断通知；runtime 状态落任务仓库 `<工作区>/runtime/training/`），然后在任务仓库**以后台分离方式启动训练**（`setsid`/`nohup`/tmux 做 OS 级分离——训练是长任务而智能体会话无法长期驻留，绝不能作为会话的前台子进程，见协议模板「训练进程必须后台分离运行」节），启动后验证进程存活与日志增长；训练结束后依据训练/验证曲线，**特别是不同损失函数与目标函数下的行为差异**（含不可导指标与可导目标是否背离），判断当前训练状态（欠拟合/过拟合/lr 失当/损失失衡/收敛平台等）。
3. **推荐下一轮**（methods/next-config-recommendation.md）：将诊断结果映射为具体的训练逻辑/配置变更（变更优先级：损失与超参 → 预处理 → 模型），并给出验证变更是否生效的预期信号。
4. **记录实验（三时点）**：规划时经 exp-mgmt `new` 分配全局唯一编号建 `planned` 记录；训练启动与收尾由训练进程经 `medpowers.training_session` 自动完成（start 门禁转 `running`、finish 落账 `done`/`interrupted`，指标/产物/时长自动入账）；取消用 `cancel`（智能体）；训练结束后智能体把诊断与下一轮决策补写进记录，规范见下；然后回到第 1 步循环。

> **前置契约**：训练开始前必须具备 (a) 数据集登记中的 `annotation` 标注契约；(b) 实验记录「验收口径」栏。模型进入平台期时先按 `methods/label-ceiling-diagnosis.md` 测标注一致性上限。具名优秀版本更新任务仓库 `<工作区>/runtime/versions.json`（baseline/sota/current）。

## 实验记录规范（强制，三时点，git 管理）

- **凡是经本技能方法论指导的实验，无论在哪个任务仓库执行、成功与否，都必须**有实验记录。记录由技能 **exp-mgmt**（`skills/exp-mgmt/`）管理：每个实验在**规划时**分配任务仓库内**全局唯一编号**（`EXP-<NNN>`，三位零填充从 001 递增，999 后进位四位），复制 `experiments/TEMPLATE.md` 模板建 `EXP-<NNN>-<slug>.md`，存放位置默认 `<工作区>/experiments/`（任务仓库可改）；INDEX.md 由 exp-mgmt 从记录自动生成，不得手工编辑。**记录必须提交进任务仓库 git**（exp-mgmt 每个时点自动提交，路径级 add，绝不动 `-A`）；即使文件放在本仓库路径下，也**绝不纳入本仓库 git**（本仓库只跟踪模板）。没有记录的实验结论不得作为后续调优依据。
- **三时点**：①规划——`new` 分配编号建 `planned` 记录（填任务与数据、本轮配置决策），提交即占号，即使训练还没排上；②训练启动——**由训练进程经 `medpowers.training_session` 自己完成**：入口处过 `start` 编号门禁（拉取→校验编号唯一与 planned 状态→转 `running`→提交推送；失败自行退出，训练不得开始），随后写自己的 GPU lease（真实 pid，`experiment` 字段引用该编号，`machine`/`gpus` 结构化字段同步入账）——智能体在三确认后直接后台分离启动已内嵌 session 的训练入口；③结束——训练进程 `finish` 落账（`done`/`interrupted`，指标/产物/时长/中断原因自动入账），智能体随后补写诊断与下一轮决策。规划后决定不训用 `cancel`（`cancelled`，编号永久占用不复用，注明理由）。exp-mgmt CLI 的 `start`/`finish` 是训练代码未集成包时的**遗留兜底**。
- **多机同步（拉取-确认-提交，强制）**：任务仓库 git 远端是多台机器唯一的实验台账事实源，严防多机重复/遗漏实验。exp-mgmt 全部变更动作（new/start/finish/cancel/list/migrate）由脚本强制「**拉取**（fetch+快进；分叉、rebase 未完成、远端不可达默认拒绝执行，`--allow-offline` 显式豁免）→ **确认**（编号唯一与状态校验都在拉取后的最新台账上进行，他机已占用编号/已 start 同一实验即拒绝）→ **提交并推送**（提交后立即推送；**推送未成功视为动作未完成**——推送成功前不得写 GPU lease、不得视实验为已登记/已完结）」；无上游远端的单机仓库自动跳过同步；`check` 兜底警告未推送提交与落后远端。**训练进程经 medpowers 包执行的 start/finish 用同一套同步实现，红线完全一致**。**查询/汇报路径同样先同步（强制）**：回答"哪些实验在跑/可以进行/进展如何"这类状态问题前，先拉取任务仓库最新主分支（fetch+快进，只读不提交）——本地台账可能滞后于他机已推送的 running/结论，据以汇报会把别机已启动的实验误报为 planned；lease 与 `nvidia-smi` 只反映本机 GPU，跨机状态以拉取后的台账为准，两者合并才是全项目状态。
- **编号纪律**：分配一律走 `new`（扫描取 max+1，提交即占号），禁止手工挑号；存在旧格式 `EXP-YYYYMMDD-<序号>-<slug>` 记录时 `new`/`start` 拒绝执行，先用 `migrate` 迁移（重编号 + 改名 + 补齐 v2 时间字段 + 改写 versions.json 引用 + 报告悬空引用）。`check` 兜底校验：编号唯一、状态与时间字段合法、INDEX 一致、lease 引用存在且 running。
- 记录须包含：任务与数据、模型选择理由、预处理与目标配置、结果曲线特征、诊断与证据、下一轮决策、产物（checkpoint/log）在任务仓库中的位置。
- **机器指代用配置名，禁称"本机"**：实验记录、GPU lease 与 exp 提交信息是多机共读的共享台账，提到训练在哪台机器执行时一律用任务仓库 `<工作区>/runtime/notify.json` 的 `machine_name` 值称呼（如"机器A 的 GPU 0"），禁止"本机"字样；`machine_name` 未配置时用 `hostname` 输出兜底并建议配置。

## 资源调度（文件系统，协议模板见 templates/runtime/README.md）

- 目的：给所有智能体提供"哪些任务正在训练、占用哪张卡多少显存"的上下文，避免撞卡。哪些卡归本训练目录可用，以任务仓库 `<工作区>/runtime/env.json`（初始化时用户答复的记录）为准。
- 协议三步：**训练前（智能体）**读任务仓库 `<工作区>/runtime/training/` 全部 lease 并用 `nvidia-smi` 实测（快照不能替代实测），三确认（env.json 可用性 + 无冲突 lease + 实测空闲）都满足才启动，并确认训练入口已内嵌 `medpowers.training_session`；**启动时**训练进程必须后台分离启动（`setsid`/`nohup`/tmux 脱离智能体会话，会话终止不得连带杀掉训练，日志重定向到文件），进程入口经包先过 start 编号门禁（失败自行退出），再原子写一个 lease JSON（任务、机器名、GPU、**分离后真实训练进程自身 pid**、起止时间、日志位置；机器名 = `notify.json` 的 `machine_name`，未配置用 hostname）到 `<工作区>/runtime/training/`；**结束时（训练进程，经包）**finish 落账后立即删除自己的 lease。
- lease 是声明式参考，不是硬锁；真实占用以 `nvidia-smi` 为准。stale lease（进程已不存在）核对 pid 后可直接清理。只有训练和长时间占 GPU 的任务需要登记，短时推理不必。
- lease 与 `nvidia-smi` **只反映本机**：哪些实验在整个项目里处于 running，跨机唯一事实源是任务仓库 git 的实验台账（见上「多机同步」）——查询/汇报训练状态前先拉取同步，再读本机 lease 与实测合并汇报。
- 训练开始/结束/异常中断的 webhook 提醒由**训练进程经 medpowers 包内嵌发送**（全接管正路 `training_session`；纯通知场景 `training_lifecycle`；智能体准备训练入口时确认已内嵌；配置在 `<工作区>/runtime/notify.json`，用户未配置则自动跳过；尽力而为，绝不影响训练）；SIGKILL/断电等进程无法自报的终止，由清理 stale lease 的智能体补发 interrupt 兜底；实验取消由智能体在 cancel 后发 cancel 通知。

## 资产收录边界

- assets/ 只收**任务无关**的内容：讲机制、适用场景、参数经验，不绑定具体数据集/器官/项目。任务特定的经验写进对应实验记录，不进资产卡。
- **每条资产必须有论文支撑**：用简体中文按"统一四问"注明——哪篇论文用了它、目的、效果、结论（每项一两句话，写简单；条目含代码时同样写进代码注释）。格式见各资产目录 README。无论文支撑的先放各 README 的"补录起点"，不算正式条目。
- 收录新资产时与实验记录互加反链，保证结论可溯源。

## 环境事实（未来 agent 必须知道）

- GPU：2× NVIDIA RTX 4090（每卡约 23 GB）。本仓库虽不训练，但推荐配置时必须按此预算约束任务仓库的 batch/模型规模（优先 AMP）。
- Python：系统 python3.12 无 torch。兄弟项目一律用 **uv**（pyproject.toml + uv.lock + 每项目 .venv）；训练环境同样默认 uv，由技能 init-env 检查并按需初始化在**任务仓库工作区**（pyproject + `.venv`，未指定解释器时），实际可用性以任务仓库 `<工作区>/runtime/env.json` 为准（技能 init-env 检查并维护）。**medpowers Python 包**随插件整仓分发，init-env 在训练环境就绪后把它装进任务仓库，版本与插件严格一致（pyproject 动态读 plugin.json）。**源必须跨机可解析（硬规则）**：默认仓内相对源（各机备 `.agents/medpowers` 副本后 `uv add ./.agents/medpowers`）或 git tag 依赖；兄弟目录相对路径（`../mtskills`）与插件缓存绝对路径仅写入机有效，不得进多机共享 pyproject——失效源修复流程见 INSTALL.md §6.5。
- **Git 布局**：本目录自 2026-10-05 起是**独立 git 仓库**（默认分支 `main`，repo-local 身份 `liuxin <liuxin@local>`）。双远端：**origin = 私有 GitLab**（日常开发默认推送目标）、**github = 公有 GitHub**（开源发布远端，须用户当次明确指示才可推送）——规则与脚本见 `.agents/skills/git-mgmt/`（本仓库自身的开发管理技能，开发本仓库的智能体自动识别；训练仓库安装副本只拉不推）。它物理上位于 `/ssd/lx` 大仓库的工作区内，但外层仓库不追踪本目录——**本目录内的一切 git 操作都作用于本仓库自身**。安装到训练仓库（submodule 或嵌套 clone 到 `.agents/medpowers`）与更新方法见 README.md；技能 **self-update** 负责拉取最新主分支。**插件分发通道**：仓库根的 `.claude-plugin/` 使本仓库自身即一个插件市场（插件名与市场名均为 medpowers），可经 ZCode / Claude Code / Codex 的插件市场机制安装为用户级插件（命令见 README.md「插件安装」），插件模式下的 runtime/experiments 状态同样外置到任务仓库工作区；无论哪种分发方式，推送 GitHub 发布都须用户当次明确指示。外层 `/ssd/lx` 大仓库仍混有大量无关改动，勿在其中 `git add -A`。
- 试验场与素材来源：`/ssd/lx` 下的 BC3-seg、totalsegmentator、spine_locator、eat-model 等任务项目，以及 `net_arch/`（架构）、`metrics/metrics.py`（指标）、既有预处理流水（norm → crop3D → resample 2D）——补录资产时从这里起步。
