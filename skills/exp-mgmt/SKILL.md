---
name: exp-mgmt
description: 实验记录全生命周期管理（实现在 medpowers Python 包，本目录脚本为薄启动器）：规划实验时分配任务仓库内全局唯一编号（EXP-001 起递增）并建 planned 记录；训练启动与收尾由训练进程经 medpowers.training_session 自己完成（入口过 start 编号门禁转 running、门禁失败训练不开始；结束 finish 落账 done/interrupted、指标/产物自动入账）；本技能 CLI 的 start/finish 是训练代码未集成包时的遗留兜底；规划取消用 cancel（编号保留不复用）；list 从记录自动生成 INDEX.md；check 校验编号唯一/状态合法/INDEX 一致/lease 引用；migrate 迁移旧格式（EXP-YYYYMMDD-序号-slug）记录。多台机器共用任务仓库时，全部变更动作由同一套同步实现强制「拉取→确认→提交→推送」防撞号防漏记。凡规划实验、新建或查询实验记录、安排或结束训练、校验或迁移实验记录、向用户汇报实验/训练状态时使用；汇报状态前必须先拉取同步远端台账——本地副本可能滞后于他机已启动的实验。
---

# exp-mgmt — 实验记录管理

实验记录是训练调优回路的台账：**规划时占号**建记录，**训练启动与收尾由训练进程经 `medpowers.training_session` 自动完成**（转 running、落账 done/interrupted），**结束时**智能体补写诊断与下一轮决策。记录文件（任务仓库 `<工作区>/experiments/EXP-<NNN>-<slug>.md`）是唯一事实源，INDEX.md 由本技能生成、不得手工编辑。**记录必须提交进任务仓库 git 并推送远端**——无论智能体 CLI 还是训练进程经包执行，每个变更动作都走同一套「拉取 → 确认 → 提交 → 推送」（路径级 add，绝不动 `-A`）：先快进拉取远端最新台账，在最新状态上校验编号与状态，提交后立即推送；**推送未成功 = 动作未完成**（多台机器共用任务仓库远端时，这是防重复/漏记实验的根本手段）。分支无上游远端视为单机仓库，自动跳过同步。记录绝不纳入 medpowers 本仓库 git（本仓库只跟踪模板）。

## 状态机（三时点 + 取消旁路，全接管模式）

```
planned（规划：new 分配编号、建记录）
   → running（训练进程经 training_session 过 start 门禁，随后写自己的 lease）
   → done / interrupted（训练进程经 training_session finish 落账，事实自动入账；
                         智能体随后补写诊断/下一轮决策）
planned|running → cancelled（cancel：编号永久占用不复用）
```

## 动作（脚本 skills/exp-mgmt/exp_mgmt.py 是 medpowers 包的薄启动器；包安装后也可 `medpowers exp ...`；零第三方依赖）

| 动作 | 命令 | 何时用 |
|---|---|---|
| new | `python3 skills/exp-mgmt/exp_mgmt.py new --dir <记录目录> --slug <短横线小写> [--template <路径>]` | 规划实验（下一轮决策产出即可规划，不必等训练排上），可用 `--template` 换模板。先拉取远端再分配编号（防多机撞号），从模板建 planned 记录，提交 `exp(EXP-NNN): plan` 后立即推送 |
| start | `... start --dir <记录目录> EXP-001` | **[遗留兜底]** 安排训练门禁：planned → running。推送成功才算过门禁。**正路是训练框架经 `medpowers.training_session` 在训练进程内 start**（门禁失败训练不开始、随后自写 lease）——本命令仅用于训练代码未集成包的过渡期，代跑门禁后智能体还要代写 lease |
| finish | `... finish --dir <记录目录> EXP-001 --status done\|interrupted` | **[遗留兜底]** 训练结束：running → done\|interrupted。先把结果/诊断/下一轮决策/产物写进记录，再调用（内容随本次提交入库）。**正路是训练框架经 training_session 在训练进程内 finish**（指标/产物/时长自动入账，智能体只补诊断/下一轮） |
| cancel | `... cancel --dir <记录目录> EXP-001 --reason "<理由>"` | 规划后决定不训，或训练中废弃。训练进程不参与取消，由智能体随后发 `--event cancel` 通知（纪律 7） |
| list | `... list --dir <记录目录>` | 查询实验状态；INDEX.md 有变化时重生成并提交 |
| check | `... check --dir <记录目录> [--runtime <runtime目录>]` | 兜底校验：编号唯一、状态与时间字段一致、INDEX 一致、lease 引用存在且 running；旧格式与未提交变更给警告 |
| migrate | `... migrate --dir <记录目录> [--runtime <runtime目录>]` | 旧格式 `EXP-YYYYMMDD-<序号>-<slug>` 重编号为 `EXP-001..N`、补齐 v2 时间字段（由文件名日期推导）、改写 versions.json 的 exp 引用、报告悬空引用 |

目录定位：`--dir` 显式指定优先；否则环境变量 `MEDPOWERS_EXPERIMENTS`；否则 `./experiments`。以上全部变更动作（new/start/finish/cancel/list/migrate）均自动「拉取 → 确认 → 提交 → 推送」；`--allow-offline` 是远端不可达时的显式豁免（跳过拉取与推送，撞号风险自担）。

## 纪律（强制）

1. **编号门禁**：启动训练前必须 `start` 成功（exit 0）才允许写 lease；存在旧格式记录时 `new`/`start` 拒绝执行，先 `migrate`。
2. **三时点缺一不可**：规划即 `new`（哪怕训练还没排上）；训练启动即 `start`；结束即 `finish`。没有记录的实验结论不得作为调优依据。
3. **内容先填后 finish**：`finish` 只翻状态；结果/诊断/下一轮决策由智能体在调用前写入记录文件，随 finish 提交入库；时点之间补填内容可主动以 `exp(EXP-NNN): update` 提交（或留待下一次脚本动作顺带入库）。
4. **提交由脚本完成**：不得绕过脚本手工 `git add -A`；脚本只对记录目录（migrate 另含 versions.json）做路径级 add。
5. **编号不复用**：cancelled 的编号永久占用；分配一律走 `new`（扫描取 max+1，提交即占号），禁止手工挑号。
6. **lease 与记录的衔接**：`start` 只做记录侧门禁，不写 lease；lease 按协议模板（`templates/runtime/README.md`）由训练启动流程写，顺序 = start 通过 → 写 lease → 三确认 → 后台分离启动。
7. **通知触点**：训练生命周期通知经 medpowers 包推送 webhook（机器名称/GPU 编号/实验编号/操作说明；未配置时自动跳过，不视为违规）。**start/finish/interrupt 三个信号由训练进程经 `training_session` 内嵌发送**——智能体在三确认后直接后台分离启动已内嵌 session 的训练入口，不再代发这三个信号。智能体仅两处用 CLI send：`cancel` 之后发 `--event cancel`（训练进程不参与取消）；清理 stale lease 时发现进程无法自报地终止（SIGKILL/断电，核对训练日志确认内嵌钩子没来得及发过）补发 `--event interrupt`。通知是尽力而为：失败重试 1 次后放弃，**不得用 `&&` 把 send 与本技能动作串联，更不得因通知失败回滚记录状态**。
8. **多机同步（拉取-确认-提交，强制）**：任务仓库 git 远端是多台机器唯一的实验台账事实源，多机共用同一任务仓库时严防重复/遗漏实验。脚本对全部变更动作强制：**拉取**（fetch + 快进；本地与远端分叉拒绝；rebase 未完成（HEAD 游离）拒绝；远端不可达默认拒绝，`--allow-offline` 显式豁免）→ **确认**（编号唯一与状态校验都在拉取后的最新台账上进行，他机已占用编号/已 start 同一实验即拒绝）→ **提交并推送**（推送被拒 = 动作未完成，按报错内恢复步骤执行：`git pull --rebase` → 冲突取远端侧、new 撞号把自己的记录改名到下一个空闲编号（文件名+一级标题）→ `git add <记录目录>`（须覆盖改名前后两个路径）→ `git rebase --continue` → 重新推送）。无上游远端的仓库自动跳过同步（单机模式）；`check` 兜底警告未推送提交（他机不可见）与落后远端。**查询/汇报路径同样先同步**：回答"哪些实验在跑/可以进行/进展如何"前，先在任务仓库 `git fetch` + 快进拉取（只读，不产生提交）——本地台账可能滞后于他机推送的 running/结论，据以汇报会把别机已启动的实验误报为 planned（lease 与 `nvidia-smi` 只反映本机，跨机以台账为准）。
9. **机器指代用配置名，禁称"本机"**：实验记录（含 status_note / 启动事实 / notes）、GPU lease 与 exp 提交信息中提到训练在哪台机器执行时，一律用任务仓库 `<工作区>/runtime/notify.json` 的 `machine_name` 值称呼该机器（如"机器A 的 GPU 0"），**禁止"本机"字样**——台账是多机共读的，"本机"无法区分是哪台机器写的。`machine_name` 未配置时用 `hostname` 输出兜底，并建议用户经 init-env 问卷或 notify `set` 配置。
