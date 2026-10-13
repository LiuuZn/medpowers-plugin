---
name: exp-mgmt
description: 实验记录全生命周期管理：规划实验时分配任务仓库内全局唯一编号（EXP-001 起递增）并建 planned 记录；安排训练/启动训练前必须先经本技能 start 编号门禁转 running（之后才写 GPU lease）；训练结束/中断经 finish 补全（done/interrupted）；规划取消用 cancel（编号保留不复用）；list 从记录自动生成 INDEX.md；check 校验编号唯一/状态合法/INDEX 一致/lease 引用；migrate 迁移旧格式（EXP-YYYYMMDD-序号-slug）记录。凡规划实验、新建或查询实验记录、安排或结束训练、校验或迁移实验记录时使用。
---

# exp-mgmt — 实验记录管理

实验记录是训练调优回路的台账：**规划时占号**建记录，**安排训练时**转 running 并被 GPU lease 引用，**结束时**补全结论。记录文件（任务仓库 `<工作区>/experiments/EXP-<NNN>-<slug>.md`）是唯一事实源，INDEX.md 由本技能生成、不得手工编辑。**记录必须提交进任务仓库 git**——本技能每个动作自动提交（路径级 add，绝不动 `-A`）；记录绝不纳入 mtskills 本仓库 git（本仓库只跟踪模板）。

## 状态机（三时点 + 取消旁路）

```
planned（规划：new 分配编号、建记录）
   → running（安排训练：start 门禁，再写 lease）
   → done / interrupted（finish 补全）
planned|running → cancelled（cancel：编号永久占用不复用）
```

## 动作（脚本 skills/exp-mgmt/exp_mgmt.py，零第三方依赖）

| 动作 | 命令 | 何时用 |
|---|---|---|
| new | `python3 skills/exp-mgmt/exp_mgmt.py new --dir <记录目录> --slug <短横线小写> [--template <路径>]` | 规划实验（下一轮决策产出即可规划，不必等训练排上），可用 `--template` 换模板。从模板建 planned 记录，提交 `exp(EXP-NNN): plan` |
| start | `... start --dir <记录目录> EXP-001` | **安排训练、写 GPU lease 之前（强制门禁）**：校验编号唯一与 planned 状态，转 running。lease 的 `experiment` 字段引用该编号 |
| finish | `... finish --dir <记录目录> EXP-001 --status done\|interrupted` | 训练结束/中断。先把结果/诊断/下一轮决策/产物写进记录，再调用（内容随本次提交入库） |
| cancel | `... cancel --dir <记录目录> EXP-001 --reason "<理由>"` | 规划后决定不训，或训练中废弃 |
| list | `... list --dir <记录目录>` | 查询实验状态；INDEX.md 有变化时重生成并提交 |
| check | `... check --dir <记录目录> [--runtime <runtime目录>]` | 兜底校验：编号唯一、状态与时间字段一致、INDEX 一致、lease 引用存在且 running；旧格式与未提交变更给警告 |
| migrate | `... migrate --dir <记录目录> [--runtime <runtime目录>]` | 旧格式 `EXP-YYYYMMDD-<序号>-<slug>` 重编号为 `EXP-001..N`、补齐 v2 时间字段（由文件名日期推导）、改写 versions.json 的 exp 引用、报告悬空引用 |

目录定位：`--dir` 显式指定优先；否则环境变量 `MTSKILLS_EXPERIMENTS`；否则 `./experiments`。

## 纪律（强制）

1. **编号门禁**：启动训练前必须 `start` 成功（exit 0）才允许写 lease；存在旧格式记录时 `new`/`start` 拒绝执行，先 `migrate`。
2. **三时点缺一不可**：规划即 `new`（哪怕训练还没排上）；训练启动即 `start`；结束即 `finish`。没有记录的实验结论不得作为调优依据。
3. **内容先填后 finish**：`finish` 只翻状态；结果/诊断/下一轮决策由智能体在调用前写入记录文件，随 finish 提交入库；时点之间补填内容可主动以 `exp(EXP-NNN): update` 提交（或留待下一次脚本动作顺带入库）。
4. **提交由脚本完成**：不得绕过脚本手工 `git add -A`；脚本只对记录目录（migrate 另含 versions.json）做路径级 add。
5. **编号不复用**：cancelled 的编号永久占用；分配一律走 `new`（扫描取 max+1，提交即占号），禁止手工挑号。
6. **lease 与记录的衔接**：`start` 只做记录侧门禁，不写 lease；lease 按协议模板（`templates/runtime/README.md`）由训练启动流程写，顺序 = start 通过 → 写 lease → 三确认 → 后台分离启动。
