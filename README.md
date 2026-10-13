# MTSkills — Medical Model Skill

git 管理的方法论仓库，服务对象是医疗场景深度学习**小模型**的训练效率与收敛精度。**本仓库不承载训练**：训练在任务仓库（`/ssd/lx` 下各子项目）中进行；本仓库提供任务无关的知识资产（模型/预处理/目标函数）、"选模型 → 诊断训练状态 → 推荐下一轮配置"的方法论、技能（init-env / data-mgmt / exp-mgmt / self-update / badcase-analysis）与实验记录规范（EXP-<NNN> 全局编号、三时点、任务仓库 git 管理，由 exp-mgmt 技能兜底）。
本仓库自身的开发管理技能（双远端 git 规范）单独放 `.agents/skills/git-mgmt/`。

**智能体必读 [AGENTS.md](AGENTS.md)**（仓库定位、目录结构、实验记录与资源调度协议）；人类读者可从 `methods/` 入门。

本仓库自 2026-10-05 起是**独立 git 仓库**（默认分支 `main`，源副本位于 `/ssd/lx/mtskills`）。它物理上位于 `/ssd/lx` 大仓库的工作区内，但外层仓库不追踪本目录。

## 插件安装（仓库即市场）

本仓库根目录带 `.claude-plugin/` 清单，**仓库本身即一个插件市场**（市场名与插件名均为 `mtskills`），可经各平台的插件市场机制安装为**用户级插件**：技能（`skills/`）与方法论对所有项目生效，插件目录保持只读，运行时状态（runtime / 实验记录）外置到各任务仓库工作区。

| 平台 | 安装方式 |
|---|---|
| ZCode | Settings → Plugin Marketplace → Add，粘贴本仓库 Git URL（`https://gitlab.suoyin.info/liuxin/mtskills.git`）、GitHub 仓库或本地目录（如 `/ssd/lx/mtskills`） |
| Claude Code | `/plugin marketplace add https://gitlab.suoyin.info/liuxin/mtskills.git`，然后 `/plugin install mtskills@mtskills`（`marketplace add` 也接受本地路径） |
| Codex | `codex plugin marketplace add https://gitlab.suoyin.info/liuxin/mtskills.git` |

更新：插件模式由各平台的插件/市场管理器更新（对市场重新 add / update 即拉取最新 main），**不使用 self-update 技能**。

### 与 git 安装（双轨）的关系

| | 插件安装 | git 安装（submodule / 嵌套 clone 到 `.agents/mtskills`） |
|---|---|---|
| 作用范围 | 用户级全局，所有项目共享一套 | 每个训练仓库一份独立副本 |
| 版本 | 跟随市场更新（最新 main） | 副本锁定到某次提交，升级是显式动作 |
| 更新方式 | 插件管理器 | `self-update` 技能或 `git pull --ff-only` |
| 状态归属 | runtime / 实验记录在任务仓库工作区 | 同样在任务仓库工作区（`<工作区>/runtime/` 等） |

适用场景：个人机器上多个任务仓库共用一套方法论 → 插件；训练仓库要锁定方法论文本、随仓库分发且可复现 → git 安装。两者可并存；任务仓库工作区的同名技能会遮蔽插件技能（见 FAQ）。

### 版本纪律

`.claude-plugin/plugin.json` 的 `version` 是唯一事实源（`.claude-plugin/marketplace.json` 条目里的 `version` 随之保持一致）。**发版** = 改 `version` + 打 `vX.Y.Z` tag + 提交 `chore(release): vX.Y.Z`；日常滚动 main 不动 `version`。

## 安装到训练仓库（.agents 目录，保持 git 可更新）

安装后训练仓库内的智能体即可使用本仓库的全部技能与规范。两种方式二选一，都保留 git 链接、都可 `git pull` 更新。

### 方式一：git submodule（推荐——训练仓库记住这个依赖及其版本）

```bash
cd <训练仓库>
git -c protocol.file.allow=always submodule add /ssd/lx/mtskills .agents/mtskills
git commit -m "chore: 安装 mtskills 到 .agents/mtskills（submodule）"
```

说明：`-c protocol.file.allow=always` 仅在**从本地路径**安装时需要（git 对本地路径子模块的安全默认）；从 http(s) 远端安装时省略。

特点：训练仓库记录子模块指针（固定到某次提交，升级是显式动作）；`git status` 能看到副本落后；缺点是 submodule 的命令略繁琐。

### 方式二：嵌套 clone（不动训练仓库的 git 历史，最简单）

```bash
git clone /ssd/lx/mtskills <训练仓库>/.agents/mtskills
echo ".agents/mtskills/" >> <训练仓库>/.gitignore   # 防止误提交嵌套仓库
```

特点：`.agents/mtskills` 是独立 git 仓库，训练仓库完全不知道它；务必加 `.gitignore`，否则 `git add -A` 会把它记成 gitlink、内容全部"消失"。

### 更新已安装副本（拉取最新主分支）

首选**self-update 技能**（安全策略见下）：

```bash
python3 .agents/mtskills/skills/self-update/self_update.py
```

或手动：

```bash
cd .agents/mtskills && git pull --ff-only origin main
```

- submodule 方式更新后，训练仓库里 `git status` 会显示子模块指针前移，需再 commit 一次指针；也可用 `git submodule update --remote .agents/mtskills` 一步到位（同样要 commit 指针）。
- self-update 的安全策略：**只快进（--ff-only）**，绝不产生合并提交；副本中 `runtime/` 以外的本地改动会**拒绝更新**并逐文件列出（先处理再更新）；`runtime/` 内的本地状态自动 stash → 更新 → pop，冲突时明确报告（新布局下运行时状态在任务仓库工作区 `<工作区>/runtime/`，副本内的 `runtime/` 只在源副本/历史副本中还存在，此分支通常为空）。

## 更换远端

安装源不限于本机路径。本仓库推送到内部 GitLab / 任意远端后：

```bash
# 新安装
git clone <远端URL> <训练仓库>/.agents/mtskills
# 已有副本换源
cd <副本> && git remote set-url origin <远端URL>
```

源副本（/ssd/lx/mtskills）推送远端：`git remote add origin <远端URL> && git push -u origin main`。

## 双远端（私有 GitLab 开发 + 公有 GitHub 发布）

本仓库维持两个远端：**origin = 私有 GitLab**（日常开发，未指明时一律推送这里）、**github = 公有 GitHub**（开源发布，仅当用户明确指示才推送，且每次推送都需当次确认）。规则、泄露面扫描与脚本见 `.agents/skills/git-mgmt/`（本仓库自身的开发管理技能，放在智能体自动发现路径下、与分发给训练仓库的 `skills/` 技能包区分开）。配置 github 远端：

```bash
python3 .agents/skills/git-mgmt/git_mgmt.py add-github --url https://github.com/<user>/<repo>.git
```

## FAQ

- **运行时状态归任务仓库，不随插件/副本走**：env.json（任务画像/GPU/训练环境）、data.json（数据登记）、training/ lease 都由技能写到**任务仓库工作区** `<工作区>/runtime/`（与安装方式无关；环境变量 `MTSKILLS_RUNTIME` 可显式指定 runtime 目录），是机器本地状态。在某个任务仓库做的初始化与数据登记只对该任务仓库生效；要沉淀为方法论默认，回源副本改资产/方法/技能并提交。实验记录（EXP-*）同样归任务仓库：存 <工作区>/experiments/ 并**提交进任务仓库 git**，由技能 exp-mgmt 管理。
- **插件技能的命名空间**：插件安装后，技能以 `mtskills:技能名` 命名空间出现（如 `mtskills:init-env`、`mtskills:data-mgmt`）。
- **插件技能的优先级最低**：任务仓库工作区的同名技能会遮蔽插件技能——需要在某仓库定制行为时，把技能复制进工作区改即可覆盖。
- **嵌套 clone 方式忘了加 .gitignore 怎么办**：若已 `git add` 成 gitlink，在训练仓库 `git rm --cached .agents/mtskills` 再补 `.gitignore` 即可。
- **submodule 更新后训练仓库出现 "modified: .agents/mtskills (new commits)"**：正常，`git add .agents/mtskills && git commit` 固定指针即可。
- **克隆来的副本里 runtime/env.json 是源机器的画像**：在新机器上应重新运行 init-env 技能初始化，不要沿用别机的 GPU/环境结论。

