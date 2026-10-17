# MedPowers 安装手册（AI Agent 执行版）

**本文档是 medpowers 安装的唯一事实源**，供各类 AI 智能体（ZCode / Claude Code / Codex / Cursor / Trae 等）拉取后**自动完成安装**。把下面这行提示词发给任意 AI 助手，它应能仅凭本文档完成安装、验证与收尾汇报：

```
帮我安装 medpowers：https://gitlab.suoyin.info/liuxin/medpowers/-/raw/main/INSTALL.md
```

人类读者亦可照读；背景与全景见 [README.md](README.md)。若上面的 raw URL 因私有仓库鉴权不可达：先 `git clone https://gitlab.suoyin.info/liuxin/medpowers.git`，再直接读仓库根的本文档。

- **适用版本**：v0.11.0 及以上（版本事实源：`.claude-plugin/plugin.json` 的 `version`）
- **上游仓库**：`https://gitlab.suoyin.info/liuxin/medpowers.git`（私有 GitLab）；本机源副本：`/ssd/lx/medpowers`

---

## 0. 执行须知（智能体先读）

1. **按序执行，逐步验证**：每条路线都有「验证点」，验证不过先按 §9 故障排查处理，再继续。
2. **不确定就问**：安装模式（§2）与目标训练仓库路径无法从上下文确定时，向用户提问，不要猜。
3. **只做安装**：不要 push 任何远端、不要修改 medpowers 仓库内容、不要覆盖既有目录；每步动作对用户可见、可回退。唯一例外是 §3 的旧版清理——删除类动作，必须先报告并经用户确认。
4. **收尾汇报**：完成后向用户报告——安装模式、安装位置、旧版清理结果（如有）、验证结果、后续如何更新（§8）。

## 1. medpowers 是什么

医疗影像**小模型**训练方法论技能库：任务无关的知识资产（模型 / 预处理 / 增强 / 目标函数，41 张论文支撑的资产卡）+ 方法论（选基础模型 → 诊断训练状态 → 推荐下一轮配置）+ 7 个智能体技能（init-env / data-mgmt / exp-mgmt / self-update / badcase-analysis / seg-error-topology / notify）+ 实验记录规范（EXP-NNN 三时点）。

它**不跑训练**：训练在用户的任务仓库里；运行时状态（环境画像、数据登记、GPU lease、实验记录）也写在任务仓库工作区，绝不写回安装目录。

## 2. 第一步：确定安装模式

| | **A 插件安装** | **B git 安装** |
|---|---|---|
| 作用范围 | 用户级，所有项目共享一套 | 单个训练仓库一份独立副本 |
| 版本策略 | 跟随市场（最新 main） | 锁定到某次提交，升级是显式动作 |
| 更新方式 | 平台插件管理器 | `self-update` 技能或 `git pull --ff-only` |
| 适用 | 个人机多仓库共用方法论 | 训练仓库要锁定方法论文本、随仓库分发可复现 |

两种模式可并存；任务仓库工作区的同名技能会**遮蔽**插件技能（要定制行为时把技能复制进工作区改即可覆盖）。

**判断规则**：用户明确说了模式 → 照办；只说「装到某个训练仓库 / 这个项目」→ 模式 B；只说「装一下 medpowers」而无上下文 → 向用户复述上表并让其选择。

## 3. 旧版（mtskills）检查与清理

本插件原名 **mtskills**，2026-10-15 起更名为 **medpowers**（GitLab 路径同步迁移为 `https://gitlab.suoyin.info/liuxin/medpowers.git`，旧路径自动重定向）。曾安装旧版的机器上新旧两套会**并存**（技能前缀 `mtskills:` 与 `medpowers:` 重复加载；旧副本脚本读旧环境变量、行为停在旧版）。安装 medpowers 前先按下表检查三个遗留面；**全部未命中 → 无旧版，直接进 §4**。

| 遗留面 | 检查 | 命中后清理（先报告用户，确认后执行） |
|---|---|---|
| 插件市场与插件（模式 A 遗留） | 平台插件列表中有无市场/插件 `mtskills`（旧 URL `https://gitlab.suoyin.info/liuxin/mtskills.git`）；本机缓存 `test -d ~/.zcode/cli/plugins/cache/mtskills` | 在平台插件管理器中移除旧市场与旧插件（ZCode：Settings → Plugin Marketplace；Claude Code / Codex 用各自的 plugin 命令）；缓存目录 `~/.zcode/cli/plugins/cache/mtskills/` 可一并删除 |
| 训练仓库旧副本（模式 B 遗留） | 目标训练仓库内 `test -d .agents/mtskills` | 嵌套 clone：`rm -rf .agents/mtskills`，并删去 `.gitignore` 中的 `.agents/mtskills/` 行；submodule：`git submodule deinit .agents/mtskills && git rm .agents/mtskills && git commit`；之后按 §5 全新安装 `.agents/medpowers` |
| shell 环境变量（两种模式都查） | `grep -n "MTSKILLS_" ~/.bashrc ~/.zshrc ~/.profile 2>/dev/null` | 环境变量已随更名切换且新版本不读旧名：把导出行 `MTSKILLS_RUNTIME` / `MTSKILLS_EXPERIMENTS` 改名为 `MEDPOWERS_RUNTIME` / `MEDPOWERS_EXPERIMENTS` |

执行纪律：

- **先报告再删除**：向用户逐项列出发现的旧版遗留，确认后再清理；用户要求保留时照办（只提示并存会重复加载两套技能）。
- **绝不波及用户数据**：清理只针对旧安装目录本身——任务仓库工作区的 `runtime/` 与 `experiments/` 是用户数据且新旧版本共用，原样保留。

## 4. 路线 A：插件安装（用户级）

> 插件安装后技能以 `medpowers:技能名` 命名空间出现（如 `medpowers:init-env`）。安装与更新都由平台插件管理器负责，**不要**运行 self-update 技能。

分平台执行（各平台智能体可代办的程度不同，注意区分「谁来操作」）：

| 平台 | 谁来操作 | 步骤 |
|---|---|---|
| **ZCode** | 用户（图形界面） | 告知用户：Settings → Plugin Marketplace → Add，粘贴 `https://gitlab.suoyin.info/liuxin/medpowers.git`（或本地路径 `/ssd/lx/medpowers`） |
| **Claude Code** | 用户（会话命令） | 把两行命令给用户执行：`/plugin marketplace add https://gitlab.suoyin.info/liuxin/medpowers.git` → `/plugin install medpowers@medpowers` |
| **Codex** | 智能体（shell） | `codex plugin marketplace add https://gitlab.suoyin.info/liuxin/medpowers.git`，随后按 `codex plugin --help` 的实际子命令安装 medpowers |

**验证点 A**：让用户重启会话；新会话的可用技能列表中出现 `medpowers:` 前缀技能即成功，且**不应**再出现 `mtskills:` 前缀技能（若仍出现，说明旧插件未清理，回 §3 处理）。智能体无法自行重启会话，此步以用户确认为准。

## 5. 路线 B：git 安装（装进训练仓库 `.agents/medpowers`）

前置确认：目标训练仓库的绝对路径（用户未给则问）。源地址二选一：本机存在 `/ssd/lx/medpowers` 时可用本地路径；否则用 GitLab URL。

### 方式 B1：git submodule（推荐——训练仓库记住依赖及版本）

```bash
cd <训练仓库>
git -c protocol.file.allow=always submodule add <URL 或 /ssd/lx/medpowers> .agents/medpowers
git commit -m "chore: 安装 medpowers 到 .agents/medpowers（submodule）"
```

- `-c protocol.file.allow=always` 仅**从本地路径**安装时需要（git 对本地路径子模块的安全默认）；http(s) 源省略。
- **验证点 B1**：`test -f .agents/medpowers/skills/init-env/SKILL.md` 为真；训练仓库 `git status` 干净（指针已提交）。
- 特性：训练仓库记录子模块指针（锁定版本），副本落后时 `git status` 可见。

### 方式 B2：嵌套 clone（不动训练仓库 git 历史，最简单）

```bash
git clone <URL 或 /ssd/lx/medpowers> <训练仓库>/.agents/medpowers
cd <训练仓库>
grep -qxF ".agents/medpowers/" .gitignore 2>/dev/null || echo ".agents/medpowers/" >> .gitignore
git add .gitignore && git commit -m "chore: 忽略嵌套仓库 .agents/medpowers"   # 若 .gitignore 被跟踪
```

**验证点 B2**：同 B1 的文件存在性检查；`.gitignore` 已含该行。
**必须加 `.gitignore`**：否则训练仓库 `git add -A` 会把副本记成 gitlink、内容全部"消失"。

## 6. 安装后验证（两种模式通用清单）

```bash
cd <安装根>   # 模式 A：平台插件缓存目录（如 ~/.zcode/cli/plugins/cache/medpowers/medpowers/<版本>/，因平台而异）
              # 模式 B：<训练仓库>/.agents/medpowers
for f in AGENTS.md README.md INSTALL.md experiments/TEMPLATE.md pyproject.toml \
         medpowers/__init__.py medpowers/session.py medpowers/records.py \
         medpowers/notify.py medpowers/lease.py medpowers/exp_cli.py medpowers/cli.py \
         skills/init-env/SKILL.md skills/data-mgmt/SKILL.md skills/exp-mgmt/SKILL.md \
         skills/self-update/SKILL.md skills/badcase-analysis/SKILL.md \
         skills/seg-error-topology/SKILL.md skills/notify/SKILL.md \
         methods/base-model-selection.md methods/training-state-diagnosis.md \
         methods/next-config-recommendation.md methods/label-ceiling-diagnosis.md; do
  test -f "$f" || echo "缺失: $f"
done && echo "验证通过"
```

输出 `验证通过` 且无「缺失」行 → 安装完成。

## 6.5 Python 包安装（插件主导；训练框架集成的 prerequisite）

安装根同时是 Python 包源（插件缓存为整仓完整复制，包随插件分发）。init-env 在训练环境就绪后执行：

```bash
cd <任务仓库工作区>
uv add <安装根>          # 即上面的 <安装根>：插件缓存目录（模式 A）或 .agents/medpowers（模式 B）
uv run python -c "from medpowers import training_session; print('ok')"   # 冒烟
```

以 path 依赖写进任务仓库 pyproject，版本与插件严格一致。注意：**多机共享的任务仓库**用模式 B 的相对路径源（`.agents/medpowers`，跨机可复现）；模式 A 的插件缓存绝对路径仅本机有效，多机仓库请在各机各自 `uv add` 或改用模式 B。训练入口集成见 `skills/notify/SKILL.md` 与 `medpowers/session.py` 的 docstring（`with training_session("EXP-001", gpus=[0,1], ...) as s:` 包住训练主流程）。

## 7. 安装后必转达用户（智能体不要省略）

1. **第一次使用**：对新任务仓库先运行 init-env 技能——第一件事是**任务画像问卷**（目标、模态、2D/2.5D/3D、任务类型），用户必须全部答复才能继续后续步骤。
2. **运行时状态外置**：runtime（env.json / data.json / training lease / notify.json）与实验记录（`<工作区>/experiments/`，EXP-NNN）都写在**任务仓库工作区**，绝不写回安装目录、绝不提交进 medpowers 仓库。
3. **可视化红线**（全文见安装根 AGENTS.md）：智能体不得把自己的影像视觉判断作为证据；对错判定只来自**程序化测量**与**用户审片反馈**。
4. **实验记录纪律**：凡经本方法论指导的实验（无论成败）必须三时点记录——`new` 占号 → 训练进程经 `medpowers.training_session` 自己 start/finish（门禁失败训练不开始；指标/产物自动入账；CLI 的 start/finish 仅为训练代码未集成包时的遗留兜底）→ `cancel` 取消；多机同步强制「拉取 → 确认 → 提交并推送」（训练进程经包执行的动作同一套实现）。

## 8. 更新

- **模式 A**：平台插件管理器更新（对市场重新 add / update 即拉最新 main）。**不要**用 self-update。
- **模式 B**：`python3 .agents/medpowers/skills/self-update/self_update.py`（只快进 `--ff-only`；`runtime/` 外本地改动会拒绝更新并逐文件列出）或手动 `cd .agents/medpowers && git pull --ff-only origin main`。
- submodule 方式更新后，训练仓库需再 commit 一次指针（`git add .agents/medpowers && git commit`）。

## 9. 故障排查速查

| 症状 | 处置 |
|---|---|
| 嵌套 clone 后副本被 `git add -A` 记成 gitlink | 训练仓库 `git rm --cached .agents/medpowers`，补 `.gitignore` 后重新 add |
| submodule 更新后 `modified: .agents/medpowers (new commits)` | 正常现象；`git add .agents/medpowers && git commit` 固定指针 |
| 克隆副本的 runtime/env.json 是别机的画像 | 新机器重跑 init-env，不要沿用他机 GPU/环境结论 |
| raw URL 拉不到（私有仓库鉴权） | `git clone` 上游仓库后读本地 INSTALL.md |
| 插件安装后看不到技能 | 重启会话；确认技能带 `medpowers:` 前缀；检查平台插件列表中 medpowers 是否启用 |
| 技能列表同时出现 `mtskills:` 与 `medpowers:` 两套前缀 | 旧版插件未清理：按 §3 移除旧市场/旧插件后再重启会话 |

## 10. 卸载

- **模式 A**：平台插件管理器中移除 medpowers。
- **模式 B（submodule）**：`git submodule deinit .agents/medpowers && git rm .agents/medpowers && git commit`。
- **模式 B（嵌套 clone）**：`rm -rf .agents/medpowers`，并删去 `.gitignore` 中对应行。
- 卸载**不影响**任务仓库工作区的 `runtime/` 与 `experiments/`——那是用户数据，不要删。
- 若要卸载的是**旧版 mtskills**（更名前安装），处置方法同上，目录/市场名换成 `mtskills`；安装 medpowers 时的统一清理流程见 §3。
