---
name: git-mgmt
description: mtskills 仓库自身的开发管理规范（双远端 git 策略），位于 .agents/skills/ 供开发本仓库的智能体自动识别，不随 skills/ 分发给训练仓库。凡在本仓库做 git 提交、推送、远端同步，或用户提到 push / 推送 / 发布 / 开源 / GitHub / GitLab 时使用：origin=私有 GitLab 是默认推送目标；github=公有 GitHub 只有用户当次明确指示才允许推送，绝不默认发布。脚本提供 status / push / add-github / check / validate 五个动作，validate 为插件清单与收录边界的发布前只读校验。
---

# git-mgmt — 本仓库开发管理的双远端 git 规范

**定位**：本技能面向 **mtskills 仓库自身的开发管理**（源副本 `/ssd/lx/mtskills`），放在 `.agents/skills/`（ZCode 等智能体的技能自动发现路径，开发本仓库时自动可识别）而非 `skills/`——`skills/` 是随仓库分发给训练仓库的算法开发技能包（init-env / data-mgmt / self-update），本技能不承担该职责；安装副本（训练仓库 `.agents/mtskills`）只拉不推，也用不到这里的推送能力。

本仓库维持两份 git 远端：私有 GitLab 承载日常开发与调试，公有 GitHub 对应开源发布。脚本自动定位**自己所在副本**的仓库根（与当前工作目录无关）：

```bash
python3 .agents/skills/git-mgmt/git_mgmt.py status            # 双远端概览（地址、ahead/behind、未提交改动）
python3 .agents/skills/git-mgmt/git_mgmt.py push              # 推 origin（默认，安全）
python3 .agents/skills/git-mgmt/git_mgmt.py push --remote github --publish   # 推 GitHub（双确认，见下）
python3 .agents/skills/git-mgmt/git_mgmt.py add-github --url <GitHub仓库URL>  # 首次配置/更换 github 远端
python3 .agents/skills/git-mgmt/git_mgmt.py check             # 发布前泄露面扫描（启发式）
python3 .agents/skills/git-mgmt/git_mgmt.py validate          # 发布前校验：插件清单与收录边界（只读）
```

## 双远端布局

| 远端 | 地址 | 用途 | 推送规则 |
|---|---|---|---|
| `origin` | 私有 GitLab（gitlab.suoyin.info/liuxin/mtskills） | 日常开发与调试 | **未指明远端时一律推这里** |
| `github` | 公有 GitHub（开源仓库） | 对外发布 | 仅当用户**当次明确指示**（远端未配置时先 `add-github`） |

命令纪律：永远显式写远端与分支（`git push origin main`），**不用裸 `git push`**——双远端下裸命令的默认目标不明确；拉取统一走 self-update 技能或 `git pull --ff-only origin main`。

## 推送策略（硬规则）

1. **默认 origin**：用户没说推哪里，就是 `origin`（私有 GitLab），不需要再问。
2. **推 github 须当次明确指示**：用户在当前对话里明确要求发布到 GitHub 才可以推。此前某次同意过、或"反正迟早要开源"的推断，都**不构成授权**；同一会话内一次明确授权（如"这轮改完发到 github"）可覆盖该会话内由此产生的推送，跨会话无效。
3. **提醒允许，提醒 ≠ 授权**：完成一批有价值的更新后可以问一句"是否发布到 GitHub？"；用户未明确答复前不推。
4. **脚本双确认**：`push --remote github` 不带 `--publish` 直接拒绝；这是工具层的最后防线，不得绕过脚本改用裸 git 推 github。
5. **禁止 force push**：脚本不提供任何 force 选项；确需改写远端历史时人工操作，并向用户说明后果。

## 提交规范

- Conventional Commits 前缀 + 简体中文描述，一行主题（与现有历史一致）：`feat:` 新功能、`fix:` 修复、`docs:` 文档、`refactor:` 重构、`perf:` 性能、`test:` 测试、`chore:` 杂项。
- 示例（取自真实历史）：`feat: init-env 的 uv 初始化默认走阿里 PyPI 镜像（--index 可换源）`、`fix: 安装形态提示区分 submodule（superproject 检测）与 .agents 嵌套 clone`。
- 提交前 `git status` 逐项确认要进版本的文件；本仓库不进大数据、模型权重、训练产物（数据归 data_root，产物归任务仓库）。

## 仓库边界

- 本目录内的一切 git 操作只作用于本仓库自身；外层 `/ssd/lx` 大仓库混有大量无关改动，**绝不在其中 `git add -A`**。
- 安装副本（训练仓库 `.agents/mtskills`）**只拉不推**：脚本检测到安装形态即拒绝 push；要改内容回源副本改，经 origin 分发。
- `runtime/` 内 env.json、data.json 等是机器本地状态，按现有约定进 git，但提交前不要把训练 lease、临时产物带进版本。

## 发布前检查（推 github 之前）

先跑 `git_mgmt.py validate`（插件清单与收录边界）和 `git_mgmt.py check`（泄露面扫描），把结果给用户过目，再人工确认：

- assets/ 条目按 CODE_STYLE.md 约定含本地绝对路径（"本地落点写绝对路径"）、experiments/ 记录含任务仓库内部路径与项目名——公开前由用户决定是否接受或清理；
- `runtime/` 机器状态文件（env.json / data.json）是否随仓库公开，由用户决定；
- `git push` 发布的是**全部历史**，不只当前快照，历史中的敏感信息同样可见（check 会列出历史邮箱）。

扫描只是启发式参考；是否公开、公开哪些内容，最终由用户拍板。

## 插件清单与收录边界校验（validate）

`git_mgmt.py validate` 与 status / push / add-github / check 并列：插件模式（`.claude-plugin` 市场）下发布前对清单合法性与收录边界做**只读**逐项校验（不改任何文件、不写 git 状态），任一项 fail 退出码非 0，全部项无论 ok/fail 都逐条列出供审计——

- `.claude-plugin/plugin.json` 与 `.claude-plugin/marketplace.json` 存在且 JSON 合法；
- plugin name 匹配 `^[a-z0-9][a-z0-9._-]{0,127}$`；市场条目 name 与 plugin.json 一致；条目 source 为 `"./"` 且仓库根存在 `skills/`；
- `skills/*/SKILL.md` frontmatter 含 name 与 description、name 与目录名一致、description 长度 ≤1024；
- `git ls-files experiments/` 输出仅 `experiments/TEMPLATE.md`（实验记录归任务仓库，绝不进 git）；
- `templates/runtime/` 存在且 4 个文件齐全（README.md + 3 个 example.json）。

脚本为纯 python 自包含的启发式校验；末尾会提示可手动运行 `claude plugin validate .` 做官方清单校验。

## 各场景行为

| 场景 | 行为 |
|---|---|
| `push` 未指明远端 | 推 `origin` 当前分支 |
| `push --remote github` 缺 `--publish` | 拒绝并说明授权规则 |
| `push --remote github --publish` | 先跑泄露面扫描摘要，再推送 |
| 在安装副本（.agents/mtskills）内 push | 拒绝（安装副本只拉不推） |
| `github` 远端未配置 | 提示 `add-github --url <URL>` |
| `add-github` 远端已存在 | 报告现有 URL，加 `--force` 才更换 |
| `status` 离线 | fetch 失败时注明，用陈旧引用报告 ahead/behind |
