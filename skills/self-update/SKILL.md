---
name: self-update
description: 通过 git 拉取本仓库（medpowers）最新主分支，更新当前副本。当用户要求"更新 medpowers / 更新技能库 / 拉取最新方法论 / 更新 .agents/medpowers"时使用；适用于源副本（/ssd/lx/medpowers）与任何安装副本（训练仓库 .agents/medpowers，submodule 或嵌套 clone 均可）。插件（.claude-plugin 市场）安装模式下更新由插件管理器完成，本技能仅服务于 git 安装副本。
---

# self-update — 拉取最新主分支

脚本自动定位**自己所在副本**的仓库根（与当前工作目录无关），从 origin 拉取主分支并快进：

```bash
python3 skills/self-update/self_update.py          # 在任一副本内
python3 .agents/medpowers/skills/self-update/self_update.py   # 在训练仓库内
```

手动等价：`cd <副本> && git fetch origin && git merge --ff-only origin/main`。

## 安全策略（脚本内置，不得绕过）

1. **只快进**（`--ff-only`）：本地分叉时直接失败并报告，绝不制造合并提交、绝不 rebase。
2. **runtime/ 外的本地改动 → 拒绝更新**：逐文件列出，让用户先提交/丢弃/暂存再运行；防止静默吞掉用户对资产/方法/技能的修改。
3. **runtime/ 内的本地状态 → 自动 stash → 更新 → pop**：env.json / data.json / training lease 等是机器本地状态，与上游更新无冲突时无感保留；pop 冲突时明确报告（stash 不丢，可手动 `git stash pop` 处理）。

## 各场景行为

| 场景 | 行为 |
|---|---|
| 副本无 origin 远端 | 失败并提示（源副本先 `git remote add origin <URL>`；无远端的源副本本身就是最新） |
| 已是最新 | 报告当前提交，不动任何文件 |
| 有新提交 | 快进 + 列出 `old..new` 的提交摘要 |
| runtime/ 外有本地改动 | 拒绝更新并列出文件 |
| 更新成功且位于其他仓库内（如 .agents/medpowers） | 提醒：submodule 方式需在训练仓库 commit 子模块指针；嵌套 clone 方式无后续动作 |

## 约束

- 本技能只更新**自己所在的副本**，绝不碰其他副本或外层 `/ssd/lx` 大仓库。
- 更新后建议快速核对 `runtime/env.json`、`runtime/data.json` 是否为本地状态（stash pop 正常时无需人工干预）。
- 新技能/新资产是否生效以 git 历史为准；上游若重命名了默认分支，脚本按 origin/HEAD → main → master 顺序探测。
