"""cli.py — medpowers 统一 CLI：notify 子命令组 + exp 子命令组。

    medpowers notify set --machine-name m --url ...      # 训练通知组件（skills/notify/SKILL.md）
    medpowers exp new --slug unet3d-baseline             # 实验记录管理（skills/exp-mgmt/SKILL.md）
    python -m medpowers ...                              # 等价

技能脚本（skills/notify/notify.py、skills/exp-mgmt/exp_mgmt.py）是本 CLI 的薄启动器，
agent 侧调用形态不变；本入口供 pip 安装后的任务仓库环境直接使用。"""
from __future__ import annotations

import sys

from .exp_cli import ExpError
from .exp_cli import main as exp_main
from .notify import NotifyError
from .notify import main as notify_main

USAGE = """medpowers — MedPowers 训练运行时客户端（实验记录生命周期 + GPU lease + 训练通知）

用法：
  medpowers notify <set|disable|status|send|test> [参数...]      训练通知组件
  medpowers exp <new|start|finish|cancel|list|check|migrate> [参数...]  实验记录管理
  python -m medpowers ...                                        等价入口

训练框架内嵌（正路）：from medpowers import training_session
各命令参数加 -h 查看；协议见 medpowers 仓库 AGENTS.md 与 templates/runtime/README.md。"""


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in ("notify", "exp"):
        print(USAGE, file=sys.stderr)
        return 2
    group, rest = argv[0], argv[1:]
    if group == "notify":
        try:
            return notify_main(rest)
        except NotifyError as e:
            print(f"错误：{e}", file=sys.stderr)
            return 1
    try:
        return exp_main(rest)
    except ExpError as e:
        print(f"错误：{e}", file=sys.stderr)
        return 1
