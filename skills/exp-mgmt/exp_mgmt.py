#!/usr/bin/env python3
"""薄启动器：实际实现在仓库根 medpowers/ 包（单一事实源，随插件完整分发）。

agent 侧调用形态不变：
    python3 skills/exp-mgmt/exp_mgmt.py <new|start|finish|cancel|list|check|migrate> [参数...]

插件缓存、.agents/medpowers、源仓库三种副本都完整含包目录，路径解析通用；
训练框架侧请直接 pip 安装后 `from medpowers import ...`（见 SKILL.md）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from medpowers.exp_cli import ExpError, main  # noqa: E402

if __name__ == "__main__":
    try:
        sys.exit(main())
    except ExpError as e:
        print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)
