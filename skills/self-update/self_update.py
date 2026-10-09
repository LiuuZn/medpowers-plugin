#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""self_update.py — 拉取本仓库（mtskills）最新主分支，更新脚本所在的副本。

自动定位自身所在仓库根（与 cwd 无关），安全策略：
  1) 只快进（--ff-only），绝不合并/rebase；
  2) runtime/ 外的本地改动 → 拒绝更新并逐文件列出；
  3) runtime/ 内的本地状态 → stash → 更新 → pop，冲突时明确报告。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def die(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def git(*args: str, check: bool = True):
    r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        die(f"git {' '.join(args)} 失败：{(r.stderr or r.stdout).strip()}")
    return r


def main() -> None:
    if git("rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
        die(f"{REPO} 不是 git 工作区")

    r = git("remote", "get-url", "origin", check=False)
    if r.returncode != 0 or not r.stdout.strip():
        die(f"{REPO} 没有 origin 远端，无从拉取。源副本请先 git remote add origin <URL>；"
            "训练仓库副本按 README.md 重新安装（submodule / 嵌套 clone 自带 origin）")
    url = r.stdout.strip()
    print(f"[INFO] 副本 {REPO}\n[INFO] origin {url}")

    # 本地改动分诊：untracked（??）不阻塞；tracked 改动按 runtime/ 内外分流
    dirty = [l for l in git("status", "--porcelain=v1").stdout.splitlines() if l.strip()]
    tracked = [(l[:2].strip(), l[3:].strip()) for l in dirty if not l.startswith("??")]
    blocked = [p for _c, p in tracked if not p.startswith("runtime/")]
    runtime_dirty = [p for _c, p in tracked if p.startswith("runtime/")]
    if blocked:
        die("副本在 runtime/ 外有本地改动，拒绝更新（防覆盖）。请先提交 / stash / 丢弃：\n  "
            + "\n  ".join(blocked))
    stashed = False
    if runtime_dirty:
        print(f"[INFO] runtime/ 有本地状态（{len(runtime_dirty)} 个文件），自动 stash → 更新 → pop")
        git("stash", "push", "-m", "self-update: runtime 状态暂存", "--", "runtime/")
        stashed = True

    git("fetch", "origin")

    # 主分支探测：origin/HEAD → main → master
    b = None
    r = git("symbolic-ref", "--short", "refs/remotes/origin/HEAD", check=False)
    if r.returncode == 0:
        b = r.stdout.strip().removeprefix("origin/")
    if not b or git("rev-parse", "--verify", f"refs/remotes/origin/{b}", check=False).returncode != 0:
        for cand in ("main", "master"):
            if git("rev-parse", "--verify", f"refs/remotes/origin/{cand}", check=False).returncode == 0:
                b = cand
                break
    if not b:
        die("无法确定 origin 主分支（origin/HEAD 未设置，main/master 也不存在）")

    old = git("rev-parse", "--short", "HEAD").stdout.strip()
    r = git("merge", "--ff-only", f"origin/{b}", check=False)
    if r.returncode != 0:
        die(f"快进失败（本地提交与 origin/{b} 分叉？）：{(r.stderr or r.stdout).strip()}\n"
            "请人工检查 git log 后决定 rebase 或重装副本")

    if stashed:
        pop = git("stash", "pop", check=False)
        if pop.returncode != 0:
            print(f"[WARN] 更新完成，但 runtime 状态 stash pop 冲突：{(pop.stderr or pop.stdout).strip()}\n"
                  "       stash 未丢弃，请手动处理：git -C <副本> stash pop / git status")

    new = git("rev-parse", "--short", "HEAD").stdout.strip()
    if old == new:
        print(f"[OK] 已是最新（origin/{b} @ {new}）")
    else:
        print(f"[OK] 已更新 {old} → {new}（origin/{b}）：")
        for line in git("log", "--oneline", f"{old}..{new}").stdout.splitlines():
            print(f"       {line}")

    # 安装形态提示：submodule 用 git 原生检测；嵌套 clone 看 .agents 布局启发式
    # （源仓库 /ssd/lx/mtskills 虽身处 /ssd/lx 大仓库工作区内，但既非 submodule 也非 .agents 安装，保持沉默）
    sp = git("rev-parse", "--show-superproject-working-tree").stdout.strip()
    if sp:
        print(f"[INFO] 本副本是 {sp} 的 submodule：请在训练仓库 commit 子模块指针")
    elif REPO.parent.name == ".agents" and (REPO.parent.parent / ".git").exists():
        print("[INFO] 本副本为 .agents/ 下的嵌套 clone：训练仓库无需动作（确认 .gitignore 已含 .agents/mtskills/）")


if __name__ == "__main__":
    main()
