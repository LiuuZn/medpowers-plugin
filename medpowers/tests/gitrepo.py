"""测试夹具：临时任务仓库（worktree + bare 远端），模拟多机台账的 git 环境。"""
from __future__ import annotations

import subprocess
from pathlib import Path


def _git(repo: Path, *args: str) -> str:
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if p.returncode != 0:
        raise AssertionError(f"git {args} 失败：{p.stderr}")
    return p.stdout


def make_git_repo(tmp: Path) -> tuple[Path, Path]:
    """建 <tmp>/taskrepo（main 分支，experiments/ 与 runtime/ 已建，已推上游 origin），
    返回 (repo, bare_remote)。"""
    remote = tmp / "remote.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(remote)],
                   check=True, capture_output=True)
    repo = tmp / "taskrepo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    _git(repo, "config", "user.email", "test@medpowers")
    _git(repo, "config", "user.name", "medpowers-test")
    (repo / "experiments").mkdir()
    (repo / "runtime").mkdir()
    (repo / "experiments" / ".gitkeep").write_text("", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")
    _git(repo, "remote", "add", "origin", str(remote))
    _git(repo, "push", "-u", "origin", "main")
    return repo, remote


def second_clone(tmp: Path, remote: Path, name: str = "other-machine") -> Path:
    """从 bare 远端再 clone 一份，模拟他机。"""
    repo = tmp / name
    subprocess.run(["git", "clone", str(remote), str(repo)], check=True, capture_output=True)
    _git(repo, "config", "user.email", "other@medpowers")
    _git(repo, "config", "user.name", "other-machine")
    return repo


def write_notify_config(runtime: Path, url: str, machine: str = "lx-4090") -> None:
    """写最小可用的 notify.json。"""
    import json
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "notify.json").write_text(json.dumps({
        "enabled": True, "machine_name": machine,
        "webhook": {"type": "generic", "url": url, "secret": ""},
    }, ensure_ascii=False), encoding="utf-8")
