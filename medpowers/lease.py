"""lease.py — GPU lease 读写原语（协议正本：templates/runtime/README.md）。

lease 是声明式参考而非硬锁：给所有智能体/进程提供"哪些任务在训练、占用哪张卡"的上下文。
全接管模式下由**训练进程自己**在 start 门禁通过后写入（此时才拿得到分离后的真实 pid），
训练结束/中断时删除。字段：task/repo/experiment/machine/agent/pid/gpus/gpu_mem_mb/
started/eta_hours/log/notes；文件名为任务 slug；原子写（先 .tmp 再 replace）。"""
from __future__ import annotations

import json
import os
import platform
from datetime import datetime
from pathlib import Path

from .notify import load_config, resolve_runtime


def lease_dir(runtime: str | Path | None = None) -> Path:
    return resolve_runtime(runtime if isinstance(runtime, str) else
                           (str(runtime) if runtime else None)) / "training"


def machine_identity(runtime: str | Path | None = None) -> str:
    """机器指代正名：notify.json 的 machine_name；未配置用 hostname 兜底（协议口径）。"""
    rt = resolve_runtime(str(runtime) if runtime else None)
    try:
        cfg = load_config(rt)
    except Exception:
        cfg = None
    if isinstance(cfg, dict) and cfg.get("machine_name"):
        return str(cfg["machine_name"])
    return platform.node()


def write_lease(slug: str, *, task: str = "", experiment: str = "", gpus=None,
                pid: int | None = None, machine: str | None = None,
                runtime: str | Path | None = None, repo: str = "",
                gpu_mem_mb: int | None = None, eta_hours: float | None = None,
                log: str = "", notes: str = "", agent: str = "") -> Path:
    """原子写一份 lease JSON 到 <runtime>/training/<slug>.json；pid 默认本进程。"""
    if not slug or "/" in slug or slug.startswith("."):
        raise ValueError(f"lease slug「{slug}」非法（用作文件名：非空、无路径分隔符）")
    d = lease_dir(runtime)
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": task,
        "repo": repo,
        "experiment": experiment,
        "machine": machine if machine is not None else machine_identity(runtime),
        "agent": agent,
        "pid": pid if pid is not None else os.getpid(),
        "gpus": [int(g) for g in (gpus or [])],
        "gpu_mem_mb": gpu_mem_mb,
        "started": datetime.now().astimezone().isoformat(timespec="seconds"),
        "eta_hours": eta_hours,
        "log": log,
        "notes": notes,
    }
    p = d / f"{slug}.json"
    tmp = d / f"{slug}.json.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    return p


def read_leases(runtime: str | Path | None = None) -> list[dict]:
    """读全部 lease（附带 _path 指向文件）；坏文件以 _error 标注，不抛异常。"""
    out = []
    d = lease_dir(runtime)
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            payload = {"_error": str(e)}
        if isinstance(payload, dict):
            payload.setdefault("_path", str(p))
            out.append(payload)
    return out


def remove_lease(slug: str, runtime: str | Path | None = None) -> bool:
    """删除自己的 lease（训练结束/中断时调用）；不存在返回 False。"""
    p = lease_dir(runtime) / f"{slug}.json"
    if p.exists():
        p.unlink()
        return True
    return False
