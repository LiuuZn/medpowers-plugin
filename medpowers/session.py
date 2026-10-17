"""session.py — training_session：训练框架内嵌的全接管会话（实验记录 + lease + 通知）。

一个上下文管理器包住训练主流程，MedPowers 协议的训练侧职责全部由训练进程自己完成：

    from medpowers import training_session

    with training_session("EXP-001", gpus=[0, 1], task="bc3seg-unet3d",
                          note="nnUNet 3d_fullres", log_path="runtime/logs/exp001.log") as s:
        s.artifacts.append("runs/exp001/best.pth")   # 训练中随时登记产物
        main_train()   # 正常走完 → done；异常/SIGTERM/SIGINT → interrupted（原因入账）

进入时：start 门禁（拉取→编号唯一→planned→running→提交推送）→ 写自己的 lease（真实
pid）→ 发 start 通知。退出时：finish 落账（状态/时间/指标/产物/时长自动入账）→ 删
lease → 发 finish/interrupt 通知。

两级纪律（强制，与 AGENTS.md 对齐）：
- **门禁语义**：start 门禁失败（撞号/状态不符/远端不可达/分叉）在进入时抛 ExpError，
  训练**不得开始**——调用方不捕获即可（进程带着非零码退出）；
- **尽力而为**：通知任何失败绝不影响训练与收尾；finish 的推送失败不阻断退出（本地已
  提交，check 兜底报告未推送），SIGKILL/断电等无法自报的终止仍由智能体清理 stale
  lease 时兜底。诊断/下一轮决策章节留给智能体结合曲线补写。"""
from __future__ import annotations

import contextlib
import os
import re
import signal
import time

from .lease import machine_identity, remove_lease, write_lease
from .notify import send_notification
from .records import ExpError, finish_experiment, normalize_id, start_experiment


class TrainingInterrupted(BaseException):
    """SIGTERM/SIGINT 经 session 转成可收尾的中断（BaseException：不被训练代码的
    except Exception 吞掉，沿调用栈触发本 session 的收尾路径）。"""

    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(f"收到信号 {signal.Signals(signum).name}，训练中断")


class _Session:
    """with 块内暴露的句柄：exp_id / lease_path / artifacts（训练中随时追加产物路径）。"""

    def __init__(self, exp_id: str, lease_path, started_at: float):
        self.exp_id = exp_id
        self.lease_path = lease_path
        self.started_at = started_at   # time.monotonic() 起点
        self.artifacts: list[str] = []


def _slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", s.lower()).strip("-")


@contextlib.contextmanager
def training_session(experiment: str, *, task: str = "", slug: str = "",
                     gpus=None, note: str = "", results_fn=None, artifacts=None,
                     log_path: str = "", machine: str | None = None,
                     exp_dir: str | None = None, runtime: str | None = None,
                     install_signals: bool = True, allow_offline: bool = False,
                     log=print):
    """全接管会话（见模块 docstring）。results_fn 在收尾时调用、返回 dict 指标（异常
    路径也会尽力调用一次）；artifacts 可在参数给足，也可训练中经 s.artifacts 追加。"""
    exp_id = normalize_id(experiment)  # 格式非法直接抛，训练不得开始
    ids = [int(g) for g in (gpus or [])]
    lease_slug = _slugify(slug or task) or exp_id.lower()

    # ---- 进入：门禁 → lease → 通知（任何门禁失败都让 with 块根本不执行）----
    start_experiment(exp_id, gpus=ids,
                     machine=machine if machine is not None else machine_identity(runtime),
                     exp_dir=exp_dir, allow_offline=allow_offline, log=log)
    try:
        lease_path = write_lease(lease_slug, task=task, experiment=exp_id, gpus=ids,
                                 machine=machine, runtime=runtime,
                                 repo=os.getcwd(), log=log_path or "")
    except Exception as e:
        raise ExpError(f"lease 写入失败（{e}）：记录已转 running 但训练未启动——"
                       "请人工处理（修正后重试，或 cancel 该实验）") from e
    send_notification("start", gpus=ids, experiment=exp_id, note=note, runtime=runtime)

    state = {"signals": 0}

    def _on_signal(signum, _frame):
        if state["signals"] >= 1:
            # 第二次信号：收尾路径可能卡住（网络等），逃生门——按默认语义立即终止
            signal.signal(signum, signal.SIG_DFL)
            os.kill(os.getpid(), signum)
        state["signals"] += 1
        raise TrainingInterrupted(signum)

    installed = []
    if install_signals:
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                prev = signal.getsignal(sig)
                signal.signal(sig, _on_signal)
                installed.append((sig, prev))
            except (ValueError, OSError):
                pass  # 非主线程等场景装不上：跳过信号钩子，保留异常路径收尾

    session = _Session(exp_id, lease_path, time.monotonic())
    if artifacts:
        session.artifacts.extend(str(a) for a in artifacts)
    try:
        yield session
    except BaseException as e:
        why = str(e) if isinstance(e, TrainingInterrupted) else f"{type(e).__name__}: {e}"[:300]
        _wrap_up(session, "interrupted", why, results_fn, ids, note,
                 exp_dir, runtime, allow_offline, log)
        raise
    else:
        _wrap_up(session, "done", "", results_fn, ids, note,
                 exp_dir, runtime, allow_offline, log)
    finally:
        for sig, prev in installed:
            try:
                if signal.getsignal(sig) is _on_signal:
                    signal.signal(sig, prev)
            except (ValueError, OSError):
                pass


def _wrap_up(session: _Session, status: str, why: str, results_fn, ids: list[int],
             note: str, exp_dir, runtime, allow_offline, log) -> None:
    """收尾三步：finish 落账 → 删 lease → 通知。落账尽力而为（失败不阻断进程退出）。"""
    duration_s = time.monotonic() - session.started_at
    results = None
    if results_fn is not None:
        try:
            results = results_fn()
        except Exception as e:  # 收尾路径里的结果收集失败不掩盖训练本身的结局
            log(f"[警告] results_fn 收集指标失败（{type(e).__name__}: {e}），本次落账不含指标")
    merged_note = "；".join(x for x in (note, why) if x)
    try:
        finish_experiment(session.exp_id, status, results=results,
                          artifacts=session.artifacts or None, note=merged_note,
                          duration_s=duration_s, exp_dir=exp_dir,
                          allow_offline=allow_offline, log=log)
    except ExpError as e:
        log(f"[警告] 实验收尾落账失败：{e}\n  台账未更新——请让智能体检查并兜底补 finish"
            "（exp-mgmt CLI 的 finish 为遗留兜底路径）")
    remove_lease(session.lease_path.name[:-5], runtime=runtime)
    event = "interrupt" if status == "interrupted" else "finish"
    send_notification(event, gpus=ids, experiment=session.exp_id,
                      note=merged_note, runtime=runtime)
