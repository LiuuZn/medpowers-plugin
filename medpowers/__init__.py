"""medpowers — MedPowers 训练运行时客户端（仓库即插件即包源）。

训练框架内嵌的公开 API：

    from medpowers import training_session, send_notification, training_lifecycle

    with training_session("EXP-001", gpus=[0, 1], task="bc3seg-unet3d") as s:
        s.artifacts.append("runs/exp001/best.pth")
        main_train()

training_session = 全接管会话（start 门禁 → lease → 通知 → finish 落账 → 删 lease →
通知），两级纪律见 session.py；send_notification / training_lifecycle = 纯通知
（尽力而为，绝不影响训练）。安装：任务仓库 `uv add <medpowers 副本根>`（插件主导，
版本随插件）。"""
from .lease import machine_identity, read_leases, remove_lease, write_lease
from .notify import send_notification, training_lifecycle
from .records import ExpError, finish_experiment, start_experiment
from .session import TrainingInterrupted, training_session

__all__ = [
    "ExpError", "TrainingInterrupted",
    "training_session", "training_lifecycle", "send_notification",
    "start_experiment", "finish_experiment",
    "write_lease", "read_leases", "remove_lease", "machine_identity",
]
