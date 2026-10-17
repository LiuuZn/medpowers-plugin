#!/usr/bin/env python3
"""medpowers.session（training_session 全接管链路）的测试：
git 台账 + 本地 webhook 双夹具，验证 门禁→lease→通知→训练→落账→删 lease→通知 全链路。
运行（仓库根）：python3 -m unittest discover -s medpowers/tests -t ."""
import io
import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from medpowers import exp_cli, lease, records as R  # noqa: E402
from medpowers.session import training_session  # noqa: E402
from .gitrepo import make_git_repo, write_notify_config  # noqa: E402
from .test_notify import HookHandler  # noqa: E402


def noop(*a, **k):
    pass


def _notes():
    return [(r.get("event"), r.get("note", "")) for r in HookHandler.records]


class SessionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="session-test-"))
        self.repo, self.remote = make_git_repo(self.tmp)
        self.exp_dir = self.repo / "experiments"
        self.runtime = self.repo / "runtime"
        HookHandler.records = []
        HookHandler.behavior()
        self.server = HTTPServer(("127.0.0.1", 0), HookHandler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        write_notify_config(self.runtime, f"http://127.0.0.1:{self.port}/hook")

    def tearDown(self):
        self.server.shutdown()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _new(self, slug="baseline"):
        with contextlib.redirect_stdout(io.StringIO()):
            exp_cli.main(["new", "--slug", slug, "--dir", str(self.exp_dir)])

    def _record(self, num="EXP-001"):
        for p in self.exp_dir.glob(f"{num}-*.md"):
            return R.parse_record(p)
        raise AssertionError(f"{num} 记录不存在")

    def _wait_records(self, n, timeout=10.0):
        deadline = time.monotonic() + timeout
        while len(HookHandler.records) < n and time.monotonic() < deadline:
            time.sleep(0.05)
        self.assertGreaterEqual(len(HookHandler.records), n)

    # ---------- 全链路 ----------

    def test_happy_path_full_takeover(self):
        self._new()
        with training_session("EXP-001", gpus=[0, 1], task="bc3seg unet3d",
                              note="nnUNet 3d_fullres", runtime=self.runtime,
                              exp_dir=str(self.exp_dir),
                              results_fn=lambda: {"dice": 0.9},
                              artifacts=["runs/exp001/best.pth"],
                              log=noop) as s:
            s.artifacts.append("runtime/logs/exp001.log")
            # with 块内：门禁已过（running）、lease 已写（真实 pid）
            self.assertEqual(self._record()["status"], "running")
            payload = json.loads(s.lease_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["experiment"], "EXP-001")
            self.assertEqual(payload["pid"], os.getpid())
        # 收尾：done + 事实入账 + lease 删除 + 通知序列 start→finish
        rec = self._record()
        self.assertEqual(rec["status"], "done")
        self.assertTrue(rec["ended_at"])
        self.assertEqual(rec["machine"], "lx-4090")   # write_notify_config 的 machine_name
        self.assertEqual(rec["gpus"], "0,1")
        text = rec["path"].read_text(encoding="utf-8")
        self.assertIn('"dice": 0.9', text)
        self.assertIn("runs/exp001/best.pth", text)
        self.assertIn("runtime/logs/exp001.log", text)
        self.assertEqual(lease.read_leases(self.runtime), [])
        self._wait_records(2)
        self.assertEqual([e for e, _ in _notes()], ["start", "finish"])
        self.assertEqual(R.ahead_behind(self.repo, "origin/main"), (0, 0))  # 全部已推送

    def test_exception_path_interrupts_with_reason(self):
        self._new()
        with self.assertRaises(RuntimeError):
            with training_session("EXP-001", gpus=[0], runtime=self.runtime,
                                  exp_dir=str(self.exp_dir), log=noop):
                raise RuntimeError("CUDA OOM")
        rec = self._record()
        self.assertEqual(rec["status"], "interrupted")
        self.assertIn("RuntimeError: CUDA OOM", rec["path"].read_text(encoding="utf-8"))
        self.assertEqual(lease.read_leases(self.runtime), [])   # lease 已删
        self._wait_records(2)
        events = [e for e, _ in _notes()]
        self.assertEqual(events, ["start", "interrupt"])
        self.assertIn("CUDA OOM", _notes()[1][1])               # 中断通知带原因

    def test_gate_failure_training_must_not_start(self):
        self._new()
        with self.assertRaises(R.ExpError):  # 门禁失败：with 块根本不执行
            with training_session("EXP-404", gpus=[0], runtime=self.runtime,
                                  exp_dir=str(self.exp_dir), log=noop):
                raise AssertionError("门禁失败时训练不得开始")
        self.assertEqual(lease.read_leases(self.runtime), [])
        self.assertEqual(HookHandler.records, [])                # 无任何通知
        self.assertEqual(self._record()["status"], "planned")

    def test_start_gate_race_second_start_rejected(self):
        # 模拟他机抢先 start 同一实验：训练进程门禁必须拒绝
        self._new()
        R.start_experiment("EXP-001", exp_dir=str(self.exp_dir), log=noop)
        with self.assertRaises(R.ExpError):
            with training_session("EXP-001", gpus=[0], runtime=self.runtime,
                                  exp_dir=str(self.exp_dir), log=noop):
                raise AssertionError("门禁失败时训练不得开始")

    def test_sigterm_graceful_interrupt_in_subprocess(self):
        self._new()
        code = (
            "import sys, time\n"
            f"sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r})\n"
            "from medpowers import training_session, TrainingInterrupted\n"
            "try:\n"
            "    with training_session('EXP-001', gpus=[0], task='sigterm-test'):\n"
            "        time.sleep(60)\n"
            "except TrainingInterrupted:\n"
            "    pass  # 信号已转为可收尾中断：落账/删 lease/通知完成后干净退出\n"
        )
        env = {**os.environ,
               "MEDPOWERS_RUNTIME": str(self.runtime),
               "MEDPOWERS_EXPERIMENTS": str(self.exp_dir)}
        proc = subprocess.Popen([sys.executable, "-c", code], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            self._wait_records(1)  # start 通知 = 子进程已进入 with 块、lease 已写
            self.assertEqual(len(lease.read_leases(self.runtime)), 1)
            proc.send_signal(signal.SIGTERM)
            rc = proc.wait(timeout=30)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        self.assertEqual(rc, 0)                                   # 收尾完成后干净退出
        self.assertEqual(self._record()["status"], "interrupted")
        self.assertEqual(lease.read_leases(self.runtime), [])     # lease 已删
        self._wait_records(2)
        self.assertIn("SIGTERM", _notes()[1][1])
        self.assertEqual(R.ahead_behind(self.repo, "origin/main"), (0, 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
