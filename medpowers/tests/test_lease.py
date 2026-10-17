#!/usr/bin/env python3
"""medpowers.lease 的测试。运行（仓库根）：python3 -m unittest discover -s medpowers/tests -t ."""
import json
import os
import platform
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from medpowers import lease  # noqa: E402
from .gitrepo import write_notify_config  # noqa: E402


class LeaseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="lease-test-"))
        self.runtime = self.tmp / "runtime"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_write_read_remove_roundtrip(self):
        p = lease.write_lease("bc3seg-unet3d", task="BC3 腹部器官分割",
                              experiment="EXP-001", gpus=[0, 1],
                              runtime=self.runtime, log="runtime/logs/exp001.log")
        self.assertEqual(p, self.runtime / "training" / "bc3seg-unet3d.json")
        self.assertFalse((self.runtime / "training" / "bc3seg-unet3d.json.tmp").exists())  # 原子写不留残骸
        leases = lease.read_leases(self.runtime)
        self.assertEqual(len(leases), 1)
        self.assertEqual(leases[0]["experiment"], "EXP-001")
        self.assertEqual(leases[0]["gpus"], [0, 1])
        self.assertEqual(leases[0]["pid"], os.getpid())   # 训练进程自己写：真实 pid
        self.assertEqual(leases[0]["log"], "runtime/logs/exp001.log")
        self.assertTrue(lease.remove_lease("bc3seg-unet3d", runtime=self.runtime))
        self.assertEqual(lease.read_leases(self.runtime), [])
        self.assertFalse(lease.remove_lease("bc3seg-unet3d", runtime=self.runtime))

    def test_machine_identity_prefers_notify_config(self):
        # 未配置 → hostname 兜底
        self.assertEqual(lease.machine_identity(self.runtime), platform.node())
        write_notify_config(self.runtime, "http://x/hook", machine="机器A")
        self.assertEqual(lease.machine_identity(self.runtime), "机器A")

    def test_invalid_slug_rejected(self):
        with self.assertRaises(ValueError):
            lease.write_lease("../escape", runtime=self.runtime)
        with self.assertRaises(ValueError):
            lease.write_lease("", runtime=self.runtime)


if __name__ == "__main__":
    unittest.main(verbosity=2)
