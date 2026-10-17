#!/usr/bin/env python3
"""medpowers.records 的测试：临时 git 仓库 + bare 远端，覆盖门禁语义与多机同步红线。
运行（仓库根）：python3 -m unittest discover -s medpowers/tests -t ."""
import io
import contextlib
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from medpowers import records as R  # noqa: E402
from medpowers import exp_cli  # noqa: E402
from .gitrepo import make_git_repo, second_clone  # noqa: E402


def noop(*a, **k):
    pass


class RecordsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="records-test-"))
        self.repo, self.remote = make_git_repo(self.tmp)
        self.exp_dir = self.repo / "experiments"
        self.runtime = self.repo / "runtime"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---------- 夹具动作 ----------

    def _new(self, slug="baseline", num="EXP-001"):
        with contextlib.redirect_stdout(io.StringIO()):
            rc = exp_cli.main(["new", "--slug", slug, "--dir", str(self.exp_dir)])
        self.assertEqual(rc, 0)
        return self.exp_dir / f"{num}-{slug}.md"

    def _record(self, num="EXP-001"):
        for p in self.exp_dir.glob(f"{num}-*.md"):
            return R.parse_record(p)
        raise AssertionError(f"{num} 记录不存在")

    def _pushed_clean(self) -> bool:
        return R.ahead_behind(self.repo, "origin/main") == (0, 0)

    # ---------- start 门禁 ----------

    def test_start_gate_flips_running_writes_fields_and_pushes(self):
        self._new()
        res = R.start_experiment("EXP-001", gpus=[0, 1], machine="机器A",
                                 exp_dir=str(self.exp_dir), log=noop)
        self.assertTrue(res["pushed"])
        rec = self._record()
        self.assertEqual(rec["status"], "running")
        self.assertTrue(rec["started_at"])
        self.assertEqual(rec["machine"], "机器A")   # 可选结构化字段自动补插
        self.assertEqual(rec["gpus"], "0,1")
        self.assertIn("机器A", (self.exp_dir / "INDEX.md").read_text(encoding="utf-8"))
        self.assertTrue(self._pushed_clean())       # 提交已推送（多机可见）

    def test_start_gate_rejects_non_planned(self):
        self._new()
        R.start_experiment("EXP-001", exp_dir=str(self.exp_dir), log=noop)
        with self.assertRaises(R.ExpError):  # 已 running，不可再 start
            R.start_experiment("EXP-001", exp_dir=str(self.exp_dir), log=noop)

    def test_start_gate_rejects_unknown_id(self):
        with self.assertRaises(R.ExpError):
            R.start_experiment("EXP-404", exp_dir=str(self.exp_dir), log=noop)

    def test_start_gate_rejects_duplicate_numbers(self):
        p = self._new()
        dup = self.exp_dir / "EXP-001-dup.md"
        dup.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
        with self.assertRaises(R.ExpError):  # 撞号：门禁确认阶段拒绝
            R.start_experiment("EXP-001", exp_dir=str(self.exp_dir), log=noop)

    def test_start_gate_rejects_diverged_history(self):
        self._new()
        # 他机推一个提交，本机另有一个未推送提交 → 双向分叉，红线拒绝
        other = second_clone(self.tmp, self.remote)
        (other / "README.md").write_text("x", encoding="utf-8")
        import subprocess
        subprocess.run(["git", "-C", str(other), "add", "."], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(other), "commit", "-m", "other"], check=True,
                       capture_output=True)
        subprocess.run(["git", "-C", str(other), "push"], check=True, capture_output=True)
        (self.repo / "README.md").write_text("y", encoding="utf-8")
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True, capture_output=True)
        subprocess.run(["git", "-C", str(self.repo), "commit", "-m", "local"], check=True,
                       capture_output=True)
        with self.assertRaises(R.ExpError) as cm:
            R.start_experiment("EXP-001", exp_dir=str(self.exp_dir), log=noop)
        self.assertIn("分叉", str(cm.exception))

    def test_start_gate_rejects_unreachable_remote(self):
        self._new()
        import subprocess
        subprocess.run(["git", "-C", str(self.repo), "remote", "set-url", "origin",
                        str(self.tmp / "no-such-remote.git")], check=True, capture_output=True)
        with self.assertRaises(R.ExpError) as cm:
            R.start_experiment("EXP-001", exp_dir=str(self.exp_dir), log=noop)
        self.assertIn("拉取", str(cm.exception))  # 门禁语义：拉不下来就拒绝，训练不得开始

    # ---------- finish 落账 ----------

    def test_finish_done_appends_facts_and_fields(self):
        self._new()
        R.start_experiment("EXP-001", exp_dir=str(self.exp_dir), log=noop)
        res = R.finish_experiment("EXP-001", "done",
                                  results={"dice": 0.873, "val_loss": 0.031},
                                  artifacts=["runs/exp001/best.pth", "runtime/logs/exp001.log"],
                                  note="常规轮收尾", duration_s=12301.0,
                                  exp_dir=str(self.exp_dir), log=noop)
        self.assertTrue(res["pushed"])
        rec = self._record()
        self.assertEqual(rec["status"], "done")
        self.assertTrue(rec["ended_at"])
        text = res["path"].read_text(encoding="utf-8")
        self.assertIn('"dice": 0.873', text)          # 指标入 ## 结果
        self.assertIn("时长 3:25:01", text)           # 时长入账
        self.assertIn("说明：常规轮收尾", text)
        self.assertIn("runs/exp001/best.pth", text)   # 产物入 ## 产物
        self.assertTrue(self._pushed_clean())

    def test_finish_rejects_non_running(self):
        self._new()
        with self.assertRaises(R.ExpError):  # planned 不可 finish
            R.finish_experiment("EXP-001", "done", exp_dir=str(self.exp_dir), log=noop)

    def test_finish_push_failure_returns_not_pushed(self):
        self._new()
        R.start_experiment("EXP-001", exp_dir=str(self.exp_dir), log=noop)
        res = R.finish_experiment("EXP-001", "done", allow_offline=True,
                                  exp_dir=str(self.exp_dir), log=noop)
        self.assertFalse(res["pushed"])               # 离线豁免：本地已提交未推送，不抛
        self.assertEqual(R.ahead_behind(self.repo, "origin/main")[0], 1)
        self.assertEqual(self._record()["status"], "done")

    # ---------- 兼容与正文字段 ----------

    def test_set_field_inserts_missing_machine_line_after_existing_fields(self):
        p = self._new()
        # 模拟旧格式记录：删掉模板自带的 machine/gpus 空行
        text = "\n".join(l for l in p.read_text(encoding="utf-8").splitlines()
                         if not l.startswith("- **machine**:") and not l.startswith("- **gpus**:"))
        p.write_text(text + "\n", encoding="utf-8")
        self.assertNotIn("- **machine**:", text)
        R.set_field(p, "machine", "机器B")
        text = p.read_text(encoding="utf-8")
        self.assertIn("- **machine**: 机器B", text)
        # 补插位置在既有字段行之后（不破坏标题与正文章节）
        self.assertLess(text.index("- **ended_at**"), text.index("- **machine**"))
        self.assertEqual(R.parse_record(p)["machine"], "机器B")

    def test_append_facts_missing_headings_appends_sections(self):
        p = self._new()
        p.write_text("# EXP-001-baseline\n\n- **status**: planned\n- **planned_at**: t\n",
                     encoding="utf-8")  # 无 ## 结果/## 产物 节的极简记录
        R.append_facts(p, results={"dice": 0.5}, artifacts=["a.pth"])
        text = p.read_text(encoding="utf-8")
        self.assertIn("## 结果", text)
        self.assertIn("## 产物", text)
        self.assertIn("a.pth", text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
