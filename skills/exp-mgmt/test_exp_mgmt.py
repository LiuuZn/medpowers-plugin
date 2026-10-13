#!/usr/bin/env python3
"""exp_mgmt.py 的测试：临时 git 仓库 fixture，全流程与异常路径。运行：python3 skills/exp-mgmt/test_exp_mgmt.py -v"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp_mgmt  # noqa: E402


def make_repo(tmp: Path) -> Path:
    """建一个带 git 身份的任务仓库 fixture，返回仓库根。"""
    subprocess.run(["git", "init", "-q", str(tmp)], check=True)
    subprocess.run(["git", "-C", str(tmp), "config", "user.email", "t@local"], check=True)
    subprocess.run(["git", "-C", str(tmp), "config", "user.name", "tester"], check=True)
    (tmp / "experiments").mkdir()
    return tmp


def write_record(exp_dir: Path, name: str, status: str, planned="2026-10-01T10:00:00+08:00",
                 started="", ended="") -> Path:
    p = exp_dir / name
    p.write_text(
        f"# {name[:-3]}\n\n"
        f"- **status**: {status}\n"
        f"- **planned_at**: {planned}\n"
        f"- **started_at**: {started}\n"
        f"- **ended_at**: {ended}\n\n"
        f"## 任务与数据【规划时填】\n\n- 占位\n",
        encoding="utf-8",
    )
    return p


class CoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="exp-mgmt-test-"))
        self.repo = make_repo(self.tmp)
        self.exp_dir = self.tmp / "experiments"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_filename_regexes(self):
        self.assertTrue(exp_mgmt.NEW_RE.match("EXP-001-unet3d-baseline.md"))
        self.assertTrue(exp_mgmt.NEW_RE.match("EXP-1000-x.md"))
        self.assertFalse(exp_mgmt.NEW_RE.match("EXP-001.md"))
        self.assertTrue(exp_mgmt.OLD_RE.match("EXP-20261005-010-ensemble.md"))
        self.assertFalse(exp_mgmt.OLD_RE.match("EXP-001-ensemble.md"))

    def test_normalize_id(self):
        self.assertEqual(exp_mgmt.normalize_id("exp-1"), "EXP-001")
        self.assertEqual(exp_mgmt.normalize_id("EXP-1000"), "EXP-1000")
        with self.assertRaises(exp_mgmt.ExpError):
            exp_mgmt.normalize_id("001")

    def test_parse_record_fields(self):
        p = write_record(self.exp_dir, "EXP-001-foo.md", "running", started="2026-10-02T09:00:00+08:00")
        rec = exp_mgmt.parse_record(p)
        self.assertEqual(rec["num"], 1)
        self.assertEqual(rec["id"], "EXP-001")
        self.assertEqual(rec["slug"], "foo")
        self.assertEqual(rec["status"], "running")
        self.assertEqual(rec["started_at"], "2026-10-02T09:00:00+08:00")

    def test_parse_record_rejects_bad_name_and_status(self):
        with self.assertRaises(exp_mgmt.ExpError):
            exp_mgmt.parse_record(self.exp_dir / "EXP-001.md")
        p = write_record(self.exp_dir, "EXP-001-foo.md", "finished")
        with self.assertRaises(exp_mgmt.ExpError):
            exp_mgmt.parse_record(p)

    def test_scan_reports_old_format(self):
        write_record(self.exp_dir, "EXP-001-a.md", "planned")
        old = self.exp_dir / "EXP-20261005-010-ensemble.md"
        old.write_text("# 旧\n", encoding="utf-8")
        records, old_format = exp_mgmt.scan_records(self.exp_dir)
        self.assertEqual(len(records), 1)
        self.assertEqual(old_format, [old])

    def test_scan_rejects_unrecognized_name(self):
        (self.exp_dir / "EXP-xyz.md").write_text("# x\n", encoding="utf-8")
        with self.assertRaises(exp_mgmt.ExpError):
            exp_mgmt.scan_records(self.exp_dir)

    def test_duplicates_detected(self):
        write_record(self.exp_dir, "EXP-001-a.md", "planned")
        write_record(self.exp_dir, "EXP-001-b.md", "planned")
        records, _ = exp_mgmt.scan_records(self.exp_dir)
        with self.assertRaises(exp_mgmt.ExpError):
            exp_mgmt.check_duplicates(records)

    def test_time_consistency(self):
        ok = {"path": Path("EXP-001-a.md"), "status": "running",
              "started_at": "2026-10-02T09:00:00+08:00", "ended_at": ""}
        self.assertEqual(exp_mgmt.check_time_consistency(ok), [])
        bad = [
            {"path": Path("a.md"), "status": "planned", "started_at": "x", "ended_at": ""},
            {"path": Path("b.md"), "status": "running", "started_at": "", "ended_at": ""},
            {"path": Path("c.md"), "status": "done", "started_at": "x", "ended_at": ""},
        ]
        for rec in bad:
            self.assertTrue(exp_mgmt.check_time_consistency(rec), rec)

    def test_next_num(self):
        self.assertEqual(exp_mgmt.next_num([]), 1)
        write_record(self.exp_dir, "EXP-002-a.md", "planned")
        write_record(self.exp_dir, "EXP-009-b.md", "planned")
        records, _ = exp_mgmt.scan_records(self.exp_dir)
        self.assertEqual(exp_mgmt.next_num(records), 10)

    def test_render_index(self):
        write_record(self.exp_dir, "EXP-001-a.md", "done", started="2026-10-02T09:00:00+08:00",
                     ended="2026-10-03T09:00:00+08:00")
        records, _ = exp_mgmt.scan_records(self.exp_dir)
        text = exp_mgmt.render_index(records)
        self.assertIn("| EXP-001 | done | a | 2026-10-01 | 2026-10-02 | 2026-10-03 |", text)
        self.assertIn("| 编号 | 状态 | 标题 |", text)

    def test_require_repo_rejects_non_git(self):
        bare = Path(tempfile.mkdtemp(prefix="exp-mgmt-nogit-"))
        try:
            with self.assertRaises(exp_mgmt.ExpError):
                exp_mgmt.require_repo(bare)
        finally:
            shutil.rmtree(bare, ignore_errors=True)

    def test_commit_paths_skips_empty(self):
        self.assertFalse(exp_mgmt.commit_paths(self.repo, "exp: index 同步", self.exp_dir))


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="exp-mgmt-life-"))
        self.repo = make_repo(self.tmp)
        self.exp_dir = self.tmp / "experiments"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def gitlog(self) -> str:
        return subprocess.run(["git", "-C", str(self.repo), "log", "--format=%s"],
                              capture_output=True, text=True, check=True).stdout

    def test_new_creates_planned_and_commits(self):
        rc = exp_mgmt.main(["new", "--dir", str(self.exp_dir), "--slug", "unet3d-baseline"])
        self.assertEqual(rc, 0)
        rec_path = self.exp_dir / "EXP-001-unet3d-baseline.md"
        self.assertTrue(rec_path.exists())
        rec = exp_mgmt.parse_record(rec_path)
        self.assertEqual(rec["status"], "planned")
        self.assertTrue(rec["planned_at"])
        self.assertIn("exp(EXP-001): plan", self.gitlog())
        index = (self.exp_dir / "INDEX.md").read_text(encoding="utf-8")
        self.assertIn("| EXP-001 | planned | unet3d-baseline |", index)

    def test_new_numbers_increment(self):
        exp_mgmt.main(["new", "--dir", str(self.exp_dir), "--slug", "a"])
        exp_mgmt.main(["new", "--dir", str(self.exp_dir), "--slug", "b"])
        self.assertTrue((self.exp_dir / "EXP-002-b.md").exists())

    def test_new_rejects_bad_slug(self):
        with self.assertRaises(exp_mgmt.ExpError):
            exp_mgmt.main(["new", "--dir", str(self.exp_dir), "--slug", "Big Slug"])

    def test_new_rejects_when_old_format_present(self):
        (self.exp_dir / "EXP-20261005-010-ensemble.md").write_text("# 旧\n", encoding="utf-8")
        with self.assertRaises(exp_mgmt.ExpError):
            exp_mgmt.main(["new", "--dir", str(self.exp_dir), "--slug", "a"])

    def test_start_rejects_when_old_format_present(self):
        exp_mgmt.main(["new", "--dir", str(self.exp_dir), "--slug", "a"])
        (self.exp_dir / "EXP-20261005-010-ensemble.md").write_text("# 旧\n", encoding="utf-8")
        with self.assertRaises(exp_mgmt.ExpError):
            exp_mgmt.main(["start", "--dir", str(self.exp_dir), "EXP-001"])

    def test_new_leaves_unrelated_staged_file_alone(self):
        unrelated = self.tmp / "staged-unrelated.txt"
        unrelated.write_text("无关内容\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(self.repo), "add", "--", str(unrelated)], check=True)
        self.assertEqual(exp_mgmt.main(["new", "--dir", str(self.exp_dir), "--slug", "a"]), 0)
        names = subprocess.run(["git", "-C", str(self.repo), "show", "--name-only", "--format=",
                                "HEAD"], capture_output=True, text=True, check=True).stdout.split()
        self.assertTrue(names)  # exp 提交本身非空
        # 提交内容只含 experiments/ 下文件，无关预暂存文件不入提交
        self.assertTrue(all(n.startswith("experiments/") for n in names), names)
        # 无关文件仍处已暂存未提交状态
        status = subprocess.run(["git", "-C", str(self.repo), "status", "--porcelain"],
                                capture_output=True, text=True, check=True).stdout
        self.assertRegex(status, r"^A\s+\S*staged-unrelated\.txt$", re.M)

    def test_start_gate(self):
        exp_mgmt.main(["new", "--dir", str(self.exp_dir), "--slug", "a"])
        self.assertEqual(exp_mgmt.main(["start", "--dir", str(self.exp_dir), "EXP-001"]), 0)
        rec = exp_mgmt.parse_record(self.exp_dir / "EXP-001-a.md")
        self.assertEqual(rec["status"], "running")
        self.assertTrue(rec["started_at"])
        self.assertIn("exp(EXP-001): start", self.gitlog())

    def test_start_twice_rejected(self):
        exp_mgmt.main(["new", "--dir", str(self.exp_dir), "--slug", "a"])
        exp_mgmt.main(["start", "--dir", str(self.exp_dir), "EXP-001"])
        with self.assertRaises(exp_mgmt.ExpError):
            exp_mgmt.main(["start", "--dir", str(self.exp_dir), "EXP-001"])

    def test_start_missing_rejected(self):
        with self.assertRaises(exp_mgmt.ExpError):
            exp_mgmt.main(["start", "--dir", str(self.exp_dir), "EXP-009"])

    def test_finish_done(self):
        exp_mgmt.main(["new", "--dir", str(self.exp_dir), "--slug", "a"])
        exp_mgmt.main(["start", "--dir", str(self.exp_dir), "EXP-001"])
        self.assertEqual(
            exp_mgmt.main(["finish", "--dir", str(self.exp_dir), "EXP-001", "--status", "done"]), 0)
        rec = exp_mgmt.parse_record(self.exp_dir / "EXP-001-a.md")
        self.assertEqual(rec["status"], "done")
        self.assertTrue(rec["ended_at"])
        self.assertIn("exp(EXP-001): done", self.gitlog())

    def test_finish_requires_running(self):
        exp_mgmt.main(["new", "--dir", str(self.exp_dir), "--slug", "a"])
        with self.assertRaises(exp_mgmt.ExpError):
            exp_mgmt.main(["finish", "--dir", str(self.exp_dir), "EXP-001", "--status", "done"])

    def test_cancel_and_id_not_reused(self):
        exp_mgmt.main(["new", "--dir", str(self.exp_dir), "--slug", "a"])
        exp_mgmt.main(["cancel", "--dir", str(self.exp_dir), "EXP-001", "--reason", "标注契约未定"])
        rec = exp_mgmt.parse_record(self.exp_dir / "EXP-001-a.md")
        self.assertEqual(rec["status"], "cancelled")
        self.assertIn("标注契约未定", rec["path"].read_text(encoding="utf-8"))
        # planned 直接取消也须补齐时间字段（check 要求 cancelled 必有 started_at 与 ended_at）
        self.assertTrue(rec["started_at"] and rec["ended_at"])
        self.assertEqual(exp_mgmt.main(["check", "--dir", str(self.exp_dir)]), 0)
        exp_mgmt.main(["new", "--dir", str(self.exp_dir), "--slug", "b"])
        self.assertTrue((self.exp_dir / "EXP-002-b.md").exists())

    def test_cancel_requires_reason(self):
        exp_mgmt.main(["new", "--dir", str(self.exp_dir), "--slug", "a"])
        with self.assertRaises(SystemExit):
            exp_mgmt.main(["cancel", "--dir", str(self.exp_dir), "EXP-001"])


class ListCheckTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="exp-mgmt-lc-"))
        self.repo = make_repo(self.tmp)
        self.exp_dir = self.tmp / "experiments"
        exp_mgmt.main(["new", "--dir", str(self.exp_dir), "--slug", "a"])
        exp_mgmt.main(["start", "--dir", str(self.exp_dir), "EXP-001"])

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_list_regenerates_index_and_commits(self):
        (self.exp_dir / "INDEX.md").write_text("手改坏了\n", encoding="utf-8")
        # 手改内容先入 HEAD：否则 list 重生成的内容与 start 已提交的 INDEX 字节相同，
        # 无 diff、commit_paths 按设计不产生空提交，"exp: index 同步"无从出现。
        subprocess.run(["git", "-C", str(self.repo), "add", "--", str(self.exp_dir)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "commit", "-q", "-m", "手改 INDEX"], check=True)
        self.assertEqual(exp_mgmt.main(["list", "--dir", str(self.exp_dir)]), 0)
        index = (self.exp_dir / "INDEX.md").read_text(encoding="utf-8")
        self.assertIn("| EXP-001 | running | a |", index)
        log = subprocess.run(["git", "-C", str(self.repo), "log", "--format=%s"],
                             capture_output=True, text=True, check=True).stdout
        self.assertIn("exp: index 同步", log)

    def test_check_ok_full_flow(self):
        exp_mgmt.main(["finish", "--dir", str(self.exp_dir), "EXP-001", "--status", "done"])
        self.assertEqual(exp_mgmt.main(["check", "--dir", str(self.exp_dir)]), 0)

    def test_check_index_mismatch_fails(self):
        (self.exp_dir / "INDEX.md").write_text("不一致\n", encoding="utf-8")
        self.assertEqual(exp_mgmt.main(["check", "--dir", str(self.exp_dir)]), 1)

    def test_check_duplicate_fails(self):
        src = self.exp_dir / "EXP-001-a.md"
        shutil.copy(src, self.exp_dir / "EXP-001-a-copy.md")
        self.assertEqual(exp_mgmt.main(["check", "--dir", str(self.exp_dir)]), 1)

    def test_check_time_consistency_fails(self):
        p = self.exp_dir / "EXP-001-a.md"
        exp_mgmt.set_field(p, "started_at", "")  # 破坏 running 必有 started_at
        self.assertEqual(exp_mgmt.main(["check", "--dir", str(self.exp_dir)]), 1)

    def test_check_lease_ghost_and_status(self):
        runtime = self.tmp / "runtime" / "training"
        runtime.mkdir(parents=True)
        # 幽灵引用：指向不存在的 EXP-009（check 契约：错误返回 1，不抛异常）
        (runtime / "l1.json").write_text(json.dumps({"experiment": "EXP-009"}), encoding="utf-8")
        self.assertEqual(exp_mgmt.main(["check", "--dir", str(self.exp_dir),
                                        "--runtime", str(self.tmp / "runtime")]), 1)
        # 状态不符：记录被 cancel 后 lease 仍存在
        (runtime / "l1.json").write_text(json.dumps({"experiment": "EXP-001"}), encoding="utf-8")
        exp_mgmt.main(["cancel", "--dir", str(self.exp_dir), "EXP-001", "--reason", "r"])
        self.assertEqual(exp_mgmt.main(["check", "--dir", str(self.exp_dir),
                                        "--runtime", str(self.tmp / "runtime")]), 1)

    def test_check_lease_non_dict_payload(self):
        runtime = self.tmp / "runtime" / "training"
        runtime.mkdir(parents=True)
        (runtime / "l1.json").write_text(json.dumps(["not-a-dict"]), encoding="utf-8")
        # 畸形载荷须记为错误（rc=1），不得抛 AttributeError
        self.assertEqual(exp_mgmt.main(["check", "--dir", str(self.exp_dir),
                                        "--runtime", str(self.tmp / "runtime")]), 1)

    def test_check_lease_non_string_experiment(self):
        runtime = self.tmp / "runtime" / "training"
        runtime.mkdir(parents=True)
        (runtime / "l1.json").write_text(json.dumps({"experiment": 9}), encoding="utf-8")
        # experiment 非字符串须记为错误（rc=1），不得抛 AttributeError
        self.assertEqual(exp_mgmt.main(["check", "--dir", str(self.exp_dir),
                                        "--runtime", str(self.tmp / "runtime")]), 1)

    def test_check_lease_ok_when_running(self):
        runtime = self.tmp / "runtime" / "training"
        runtime.mkdir(parents=True)
        (runtime / "l1.json").write_text(json.dumps({"experiment": "EXP-001"}), encoding="utf-8")
        self.assertEqual(exp_mgmt.main(["check", "--dir", str(self.exp_dir),
                                        "--runtime", str(self.tmp / "runtime")]), 0)

    def test_check_warns_uncommitted_but_exits_zero(self):
        (self.exp_dir / "notes.md").write_text("未提交的杂项\n", encoding="utf-8")
        self.assertEqual(exp_mgmt.main(["check", "--dir", str(self.exp_dir)]), 0)


class MigrateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="exp-mgmt-mi-"))
        self.repo = make_repo(self.tmp)
        self.exp_dir = self.tmp / "experiments"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_old(self, name: str) -> Path:
        p = self.exp_dir / name
        p.write_text(f"# {name[:-3]}\n\n- **status**: done\n\n- 旧记录内容\n", encoding="utf-8")
        return p

    def test_migrate_renames_maps_and_rewrites_versions(self):
        self.write_old("EXP-20261004-002-alpha.md")
        self.write_old("EXP-20261004-010-beta.md")
        self.write_old("EXP-20261005-001-gamma.md")
        runtime = self.tmp / "runtime"
        runtime.mkdir()
        (runtime / "versions.json").write_text(json.dumps({
            "baseline": {"exp": "EXP-20261004-010-beta"},
            "notes": "sota 暂缺：EXP-011/013 未超基线",
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        self.assertEqual(
            exp_mgmt.main(["migrate", "--dir", str(self.exp_dir), "--runtime", str(runtime)]), 0)
        for new_name in ("EXP-001-alpha.md", "EXP-002-beta.md", "EXP-003-gamma.md"):
            self.assertTrue((self.exp_dir / new_name).exists(), new_name)
        self.assertEqual(json.loads((runtime / "versions.json").read_text(encoding="utf-8"))
                         ["baseline"]["exp"], "EXP-002")
        rec = exp_mgmt.parse_record(self.exp_dir / "EXP-002-beta.md")
        self.assertEqual(rec["slug"], "beta")
        self.assertEqual(rec["status"], "done")
        text_beta = (self.exp_dir / "EXP-002-beta.md").read_text(encoding="utf-8")
        self.assertIn("- **planned_at**: 2026-10-04T00:00:00+08:00", text_beta)
        self.assertIn("- **ended_at**: 2026-10-04T00:00:00+08:00", text_beta)
        # 迁移后 check 应全绿（字段补齐 + INDEX 已重生成）
        self.assertEqual(exp_mgmt.main(["check", "--dir", str(self.exp_dir)]), 0)
        log = subprocess.run(["git", "-C", str(self.repo), "log", "--format=%s"],
                             capture_output=True, text=True, check=True).stdout
        self.assertIn("exp: migrate 旧格式记录至新编号（3 条）", log)
        # 迁移后可继续 new，编号接续
        exp_mgmt.main(["new", "--dir", str(self.exp_dir), "--slug", "fresh"])
        self.assertTrue((self.exp_dir / "EXP-004-fresh.md").exists())

    def test_migrate_rejects_statusless_old_record(self):
        p = self.exp_dir / "EXP-20261001-001-x.md"
        p.write_text("# EXP-20261001-001-x\n\n没有状态行\n", encoding="utf-8")
        with self.assertRaises(exp_mgmt.ExpError):
            exp_mgmt.main(["migrate", "--dir", str(self.exp_dir)])

    def test_migrate_atomic_when_one_record_statusless(self):
        self.write_old("EXP-20261002-001-good.md")
        (self.exp_dir / "EXP-20261003-001-bad.md").write_text(
            "# EXP-20261003-001-bad\n\n没有状态行\n", encoding="utf-8")
        # 先入初始提交：失败迁移不得产生 migrate 提交，git log 才有可断言的历史
        subprocess.run(["git", "-C", str(self.repo), "add", "--", str(self.exp_dir)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "commit", "-q", "-m", "init"], check=True)
        with self.assertRaises(exp_mgmt.ExpError):
            exp_mgmt.main(["migrate", "--dir", str(self.exp_dir)])
        # 原子性：磁盘未动——合法记录保持旧文件名，新编号文件不存在，工作区无变更，无 migrate 提交
        self.assertTrue((self.exp_dir / "EXP-20261002-001-good.md").exists())
        self.assertFalse((self.exp_dir / "EXP-001-good.md").exists())
        self.assertEqual(subprocess.run(["git", "-C", str(self.repo), "status", "--porcelain"],
                                        capture_output=True, text=True, check=True).stdout, "")
        log = subprocess.run(["git", "-C", str(self.repo), "log", "--format=%s"],
                             capture_output=True, text=True, check=True).stdout
        self.assertNotIn("migrate", log)

    def test_migrate_atomic_when_one_record_status_invalid(self):
        self.write_old("EXP-20261002-001-good.md")
        (self.exp_dir / "EXP-20261003-001-bad.md").write_text(
            "# EXP-20261003-001-bad\n\n- **status**: Finished\n", encoding="utf-8")
        # 先入初始提交：失败迁移不得产生 migrate 提交，git log 才有可断言的历史
        subprocess.run(["git", "-C", str(self.repo), "add", "--", str(self.exp_dir)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "commit", "-q", "-m", "init"], check=True)
        with self.assertRaises(exp_mgmt.ExpError) as cm:
            exp_mgmt.main(["migrate", "--dir", str(self.exp_dir)])
        msg = str(cm.exception)
        self.assertIn("EXP-20261003-001-bad", msg)  # 报错定位到非法记录
        self.assertIn("Finished", msg)              # 报错点明 status 值非法
        # 原子性：磁盘未动——合法记录保持旧文件名，新编号文件不存在，工作区无变更，无 migrate 提交
        self.assertTrue((self.exp_dir / "EXP-20261002-001-good.md").exists())
        self.assertFalse((self.exp_dir / "EXP-001-good.md").exists())
        self.assertEqual(subprocess.run(["git", "-C", str(self.repo), "status", "--porcelain"],
                                        capture_output=True, text=True, check=True).stdout, "")
        log = subprocess.run(["git", "-C", str(self.repo), "log", "--format=%s"],
                             capture_output=True, text=True, check=True).stdout
        self.assertNotIn("migrate", log)

    def test_migrate_atomic_when_versions_json_corrupt(self):
        self.write_old("EXP-20261002-001-good.md")
        runtime = self.tmp / "runtime"
        runtime.mkdir()
        (runtime / "versions.json").write_text("{oops", encoding="utf-8")
        # 解析失败须包成 ExpError（__main__ 只捕 ExpError），且任何改名都不得发生
        with self.assertRaises(exp_mgmt.ExpError):
            exp_mgmt.main(["migrate", "--dir", str(self.exp_dir), "--runtime", str(runtime)])
        self.assertTrue((self.exp_dir / "EXP-20261002-001-good.md").exists())
        self.assertFalse((self.exp_dir / "EXP-001-good.md").exists())

    def test_migrate_tracked_old_record(self):
        p = self.write_old("EXP-20261001-001-old.md")
        # 旧记录先入库成为已跟踪文件：覆盖 git mv 成功路径
        subprocess.run(["git", "-C", str(self.repo), "add", "--", str(p)], check=True)
        subprocess.run(["git", "-C", str(self.repo), "commit", "-q", "-m", "init"], check=True)
        self.assertEqual(exp_mgmt.main(["migrate", "--dir", str(self.exp_dir)]), 0)
        self.assertFalse(p.exists())
        new = self.exp_dir / "EXP-001-old.md"
        self.assertTrue(new.exists())
        self.assertEqual(exp_mgmt.parse_record(new)["id"], "EXP-001")
        status = subprocess.run(["git", "-C", str(self.repo), "status", "--porcelain"],
                                capture_output=True, text=True, check=True).stdout
        self.assertEqual(status, "")
        log = subprocess.run(["git", "-C", str(self.repo), "log", "--format=%s"],
                             capture_output=True, text=True, check=True).stdout
        self.assertEqual(log.count("exp: migrate 旧格式记录至新编号（1 条）"), 1)

    def test_migrate_none_is_noop(self):
        self.assertEqual(exp_mgmt.main(["migrate", "--dir", str(self.exp_dir)]), 0)

    def test_migrate_starts_after_existing_new_records(self):
        exp_mgmt.main(["new", "--dir", str(self.exp_dir), "--slug", "a"])
        self.write_old("EXP-20261001-001-old.md")
        self.assertEqual(exp_mgmt.main(["migrate", "--dir", str(self.exp_dir)]), 0)
        self.assertTrue((self.exp_dir / "EXP-002-old.md").exists())

    def test_rewrite_exp_fields_recursive(self):
        data = {"a": {"exp": "EXP-20261004-002-alpha"}, "b": [{"exp": "keep"}, {"exp": "EXP-20261005-001-gamma"}]}
        mapping = {"EXP-20261004-002-alpha": ("EXP-001-alpha.md", "EXP-001"),
                   "EXP-20261005-001-gamma": ("EXP-003-gamma.md", "EXP-003")}
        self.assertTrue(exp_mgmt.rewrite_exp_fields(data, mapping))
        self.assertEqual(data["a"]["exp"], "EXP-001")
        self.assertEqual(data["b"][0]["exp"], "keep")
        self.assertEqual(data["b"][1]["exp"], "EXP-003")


if __name__ == "__main__":
    unittest.main()
