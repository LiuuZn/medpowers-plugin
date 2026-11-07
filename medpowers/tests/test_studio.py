#!/usr/bin/env python3
"""medpowers.studio 的测试：本地 HTTP mock + 临时 runtime。
运行（仓库根）：python3 -m unittest discover -s medpowers/tests -t ."""
import contextlib
import io
import json
import shutil
import stat
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from medpowers import exp_cli, studio  # noqa: E402
from medpowers.session import training_session  # noqa: E402
from .gitrepo import make_git_repo  # noqa: E402


class StudioHandler(BaseHTTPRequestHandler):
    calls: list[dict] = []
    fail = False

    @classmethod
    def reset(cls):
        cls.calls = []
        cls.fail = False

    def _body(self):
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8") if length else ""
        try:
            return json.loads(raw) if raw else None
        except json.JSONDecodeError:
            return raw

    def _respond(self, status: int, payload):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def _handle(self, method: str):
        body = self._body()
        StudioHandler.calls.append(
            {"method": method, "path": self.path, "body": body, "headers": dict(self.headers)}
        )
        if StudioHandler.fail:
            self._respond(503, {"detail": "mock down"})
            return
        path = self.path.split("?", 1)[0]
        if path == "/api/auth/device/code":
            base = f"http://127.0.0.1:{self.server.server_address[1]}"
            self._respond(
                200,
                {
                    "device_code": "device-code-1234567890",
                    "user_code": "ABCD-EFGH",
                    "verification_uri": base + "/device",
                    "verification_uri_complete": base + "/device?user_code=ABCD-EFGH",
                    "expires_in": 30,
                    "interval": 1,
                },
            )
        elif path == "/api/auth/device/token":
            self._respond(
                200,
                {
                    "access_token": "mps_from_device",
                    "token_type": "Bearer",
                    "scope": "studio:read",
                    "repo_key": "repo1",
                    "expires_at": "2027-01-01T00:00:00+00:00",
                    "user": {"username": "tester"},
                },
            )
        elif path == "/api/auth/context":
            self._respond(
                200,
                {
                    "service": "MedPowers Studio",
                    "auth_method": "token",
                    "scopes": ["studio:read", "studio:experiment"],
                    "capabilities": {"read": True, "experiment": True},
                    "repo_key": "repo1",
                    "user": {"username": "tester", "display_name": "tester", "role": "admin"},
                },
            )
        elif path == "/api/studio/meta":
            self._respond(200, {"schema_version": "test", "version": "0.1.0"})
        elif path == "/api/studio/tasks":
            self._respond(200, [])
        elif path == "/api/studio/experiments":
            self._respond(
                200,
                {
                    "id": "uuid",
                    "repo_key": (body or {}).get("repo_key", "repo1"),
                    "exp_id": (body or {}).get("exp_id", "EXP-001"),
                    "task_id": "task",
                    "task_code": (body or {}).get("task_code", "model"),
                    "status": (body or {}).get("status", "running"),
                    "title": "",
                    "machine": "",
                    "git_commit": "",
                    "config": {},
                    "notes": "",
                    "record_path": "",
                    "record_sha256": "",
                    "started_at": None,
                    "ended_at": None,
                    "created_at": "2026-11-07T00:00:00+00:00",
                    "updated_at": "2026-11-07T00:00:00+00:00",
                },
            )
        elif path.endswith("/metrics:batch"):
            points = (body or {}).get("points", [])
            self._respond(200, {"accepted": len(points), "experiment_id": "uuid"})
        elif path.endswith("/evaluations"):
            self._respond(
                200,
                {
                    "id": "eval-uuid",
                    "experiment_id": "uuid",
                    "eval_id": (body or {}).get("eval_id", "e"),
                    "source": "external",
                    "task_code": "model",
                    "gt_statuses": ["approved"],
                    "dataset_version": "",
                    "summary": {},
                    "per_case": [],
                    "report_sha256": "",
                    "created_at": "2026-11-07T00:00:00+00:00",
                },
            )
        elif path.startswith("/api/studio/experiments/"):
            self._respond(
                200,
                {
                    "id": "uuid",
                    "repo_key": "repo1",
                    "exp_id": "EXP-001",
                    "task_id": "task",
                    "task_code": "model",
                    "status": (body or {}).get("status", "done"),
                    "title": "",
                    "machine": "",
                    "git_commit": "",
                    "config": {},
                    "notes": "",
                    "record_path": "",
                    "record_sha256": "",
                    "started_at": None,
                    "ended_at": None,
                    "created_at": "2026-11-07T00:00:00+00:00",
                    "updated_at": "2026-11-07T00:00:00+00:00",
                },
            )
        elif path.endswith("/request-review"):
            self._respond(200, {"id": "ann-1", "status": "review_requested"})
        else:
            self._respond(404, {"detail": "not found"})

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def do_PATCH(self):
        self._handle("PATCH")

    def log_message(self, *args):
        pass


class StudioTest(unittest.TestCase):
    def setUp(self):
        StudioHandler.reset()
        self.tmp = Path(tempfile.mkdtemp(prefix="studio-test-"))
        self.runtime = self.tmp / "runtime"
        self.server = HTTPServer(("127.0.0.1", 0), StudioHandler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.cfg = {
            "base_url": self.base_url,
            "repo_key": "repo1",
            "scopes": ["studio:read", "studio:experiment", "studio:evaluate"],
            "allow_insecure_http": True,
            "verify_tls": True,
            "ca_bundle": "",
        }

    def tearDown(self):
        self.server.shutdown()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _client(self):
        studio.save_config(self.runtime, self.cfg, token="mps_test_token")
        cfg = studio.load_config(self.runtime)
        return studio.StudioClient.from_config(cfg)

    def test_config_saves_token_0600_and_never_returns_it_in_json(self):
        studio.save_config(self.runtime, self.cfg, token="mps_secret")
        raw = json.loads((self.runtime / "studio.json").read_text(encoding="utf-8"))
        self.assertNotIn("token", raw)
        token_file = self.runtime / "studio.token"
        self.assertEqual(token_file.read_text(encoding="utf-8").strip(), "mps_secret")
        self.assertEqual(stat.S_IMODE(token_file.stat().st_mode), 0o600)
        cfg = studio.load_config(self.runtime)
        self.assertEqual(cfg["token"], "mps_secret")
        self.assertEqual(cfg["_runtime"], str(self.runtime.resolve()))

    def test_device_auth_saves_config_and_token(self):
        cfg = studio.device_auth(
            base_url=self.base_url,
            repo_key="repo1",
            scopes=["studio:read"],
            allow_insecure_http=True,
            runtime=self.runtime,
            log=lambda _: None,
        )
        self.assertEqual(cfg["repo_key"], "repo1")
        saved = studio.load_config(self.runtime)
        self.assertEqual(saved["token"], "mps_from_device")
        self.assertEqual(saved["base_url"], self.base_url)

    def test_http_requires_explicit_opt_in(self):
        with self.assertRaises(studio.StudioError):
            studio.StudioClient("http://127.0.0.1:1", allow_insecure_http=False)

    def test_client_sends_token_and_repo_key(self):
        client = self._client()
        payload = {
            "exp_id": "EXP-001",
            "task_code": "model",
            "status": "running",
            "repo_key": "repo1",
        }
        client.upsert_experiment(payload)
        call = StudioHandler.calls[-1]
        self.assertEqual(call["method"], "POST")
        self.assertEqual(call["headers"]["Authorization"], "Bearer mps_test_token")
        self.assertEqual(call["body"]["repo_key"], "repo1")

    def test_reporter_batches_and_finishes(self):
        client = self._client()
        reporter = studio.ExperimentReporter(
            client, "EXP-001", "model", runtime=self.runtime, repo_key="repo1", log=lambda _: None
        )
        reporter.start(config={"lr": 1e-4})
        reporter.log_metrics(
            [{"name": "train/loss", "value": 1.0 / (i + 1), "step": i} for i in range(250)]
        )
        reporter.finish("done", notes="ok")
        metric_calls = [
            c for c in StudioHandler.calls
            if c["path"].split("?", 1)[0].endswith("metrics:batch")
        ]
        self.assertEqual(len(metric_calls), 1)
        self.assertEqual(sum(len(c["body"]["points"]) for c in metric_calls), 250)
        self.assertTrue(any(c["method"] == "PATCH" for c in StudioHandler.calls))

    def test_failure_goes_to_queue_and_replays(self):
        client = self._client()
        reporter = studio.ExperimentReporter(
            client, "EXP-001", "model", runtime=self.runtime, repo_key="repo1", log=lambda _: None
        )
        StudioHandler.fail = True
        reporter.start()
        reporter.log_metrics([{"name": "train/loss", "value": 0.5, "step": 1}])
        reporter.flush()
        queue_file = studio.queue_dir(self.runtime, "repo1") / "EXP-001.jsonl"
        self.assertTrue(queue_file.exists())
        lines = queue_file.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)  # experiment + metrics

        StudioHandler.reset()
        sent = studio.flush_queue(client, self.runtime, "repo1", log=lambda _: None)
        self.assertEqual(sent, 2)
        self.assertFalse(queue_file.exists())

    def test_training_session_mirrors_experiment_and_metrics(self):
        repo, _ = make_git_repo(self.tmp / "repo")
        exp_dir = repo / "experiments"
        with contextlib.redirect_stdout(io.StringIO()):
            assert exp_cli.main(["new", "--slug", "studio-session", "--dir", str(exp_dir)]) == 0
        runtime = repo / "runtime"
        studio.save_config(runtime, self.cfg, token="mps_test_token")
        with training_session(
            "EXP-001",
            gpus=[0],
            task="studio session",
            runtime=str(runtime),
            exp_dir=str(exp_dir),
            studio_task="model",
            install_signals=False,
            log=lambda _: None,
        ) as session:
            session.report_metrics(
                [{"name": "train/loss", "value": 0.5, "step": 1, "split": "train"}]
            )
        paths = [c["path"].split("?", 1)[0] for c in StudioHandler.calls]
        self.assertTrue(any(p == "/api/studio/experiments" for p in paths), paths)
        self.assertTrue(any(p.endswith("metrics:batch") for p in paths), paths)
        self.assertTrue(any(p == "/api/studio/experiments/EXP-001" for p in paths), paths)
        self.assertTrue(
            any(c["method"] == "PATCH" for c in StudioHandler.calls), StudioHandler.calls
        )

    def test_evaluation_and_review_paths(self):
        client = self._client()
        client.push_evaluation(
            "EXP-001",
            {
                "eval_id": "local-1",
                "source": "external",
                "task_code": "model",
                "gt_statuses": ["approved"],
                "summary": {"dice": 0.9},
            },
        )
        self.assertTrue(StudioHandler.calls[-1]["path"].startswith(
            "/api/studio/experiments/EXP-001/evaluations"
        ))
        client.request_review("ann-1", "边界可疑", evidence={"dice": 0.4}, eval_id="local-1")
        self.assertTrue(StudioHandler.calls[-1]["path"].endswith("/request-review"))
        self.assertEqual(StudioHandler.calls[-1]["body"]["reason"], "边界可疑")


if __name__ == "__main__":
    unittest.main()
