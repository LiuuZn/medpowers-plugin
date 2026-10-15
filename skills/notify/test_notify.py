#!/usr/bin/env python3
"""notify.py 的测试：临时 runtime fixture + 本地 HTTP webhook 接收端。
运行：python3 skills/notify/test_notify.py -v"""
import base64
import hashlib
import hmac
import json
import shutil
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import notify  # noqa: E402


class HookHandler(BaseHTTPRequestHandler):
    """本地 webhook 接收端：记录请求体，behavior() 决定响应。"""
    records: list[dict] = []
    status = 200
    body = '{"code":0,"msg":"success"}'

    @classmethod
    def behavior(cls, status=200, body='{"code":0,"msg":"success"}'):
        cls.status, cls.body = status, body

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        HookHandler.records.append(json.loads(self.rfile.read(length).decode("utf-8")))
        self.send_response(HookHandler.status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(HookHandler.body.encode("utf-8"))

    def log_message(self, *a):
        pass


class NotifyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="notify-test-"))
        self.runtime = self.tmp / "runtime"
        HookHandler.records = []
        self.server = HTTPServer(("127.0.0.1", 0), HookHandler)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        # 固定 GPU 名称查询，测试不依赖 nvidia-smi
        self._gpu_names = notify.gpu_names
        notify.gpu_names = lambda ids: {0: "RTX 4090", 1: "RTX 4090"}

    def tearDown(self):
        notify.gpu_names = self._gpu_names
        self.server.shutdown()
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ---------- 配置存取 ----------

    def test_set_writes_config_and_enables(self):
        rc = notify.main(["set", "--machine-name", "lx-4090", "--url",
                          f"http://127.0.0.1:{self.port}/hook", "--runtime", str(self.runtime)])
        self.assertEqual(rc, 0)
        cfg = json.loads((self.runtime / "notify.json").read_text(encoding="utf-8"))
        self.assertTrue(cfg["enabled"])
        self.assertEqual(cfg["machine_name"], "lx-4090")
        self.assertEqual(cfg["webhook"]["type"], "feishu")
        self.assertEqual(cfg["webhook"]["url"], f"http://127.0.0.1:{self.port}/hook")
        self.assertEqual(cfg["webhook"]["secret"], "")

    def test_set_rejects_bad_input(self):
        with self.assertRaises(notify.NotifyError):
            notify.main(["set", "--machine-name", "", "--url", "http://x/hook",
                         "--runtime", str(self.runtime)])
        with self.assertRaises(notify.NotifyError):
            notify.main(["set", "--machine-name", "m", "--url", "ftp://x/hook",
                         "--runtime", str(self.runtime)])

    def test_disable_creates_stub_and_send_skips(self):
        rc = notify.main(["disable", "--reason", "用户选择不配置", "--runtime", str(self.runtime)])
        self.assertEqual(rc, 0)
        cfg = json.loads((self.runtime / "notify.json").read_text(encoding="utf-8"))
        self.assertFalse(cfg["enabled"])
        self.assertEqual(cfg["disabled_reason"], "用户选择不配置")
        rc = notify.main(["send", "--event", "start", "--runtime", str(self.runtime)])
        self.assertEqual(rc, 0)  # 未配置/停用 → 跳过且 exit 0，不报错

    def test_send_without_config_skips(self):
        self.assertFalse((self.runtime / "notify.json").exists())
        rc = notify.main(["send", "--event", "start", "--runtime", str(self.runtime)])
        self.assertEqual(rc, 0)

    def test_status_masks_secret_and_url(self):
        import io, contextlib
        notify.main(["set", "--machine-name", "m", "--url",
                     f"http://127.0.0.1:{self.port}/hook/very-secret-token-abc123",
                     "--secret", "my-sign-secret", "--runtime", str(self.runtime)])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            notify.main(["status", "--runtime", str(self.runtime)])
        out = buf.getvalue()
        self.assertNotIn("very-secret-token-abc123", out)
        self.assertNotIn("my-sign-secret", out)
        self.assertIn("已配置，不显示", out)
        self.assertIn("machine_name", out)

    # ---------- payload 构造 ----------

    def test_feishu_sign_matches_hmac_spec(self):
        ts = "1599487832"
        sign = notify.feishu_sign("my-secret", ts)
        expect = base64.b64encode(
            hmac.new(f"{ts}\nmy-secret".encode(), digestmod=hashlib.sha256).digest()).decode()
        self.assertEqual(sign, expect)
        # 换时间戳/密钥结果必须变
        self.assertNotEqual(sign, notify.feishu_sign("my-secret", "1599487833"))
        self.assertNotEqual(sign, notify.feishu_sign("other", ts))

    def test_dry_run_card_payload(self):
        notify.main(["set", "--machine-name", "lx-4090", "--url", "http://x/hook",
                     "--secret", "sec", "--runtime", str(self.runtime)])
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            notify.main(["send", "--event", "start", "--gpus", "0,1",
                         "--experiment", "EXP-002", "--note", "开始训练", "--dry-run",
                         "--runtime", str(self.runtime)])
        payload = json.loads(buf.getvalue())["payload"]
        self.assertEqual(payload["msg_type"], "interactive")
        header = payload["card"]["header"]
        self.assertEqual(header["template"], "green")
        self.assertIn("训练开始", header["title"]["content"])
        self.assertIn("lx-4090", header["title"]["content"])
        md = payload["card"]["elements"][0]["text"]["content"]
        for frag in ("**机器：** lx-4090", "**GPU：** 0（RTX 4090）、1（RTX 4090）",
                     "**实验：** EXP-002", "**说明：** 开始训练", "**时间：** "):
            self.assertIn(frag, md)
        self.assertIn("timestamp", payload)   # 有密钥 → 加签字段在 payload 顶层
        self.assertIn("sign", payload)

    def test_dry_run_text_style_and_custom_event(self):
        notify.main(["set", "--machine-name", "m", "--url", "http://x/hook",
                     "--runtime", str(self.runtime)])
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            notify.main(["send", "--event", "数据准备完成", "--style", "text", "--dry-run",
                         "--runtime", str(self.runtime)])
        payload = json.loads(buf.getvalue())["payload"]
        self.assertEqual(payload["msg_type"], "text")
        self.assertIn("数据准备完成", payload["content"]["text"])
        self.assertNotIn("sign", payload)  # 无密钥 → 不带加签字段

    def test_parse_gpus(self):
        self.assertEqual(notify.parse_gpus("0,1"), [0, 1])
        self.assertEqual(notify.parse_gpus("0， 2"), [0, 2])
        self.assertEqual(notify.parse_gpus(""), [])
        with self.assertRaises(notify.NotifyError):
            notify.parse_gpus("0,gpu1")

    # ---------- 发送与重试（本地 HTTP webhook） ----------

    def test_send_end_to_end_feishu(self):
        HookHandler.behavior(200, '{"code":0,"msg":"success"}')
        notify.main(["set", "--machine-name", "lx-4090", "--url",
                     f"http://127.0.0.1:{self.port}/hook", "--runtime", str(self.runtime)])
        rc = notify.main(["send", "--event", "finish", "--gpus", "0",
                          "--experiment", "EXP-001", "--runtime", str(self.runtime)])
        self.assertEqual(rc, 0)
        self.assertEqual(len(HookHandler.records), 1)
        payload = HookHandler.records[0]
        self.assertEqual(payload["msg_type"], "interactive")
        self.assertIn("训练结束", payload["card"]["header"]["title"]["content"])

    def test_feishu_nonzero_code_retries_then_fails(self):
        HookHandler.behavior(200, '{"code":19021,"msg":"sign match fail"}')
        notify.main(["set", "--machine-name", "m", "--url",
                     f"http://127.0.0.1:{self.port}/hook", "--runtime", str(self.runtime)])
        rc = notify.main(["send", "--event", "start", "--runtime", str(self.runtime)])
        self.assertEqual(rc, 1)  # 失败可见，但不抛异常
        self.assertEqual(len(HookHandler.records), notify.ATTEMPTS)  # 尽力而为：恰好重试 1 次

    def test_http_500_retries_then_fails(self):
        HookHandler.behavior(500, "{}")
        notify.main(["set", "--machine-name", "m", "--url",
                     f"http://127.0.0.1:{self.port}/hook", "--runtime", str(self.runtime)])
        rc = notify.main(["send", "--event", "interrupt", "--note", "OOM",
                          "--runtime", str(self.runtime)])
        self.assertEqual(rc, 1)
        self.assertEqual(len(HookHandler.records), notify.ATTEMPTS)

    def test_generic_type_accepts_any_2xx(self):
        HookHandler.behavior(200, '{"ok":true,"ignored":"arbitrary"}')
        notify.main(["set", "--machine-name", "m", "--type", "generic", "--url",
                     f"http://127.0.0.1:{self.port}/hook", "--runtime", str(self.runtime)])
        rc = notify.main(["send", "--event", "start", "--gpus", "1",
                          "--runtime", str(self.runtime)])
        self.assertEqual(rc, 0)
        payload = HookHandler.records[0]
        self.assertEqual(payload["machine"], "m")
        self.assertEqual(payload["event"], "start")
        self.assertEqual(payload["gpus"], [1])

    def test_function_api_never_raises(self):
        # 任何故障路径（配置缺失/网络拒绝）都折叠为返回值，不向上抛
        r = notify.send_notification("start", runtime=str(self.runtime))
        self.assertFalse(r["sent"])
        self.assertIn("skipped", r)
        r = notify.send_notification("start", runtime=str(self.tmp / "no-such-runtime"))
        self.assertFalse(r["sent"])
        notify.main(["set", "--machine-name", "m", "--url", "http://127.0.0.1:1/hook",
                     "--runtime", str(self.runtime)])
        r = notify.send_notification("interrupt", gpus=[0], experiment="EXP-002",
                                     note="CUDA OOM", runtime=str(self.runtime))
        self.assertFalse(r["sent"])
        self.assertIn("error", r)

    def test_render_lines_has_required_fields(self):
        line = notify.render_lines("lx-4090", "0（RTX 4090）", "EXP-001", "原因 X", bold=False)
        for frag in ("机器：lx-4090", "GPU：0（RTX 4090）", "实验：EXP-001", "说明：原因 X"):
            self.assertIn(frag, line)


if __name__ == "__main__":
    unittest.main(verbosity=2)
