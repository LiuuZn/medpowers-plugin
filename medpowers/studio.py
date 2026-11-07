"""studio.py — 训练仓库与 MedPowers Studio 的对接客户端（零第三方依赖）。

口径（与 Studio 端契约一致）：
- Studio 是数据/标注的唯一事实源；训练仓库永远不改标注内容，只能申请复核
  （approved → review_requested）。
- git 中的 EXP-<NNN> 记录是实验事实的权威；Studio 只做实验/指标/评测结果的可视化镜像。
- token 按仓库粒度发放、跟随用户，像 API key 一样保管；token 泄漏 = 用户凭证泄漏。
- 推送是尽力而为：Studio 不可用时写入本地队列，训练/CLI 不因推送失败而中断。

常用：
    medpowers studio auth --url http://10.0.0.20:8000 --repo-key spine_locator
    medpowers studio status
    medpowers studio tasks
    medpowers studio task spine_keyframe
    medpowers studio exp push --exp EXP-012 --task spine_keyframe
    medpowers studio metrics push --exp EXP-012 --file metrics.json
    medpowers studio eval push --exp EXP-012 --file eval.json
    medpowers studio review request --annotation <id> --reason "边界可疑" --eval <id>
    medpowers studio sync-exp
"""
from __future__ import annotations

import argparse
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .notify import resolve_runtime

STUDIO_FILE = "studio.json"
TOKEN_FILE = "studio.token"
QUEUE_DIR = "studio-queue"
DEFAULT_SCOPES = ["studio:read", "studio:data", "studio:experiment", "studio:evaluate"]
SCHEMA_VERSION = "2026-11-07"
MAX_BATCH = 5000


class StudioError(Exception):
    """用户可见错误：配置缺失、网络失败、权限不足、服务端拒绝。"""


def _atomic_write(path: Path, text: str, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    if mode is not None:
        os.chmod(tmp, mode)
    os.replace(tmp, path)


def _default_config() -> dict[str, Any]:
    return {
        "base_url": "",
        "repo_key": "",
        "scopes": list(DEFAULT_SCOPES),
        "machine_name": "",
        "allow_insecure_http": False,
        "verify_tls": True,
        "ca_bundle": "",
        "configured_at": "",
    }


def config_path(runtime: Path) -> Path:
    return runtime / STUDIO_FILE


def token_path(runtime: Path) -> Path:
    return runtime / TOKEN_FILE


def load_config(runtime: str | Path | None = None) -> dict[str, Any] | None:
    """读配置；未配置返回 None（组件不生效）。token 从环境变量或 0600 文件读取。"""
    root = resolve_runtime(str(runtime) if runtime else None)
    path = config_path(root)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise StudioError(f"{path} 无法解析：{exc}（重新执行 studio auth）") from exc
    if not isinstance(raw, dict):
        raise StudioError(f"{path} 不是 JSON 对象")
    cfg = {**_default_config(), **raw}
    cfg["_runtime"] = str(root)
    token = os.environ.get("MEDPOWERS_STUDIO_TOKEN", "").strip()
    if not token:
        tp = token_path(root)
        if tp.exists():
            try:
                token = tp.read_text(encoding="utf-8").strip()
            except OSError as exc:
                raise StudioError(f"{tp} 读取失败：{exc}") from exc
    cfg["token"] = token
    return cfg


def save_config(runtime: Path, cfg: dict[str, Any], token: str = "") -> None:
    root = Path(runtime).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    data = {key: value for key, value in cfg.items() if not key.startswith("_") and key != "token"}
    data["configured_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
    _atomic_write(config_path(root), json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    if token:
        _atomic_write(token_path(root), token + "\n", mode=0o600)


def _ssl_context(cfg: dict[str, Any]) -> ssl.SSLContext | None:
    url = str(cfg.get("base_url", ""))
    if not url.startswith("https://"):
        return None
    ca_bundle = str(cfg.get("ca_bundle", "")).strip()
    if ca_bundle:
        return ssl.create_default_context(cafile=ca_bundle)
    if cfg.get("verify_tls", True):
        return ssl.create_default_context()
    return ssl._create_unverified_context()  # noqa: SLF001  内网自签证书显式豁免


class StudioClient:
    def __init__(
        self,
        base_url: str,
        token: str = "",
        *,
        repo_key: str = "",
        allow_insecure_http: bool = False,
        verify_tls: bool = True,
        ca_bundle: str = "",
        timeout: float = 30.0,
        retries: int = 2,
    ) -> None:
        base_url = base_url.strip().rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise StudioError("Studio 地址必须以 http:// 或 https:// 开头")
        if base_url.startswith("http://") and not allow_insecure_http:
            raise StudioError(
                "拒绝明文 HTTP：内网部署需在 studio.json 设置 allow_insecure_http=true，"
                "并确保仅内网受控访问"
            )
        self.base_url = base_url
        self.token = token.strip()
        self.repo_key = repo_key.strip()
        self.timeout = timeout
        self.retries = max(0, retries)
        self._ssl = _ssl_context(
            {"base_url": base_url, "verify_tls": verify_tls, "ca_bundle": ca_bundle}
        )

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "StudioClient":
        if not cfg.get("token"):
            raise StudioError("未找到 Studio token：先执行 medpowers studio auth")
        return cls(
            str(cfg.get("base_url", "")),
            str(cfg.get("token", "")),
            repo_key=str(cfg.get("repo_key", "")),
            allow_insecure_http=bool(cfg.get("allow_insecure_http", False)),
            verify_tls=bool(cfg.get("verify_tls", True)),
            ca_bundle=str(cfg.get("ca_bundle", "")),
        )

    def _url(self, path: str, params: dict[str, Any] | None = None) -> str:
        url = self.base_url + (path if path.startswith("/") else "/" + path)
        if params:
            query = {k: v for k, v in params.items() if v is not None and v != ""}
            if query:
                url += "?" + urllib.parse.urlencode(query)
        return url

    def _request(
        self,
        method: str,
        path: str,
        payload: Any = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        data = None
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json"
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(
                self._url(path, params), data=data, headers=headers, method=method
            )
            try:
                with urllib.request.urlopen(
                    request, timeout=self.timeout, context=self._ssl
                ) as response:
                    body = response.read()
                    if not body:
                        return None
                    return json.loads(body.decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                detail = body
                code = ""
                try:
                    parsed = json.loads(body)
                    if isinstance(parsed, dict):
                        detail = parsed.get("detail") or parsed.get("message") or body
                        code = str(parsed.get("code") or parsed.get("error") or "")
                except json.JSONDecodeError:
                    pass
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    time.sleep(min(2 ** attempt, 5))
                    continue
                hint = ""
                if exc.code in (401, 403):
                    hint = "；检查 token 是否过期/被撤销、scope 是否足够，必要时重新 studio auth"
                raise StudioError(
                    f"Studio 返回 {exc.code}：{detail}{hint}"
                    + (f"（{code}）" if code else "")
                ) from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(min(2 ** attempt, 5))
                    continue
        raise StudioError(f"连接 Studio 失败：{last_error}")

    # ---- 只读 ----
    def context(self) -> dict[str, Any]:
        return self._request("GET", "/api/auth/context")

    def meta(self) -> dict[str, Any]:
        return self._request("GET", "/api/studio/meta")

    def tasks(self) -> list[dict[str, Any]]:
        return self._request("GET", "/api/studio/tasks")

    def task_readiness(self, task_code: str) -> dict[str, Any]:
        return self._request(
            "GET", f"/api/studio/tasks/{urllib.parse.quote(task_code)}/readiness"
        )

    def changes(self, since: str | None = None) -> dict[str, Any]:
        return self._request("GET", "/api/studio/changes", params={"since": since})

    # ---- 实验与指标 ----
    def upsert_experiment(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/studio/experiments", payload)

    def update_experiment(
        self, exp_id: str, payload: dict[str, Any], repo_key: str = ""
    ) -> dict[str, Any]:
        return self._request(
            "PATCH",
            f"/api/studio/experiments/{urllib.parse.quote(exp_id)}",
            payload,
            params={"repo_key": repo_key or self.repo_key},
        )

    def get_experiment(self, exp_id: str, repo_key: str = "") -> dict[str, Any]:
        return self._request(
            "GET",
            f"/api/studio/experiments/{urllib.parse.quote(exp_id)}",
            params={"repo_key": repo_key or self.repo_key},
        )

    def push_metrics(
        self, exp_id: str, points: list[dict[str, Any]], repo_key: str = ""
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/api/studio/experiments/{urllib.parse.quote(exp_id)}/metrics:batch",
            {"points": points},
            params={"repo_key": repo_key or self.repo_key},
        )

    def get_metrics(self, exp_id: str, **params: Any) -> list[dict[str, Any]]:
        params.setdefault("repo_key", self.repo_key)
        return self._request(
            "GET", f"/api/studio/experiments/{urllib.parse.quote(exp_id)}/metrics", params=params
        )

    def push_evaluation(
        self, exp_id: str, payload: dict[str, Any], repo_key: str = ""
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/api/studio/experiments/{urllib.parse.quote(exp_id)}/evaluations",
            payload,
            params={"repo_key": repo_key or self.repo_key},
        )

    def list_evaluations(self, exp_id: str, repo_key: str = "") -> list[dict[str, Any]]:
        return self._request(
            "GET",
            f"/api/studio/experiments/{urllib.parse.quote(exp_id)}/evaluations",
            params={"repo_key": repo_key or self.repo_key},
        )

    def task_experiments(self, task_code: str, **params: Any) -> list[dict[str, Any]]:
        return self._request(
            "GET",
            f"/api/studio/tasks/{urllib.parse.quote(task_code)}/experiments",
            params=params or None,
        )

    # ---- 标注复核 ----
    def request_review(
        self,
        annotation_id: str,
        reason: str,
        *,
        evidence: dict[str, Any] | None = None,
        eval_id: str | None = None,
    ) -> dict[str, Any]:
        return self._request(
            "POST",
            f"/api/annotations/{urllib.parse.quote(annotation_id)}/request-review",
            {"reason": reason, "evidence": evidence or {}, "eval_id": eval_id},
        )

    def review_requests(self, status: str = "open", task_code: str = "") -> list[dict[str, Any]]:
        return self._request(
            "GET",
            "/api/studio/review-requests",
            params={"status": status, "task_code": task_code or None},
        )


# ---------- 离线队列 ----------

def queue_dir(runtime: Path, repo_key: str) -> Path:
    safe = "".join(ch for ch in (repo_key or "default") if ch.isalnum() or ch in "._-")
    return Path(runtime) / QUEUE_DIR / (safe or "default")


def _enqueue(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(item, ensure_ascii=False) + "\n")


def _replay(client: StudioClient, item: dict[str, Any]) -> None:
    kind = item.get("kind")
    exp_id = str(item.get("exp_id") or "")
    payload = item.get("payload") or {}
    if kind == "experiment":
        client.upsert_experiment(payload)
    elif kind == "experiment_patch":
        client.update_experiment(exp_id, payload, str(item.get("repo_key") or ""))
    elif kind == "metrics":
        points = payload.get("points") if isinstance(payload, dict) else payload
        client.push_metrics(exp_id, list(points or []), str(item.get("repo_key") or ""))
    elif kind == "evaluation":
        client.push_evaluation(exp_id, payload, str(item.get("repo_key") or ""))
    else:
        raise StudioError(f"未知队列事件：{kind}")


def flush_queue(
    client: StudioClient,
    runtime: Path,
    repo_key: str,
    *,
    log: Callable[[str], None] = print,
) -> int:
    """重放本地队列；返回成功重放的条数。失败的行保留，下次再试。"""
    directory = queue_dir(runtime, repo_key)
    if not directory.is_dir():
        return 0
    sent = 0
    for path in sorted(directory.glob("*.jsonl")):
        lines = path.read_text(encoding="utf-8").splitlines()
        remaining: list[str] = []
        failed = False
        for line in lines:
            if failed or not line.strip():
                remaining.append(line)
                continue
            try:
                _replay(client, json.loads(line))
                sent += 1
            except (StudioError, json.JSONDecodeError) as exc:
                failed = True
                remaining.append(line)
                log(f"[警告] 队列事件重放失败，保留待下次：{exc}")
        if remaining:
            _atomic_write(path, "\n".join(remaining) + "\n")
        else:
            path.unlink(missing_ok=True)
    return sent


class ExperimentReporter:
    """训练进程内的实验/指标上报器：尽力而为，失败进本地队列，绝不影响训练。"""

    def __init__(
        self,
        client: StudioClient,
        exp_id: str,
        task_code: str,
        *,
        runtime: str | Path | None = None,
        repo_key: str = "",
        machine: str = "",
        title: str = "",
        git_commit: str = "",
        strict: bool = False,
        log: Callable[[str], None] = print,
    ) -> None:
        self.client = client
        self.exp_id = exp_id
        self.task_code = task_code
        self.repo_key = repo_key or client.repo_key
        self.runtime = resolve_runtime(str(runtime) if runtime else None)
        self.machine = machine
        self.title = title
        self.git_commit = git_commit
        self.strict = strict
        self.log = log
        self._buffer: list[dict[str, Any]] = []
        self._queue = queue_dir(self.runtime, self.repo_key) / f"{exp_id}.jsonl"

    def _fail(self, item: dict[str, Any], exc: Exception) -> None:
        _enqueue(self._queue, item)
        self.log(f"[警告] Studio 上报失败，已进本地队列：{exc}")
        if self.strict:
            raise StudioError(str(exc)) from exc

    def _run(self, item: dict[str, Any], fn: Callable[[], Any]) -> Any:
        try:
            return fn()
        except StudioError as exc:
            self._fail(item, exc)
            return None

    def start(self, config: dict[str, Any] | None = None, status: str = "running") -> dict | None:
        payload = {
            "exp_id": self.exp_id,
            "task_code": self.task_code,
            "status": status,
            "title": self.title,
            "machine": self.machine,
            "git_commit": self.git_commit,
            "config": config or {},
            "repo_key": self.repo_key,
        }
        return self._run(
            {"kind": "experiment", "exp_id": self.exp_id, "payload": payload},
            lambda: self.client.upsert_experiment(payload),
        )

    def log_metrics(self, points: list[dict[str, Any]]) -> None:
        self._buffer.extend(points)
        if len(self._buffer) >= 200:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        points, self._buffer = self._buffer, []
        for start in range(0, len(points), MAX_BATCH):
            batch = points[start : start + MAX_BATCH]
            self._run(
                {
                    "kind": "metrics",
                    "exp_id": self.exp_id,
                    "repo_key": self.repo_key,
                    "payload": {"points": batch},
                },
                lambda batch=batch: self.client.push_metrics(
                    self.exp_id, batch, self.repo_key
                ),
            )

    def finish(
        self,
        status: str,
        *,
        notes: str = "",
        record_path: str = "",
        record_sha256: str = "",
    ) -> dict | None:
        self.flush()
        payload: dict[str, Any] = {"status": status}
        if notes:
            payload["notes"] = notes
        if record_path:
            payload["record_path"] = record_path
        if record_sha256:
            payload["record_sha256"] = record_sha256
        return self._run(
            {
                "kind": "experiment_patch",
                "exp_id": self.exp_id,
                "repo_key": self.repo_key,
                "payload": payload,
            },
            lambda: self.client.update_experiment(self.exp_id, payload, self.repo_key),
        )


# ---------- 设备码授权 ----------

def _post_json(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    *,
    allow_insecure_http: bool,
    timeout: float = 30.0,
) -> tuple[int, dict[str, Any]]:
    if base_url.startswith("http://") and not allow_insecure_http:
        raise StudioError("拒绝明文 HTTP：请显式加 --allow-insecure-http 并确保内网受控")
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = {"detail": body}
        return exc.code, parsed


def device_auth(
    *,
    base_url: str,
    repo_key: str,
    client_name: str = "medpowers",
    scopes: list[str] | None = None,
    allow_insecure_http: bool = False,
    runtime: str | Path | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    scopes = scopes or list(DEFAULT_SCOPES)
    status, body = _post_json(
        base_url,
        "/api/auth/device/code",
        {"client_name": client_name, "repo_key": repo_key, "scopes": scopes},
        allow_insecure_http=allow_insecure_http,
    )
    if status != 200:
        raise StudioError(f"设备码申请失败（{status}）：{body.get('detail', body)}")
    device_code = body["device_code"]
    log("请在浏览器打开以下地址并批准授权：")
    log(f"  {body.get('verification_uri_complete') or body.get('verification_uri')}")
    log(f"  用户码：{body.get('user_code')}")
    log(f"  仓库：{repo_key or '(未命名)'}  scope：{' '.join(scopes)}")
    interval = max(1, int(body.get("interval", 5)))
    expires = time.monotonic() + int(body.get("expires_in", 600))
    while time.monotonic() < expires:
        time.sleep(interval)
        status, token_body = _post_json(
            base_url,
            "/api/auth/device/token",
            {"device_code": device_code},
            allow_insecure_http=allow_insecure_http,
        )
        if status == 200:
            root = resolve_runtime(str(runtime) if runtime else None)
            granted = str(token_body.get("scope") or "").split() or list(scopes)
            cfg = {
                **_default_config(),
                "base_url": base_url.rstrip("/"),
                "repo_key": token_body.get("repo_key") or repo_key,
                "scopes": granted,
                "allow_insecure_http": allow_insecure_http,
            }
            save_config(root, cfg, token=str(token_body["access_token"]))
            log(f"授权完成，token 已写入 {token_path(root)}（0600）")
            return cfg
        error = str(token_body.get("error") or "")
        if error == "authorization_pending":
            continue
        if error == "slow_down":
            interval += 5
            continue
        raise StudioError(
            f"设备授权失败：{token_body.get('error_description') or token_body.get('detail') or error}"
        )
    raise StudioError("设备授权超时，请重新执行 studio auth")


# ---------- CLI ----------

def _client(args) -> StudioClient:
    cfg = load_config(getattr(args, "runtime", None))
    if cfg is None:
        raise StudioError("尚未配置 Studio：先执行 medpowers studio auth --url ... --repo-key ...")
    return StudioClient.from_config(cfg)


def _load_json_file(path: str) -> Any:
    if path == "-":
        return json.loads(sys.stdin.read())
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cmd_auth(args) -> int:
    device_auth(
        base_url=args.url,
        repo_key=args.repo_key,
        client_name=args.client_name,
        scopes=args.scopes,
        allow_insecure_http=args.allow_insecure_http,
        runtime=args.runtime,
    )
    return 0


def cmd_status(args) -> int:
    cfg = load_config(args.runtime)
    if cfg is None:
        print("Studio：未配置（medpowers studio auth 开始授权）")
        return 0
    client = StudioClient.from_config(cfg)
    context = client.context()
    meta = client.meta()
    print(f"Studio：{cfg.get('base_url')}  仓库：{context.get('repo_key') or cfg.get('repo_key')}")
    print(f"用户：{context.get('user', {}).get('username')}  认证：{context.get('auth_method')}")
    print(f"scope：{' '.join(context.get('scopes') or [])}")
    caps = context.get("capabilities") or {}
    print("能力：" + " ".join(key for key, value in caps.items() if value))
    print(f"schema：{meta.get('schema_version')}  api：{meta.get('version')}")
    return 0


def cmd_tasks(args) -> int:
    for task in _client(args).tasks():
        flag = "✓" if task.get("annotation_contract_complete") else "!"
        print(
            f"{task['code']}\t{task['task_type']}\tapproved={task.get('approved', 0)}"
            f"\treview={task.get('review_requested', 0)}\tcases={task.get('total_cases', 0)}"
            f"\t契约{flag}"
        )
    return 0


def cmd_task(args) -> int:
    data = _client(args).task_readiness(args.task)
    task = data["task"]
    print(f"任务/模型：{task['code']}（{task['task_type']}）")
    print(f"病例：{data['total_cases']}  数据集：{len(data.get('datasets', []))}")
    print("标注状态：" + " ".join(f"{k}={v}" for k, v in data.get("counts", {}).items()))
    print(f"标注契约：{'完整' if data.get('annotation_contract_complete') else '未完整'}")
    if data.get("class_case_counts"):
        print("逐类病例：" + " ".join(f"{k}={v}" for k, v in data["class_case_counts"].items()))
    for issue in data.get("qc_issues", []):
        print(f"[QC] {issue.get('message')}：{issue.get('count')}")
    return 0


def cmd_exp_push(args) -> int:
    config = _load_json_file(args.config) if args.config else None
    payload = {
        "exp_id": args.exp,
        "task_code": args.task,
        "status": args.status,
        "title": args.title,
        "machine": args.machine,
        "git_commit": args.git_commit,
        "config": config or {},
        "repo_key": args.repo_key,
    }
    result = _client(args).upsert_experiment(payload)
    print(f"{result['exp_id']} 已镜像到 Studio（status={result['status']}）")
    return 0


def cmd_metrics_push(args) -> int:
    raw = _load_json_file(args.file)
    points = raw.get("points", raw) if isinstance(raw, dict) else raw
    if not isinstance(points, list):
        raise StudioError("指标文件应为数组或 {\"points\": [...]}")
    client = _client(args)
    accepted = 0
    for start in range(0, len(points), MAX_BATCH):
        result = client.push_metrics(args.exp, points[start : start + MAX_BATCH], args.repo_key)
        accepted += int(result.get("accepted", 0))
    print(f"已推送 {accepted} 个指标点到 {args.exp}")
    return 0


def cmd_eval_push(args) -> int:
    payload = _load_json_file(args.file)
    if not isinstance(payload, dict):
        raise StudioError("评测文件应为 JSON 对象")
    result = _client(args).push_evaluation(args.exp, payload, args.repo_key)
    print(f"评测结果 {result['eval_id']} 已推送到 {args.exp}")
    return 0


def cmd_review_request(args) -> int:
    evidence = _load_json_file(args.evidence) if args.evidence else {}
    result = _client(args).request_review(
        args.annotation, args.reason, evidence=evidence, eval_id=args.eval
    )
    print(f"复核申请已提交：annotation={result.get('id')} status={result.get('status')}")
    return 0


def cmd_sync_exp(args) -> int:
    client = _client(args)
    cfg = load_config(args.runtime) or {}
    runtime = Path(cfg.get("_runtime") or resolve_runtime(args.runtime))
    sent = flush_queue(client, runtime, str(cfg.get("repo_key", "")))
    print(f"本地队列已重放 {sent} 条事件")
    return 0


def _add_runtime(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--runtime", help="runtime 目录（默认 $MEDPOWERS_RUNTIME 或 <工作区根>/runtime）"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MedPowers Studio 训练仓库对接")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("auth", help="设备码授权并保存 token")
    p.add_argument("--url", required=True, help="Studio 地址，如 http://10.0.0.20:8000")
    p.add_argument("--repo-key", required=True, help="训练仓库标识（同一仓库多机共用）")
    p.add_argument("--client-name", default="medpowers", help="设备名称（默认 medpowers）")
    p.add_argument(
        "--scopes",
        nargs="+",
        default=list(DEFAULT_SCOPES),
        help="申请的 scope（默认 read/data/experiment/evaluate）",
    )
    p.add_argument(
        "--allow-insecure-http",
        action="store_true",
        help="显式允许内网明文 HTTP（有泄露 token 风险）",
    )
    _add_runtime(p)
    p.set_defaults(fn=cmd_auth)

    p = sub.add_parser("status", help="查看连接、身份与能力")
    _add_runtime(p)
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("tasks", help="列出任务（模型）与就绪摘要")
    _add_runtime(p)
    p.set_defaults(fn=cmd_tasks)

    p = sub.add_parser("task", help="查看单个任务/模型就绪报告")
    p.add_argument("task", help="任务 code")
    _add_runtime(p)
    p.set_defaults(fn=cmd_task)

    p = sub.add_parser("exp", help="实验镜像")
    exp_sub = p.add_subparsers(dest="exp_cmd", required=True)
    q = exp_sub.add_parser("push", help="推送/更新实验镜像")
    q.add_argument("--exp", required=True, help="EXP-<NNN>")
    q.add_argument("--task", required=True, help="Studio 任务 code（= 模型）")
    q.add_argument("--status", default="running", choices=["planned", "running", "done", "interrupted", "cancelled"])
    q.add_argument("--title", default="")
    q.add_argument("--machine", default="")
    q.add_argument("--git-commit", default="")
    q.add_argument("--config", help="配置 JSON 文件路径")
    q.add_argument("--repo-key", default="")
    _add_runtime(q)
    q.set_defaults(fn=cmd_exp_push)

    p = sub.add_parser("metrics", help="指标推送")
    metrics_sub = p.add_subparsers(dest="metrics_cmd", required=True)
    q = metrics_sub.add_parser("push", help="批量推送指标点")
    q.add_argument("--exp", required=True)
    q.add_argument("--file", required=True, help="JSON 文件或 -（stdin）")
    q.add_argument("--repo-key", default="")
    _add_runtime(q)
    q.set_defaults(fn=cmd_metrics_push)

    p = sub.add_parser("eval", help="评测结果")
    eval_sub = p.add_subparsers(dest="eval_cmd", required=True)
    q = eval_sub.add_parser("push", help="推送外部评测结果")
    q.add_argument("--exp", required=True)
    q.add_argument("--file", required=True, help="JSON 文件或 -（stdin）")
    q.add_argument("--repo-key", default="")
    _add_runtime(q)
    q.set_defaults(fn=cmd_eval_push)

    p = sub.add_parser("review", help="标注复核申请")
    review_sub = p.add_subparsers(dest="review_cmd", required=True)
    q = review_sub.add_parser("request", help="申请人工复核（approved → review_requested）")
    q.add_argument("--annotation", required=True, help="标注 ID")
    q.add_argument("--reason", required=True, help="复核原因")
    q.add_argument("--eval", help="相关评测 ID")
    q.add_argument("--evidence", help="证据 JSON 文件")
    _add_runtime(q)
    q.set_defaults(fn=cmd_review_request)

    p = sub.add_parser("sync-exp", help="重放本地离线队列")
    _add_runtime(p)
    p.set_defaults(fn=cmd_sync_exp)

    args = ap.parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except StudioError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
