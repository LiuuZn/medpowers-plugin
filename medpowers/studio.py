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

数据集接入（把本地影像推上 Studio 建库；需 scope studio:dataset-manage + 管理员账号）：
    medpowers studio datasets
    medpowers studio protocols
    medpowers studio protocol create --code mr_routine --name "MR 常规" --modality MR   # 仅当预置流程均不适用时；命名「部位 模态 扫描」
    medpowers studio dataset create --code mr_knee --name "膝关节 MR" --modality MR --protocol mr_routine
    medpowers studio dataset import --dataset mr_knee --upload /data/knee_mr/   # 目录自动打 ZIP 上传
    medpowers studio dataset import --dataset mr_knee --server-path /imports/knee_mr --async --wait
    medpowers studio dataset ready --dataset mr_knee
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import ssl
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .notify import resolve_runtime

STUDIO_FILE = "studio.json"
TOKEN_FILE = "studio.token"
QUEUE_DIR = "studio-queue"
DEFAULT_SCOPES = ["studio:read", "studio:data", "studio:experiment", "studio:evaluate"]
# 数据集接入（新建/导入/标记就绪/新建扫描流程）：只增不删，且要求令牌所属账号为管理员。
DATASET_MANAGE_SCOPE = "studio:dataset-manage"
DATASET_MODALITIES = ["DR", "CT", "MR", "TABLE"]
SCHEMA_VERSION = "2026-11-07"
MAX_BATCH = 5000
DEFAULT_MAX_UPLOAD_MB = 4096  # 与 Studio 端 MEDPOWERS_MAX_UPLOAD_MB 默认值一致


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


class _MultipartBody:
    """流式 multipart/form-data 体：文件内容不整读进内存，支持 GB 级上传。

    read() 依序返回 前置表单+文件头 → 文件内容分块 → 结束边界；总长度由
    _multipart_length() 预先算好，作为显式 Content-Length 传给 urllib。
    """

    def __init__(self, boundary: str, fields: dict[str, str], file_field: str,
                 file_path: Path, filename: str) -> None:
        self._chunks: list[bytes] = []
        crlf = b"\r\n"
        for name, value in fields.items():
            self._chunks.append(
                f"--{boundary}{crlf.decode()}Content-Disposition: form-data; "
                f'name="{name}"{crlf.decode()}{crlf.decode()}{value}{crlf.decode()}'.encode("utf-8")
            )
        self._chunks.append(
            (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; "
                f"filename=\"{filename}\"\r\nContent-Type: application/zip\r\n\r\n"
            ).encode("utf-8")
        )
        self._file = open(file_path, "rb")
        self._footer = f"\r\n--{boundary}--\r\n".encode("utf-8")
        self._file_done = False
        self._footer_sent = False

    def length(self) -> int:
        size = sum(len(chunk) for chunk in self._chunks) + len(self._footer)
        size += self._file.seek(0, 2)
        self._file.seek(0)  # 复位到文件头：length 计算后 read() 要从头读内容
        return size

    def read(self, size: int = -1) -> bytes:
        while self._chunks:
            chunk = self._chunks[0]
            if size < 0 or len(chunk) <= size:
                self._chunks.pop(0)
                return chunk
            self._chunks[0] = chunk[size:]
            return chunk[:size]
        if not self._file_done:
            data = self._file.read(size if size > 0 else 1024 * 1024)
            if data:
                return data
            self._file.close()
            self._file_done = True
        if not self._footer_sent:
            self._footer_sent = True
            return self._footer
        return b""


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

    def _request_multipart(
        self,
        path: str,
        file_path: Path,
        filename: str,
        fields: dict[str, str],
        timeout: float,
    ) -> Any:
        """multipart/form-data 上传（import-upload）。大文件流式读取，不做自动重试。"""
        boundary = uuid.uuid4().hex
        body = _MultipartBody(boundary, fields, "file", file_path, filename)
        headers = {
            "Accept": "application/json",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(body.length()),
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(
            self._url(path), data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(
                request, timeout=timeout, context=self._ssl
            ) as response:
                raw = response.read()
                return json.loads(raw.decode("utf-8")) if raw else None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(detail)
                if isinstance(parsed, dict):
                    detail = str(parsed.get("detail") or parsed.get("message") or detail)
            except json.JSONDecodeError:
                pass
            hint = ""
            if exc.code in (401, 403):
                hint = "；数据集接入需要 scope studio:dataset-manage 且令牌所属账号为管理员"
            raise StudioError(f"Studio 返回 {exc.code}：{detail}{hint}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise StudioError(f"上传失败：{exc}") from exc

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

    # ---- 数据集接入（studio:dataset-manage；只增不删） ----
    def scan_protocols(self) -> list[dict[str, Any]]:
        return self._request("GET", "/api/scan-protocols")

    def create_scan_protocol(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/scan-protocols", payload)

    def datasets(self, **params: Any) -> list[dict[str, Any]]:
        return self._request("GET", "/api/datasets", params=params or None)

    def dataset(self, dataset_id: str) -> dict[str, Any]:
        return self._request("GET", f"/api/datasets/{urllib.parse.quote(dataset_id)}")

    def create_dataset(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/api/datasets", payload)

    def dataset_ready(self, dataset_id: str) -> dict[str, Any]:
        return self._request("POST", f"/api/datasets/{urllib.parse.quote(dataset_id)}/ready")

    def import_path(
        self,
        dataset_id: str,
        path: str,
        *,
        recursive: bool = True,
        acknowledge_source_discarded: bool = False,
    ) -> dict[str, Any]:
        """服务端本地路径导入（同步）：path 必须是 Studio 容器内可见路径。"""
        return self._request(
            "POST",
            f"/api/datasets/{urllib.parse.quote(dataset_id)}/import-path",
            {
                "path": path,
                "recursive": recursive,
                "acknowledge_source_discarded": acknowledge_source_discarded,
            },
        )

    def import_upload(
        self,
        dataset_id: str,
        file_path: str | Path,
        *,
        filename: str = "",
        acknowledge_source_discarded: bool = False,
        timeout: float = 1800.0,
    ) -> dict[str, Any]:
        """ZIP 包上传导入（同步）。file_path 必须是 .zip（用 zip_for_upload 打包）。"""
        path = Path(file_path)
        return self._request_multipart(
            f"/api/datasets/{urllib.parse.quote(dataset_id)}/import-upload",
            path,
            filename or path.name,
            {"acknowledge_source_discarded": "true" if acknowledge_source_discarded else "false"},
            timeout=timeout,
        )

    def import_jobs(self, dataset_id: str) -> list[dict[str, Any]]:
        return self._request(
            "GET", f"/api/datasets/{urllib.parse.quote(dataset_id)}/import-jobs"
        )

    def import_job(self, dataset_id: str, job_id: str) -> dict[str, Any]:
        return self._request(
            "GET",
            f"/api/datasets/{urllib.parse.quote(dataset_id)}/import-jobs/"
            f"{urllib.parse.quote(job_id)}",
        )

    def create_import_job(
        self,
        dataset_id: str,
        path: str,
        *,
        recursive: bool = True,
        acknowledge_source_discarded: bool = False,
    ) -> dict[str, Any]:
        """服务端本地路径导入（后台任务）。服务端要求 acknowledge_source_discarded=true。"""
        return self._request(
            "POST",
            f"/api/datasets/{urllib.parse.quote(dataset_id)}/import-jobs",
            {
                "path": path,
                "recursive": recursive,
                "acknowledge_source_discarded": acknowledge_source_discarded,
            },
        )

    def wait_import_job(
        self,
        dataset_id: str,
        job_id: str,
        *,
        timeout: float = 3600.0,
        poll: float = 5.0,
        log: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        """轮询导入任务直到 succeeded/failed；超时抛错（任务仍在服务端继续）。"""
        deadline = time.monotonic() + timeout
        emit = log or (lambda _: None)
        last_status = ""
        while True:
            job = self.import_job(dataset_id, job_id)
            status = str(job.get("status") or "")
            if status != last_status:
                emit(f"导入任务 {job_id}：{status}")
                last_status = status
            if status in ("succeeded", "failed"):
                return job
            if time.monotonic() > deadline:
                raise StudioError(
                    f"等待导入任务超时（>{timeout:g}s）；任务仍在服务端执行，"
                    f"稍后用 dataset jobs --dataset {dataset_id} 查询"
                )
            time.sleep(max(1.0, poll))


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


# ---------- 数据打包（import-upload 只收 ZIP） ----------

def zip_for_upload(source: Path, dest_dir: str | Path = "") -> tuple[Path, bool]:
    """把数据源打包成 Studio import-upload 接受的 ZIP。

    已是 .zip 的文件原样返回（不重复打包）；目录/单文件打包为临时 ZIP（跳过
    隐藏文件与 __MACOSX，与服务端发现逻辑一致）。返回 (zip 路径, 是否临时文件)，
    临时文件由调用方负责删除。
    """
    source = Path(source).expanduser()
    if source.is_file() and source.suffix.lower() == ".zip":
        return source, False
    target_dir = Path(dest_dir).expanduser() if dest_dir else Path(tempfile.gettempdir())
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / f"mp-upload-{uuid.uuid4().hex[:12]}-{source.name or 'data'}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        if source.is_file():
            zf.write(source, source.name)
            return zip_path, True
        if not source.is_dir():
            raise StudioError(f"数据源不存在：{source}")
        for path in sorted(source.rglob("*")):
            if not path.is_file():
                continue
            if path.name.startswith(".") or "__MACOSX" in path.parts:
                continue
            zf.write(path, path.relative_to(source))
    return zip_path, True


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


# ---- 数据集接入命令 ----

_DATASET_PERM_HINT = (
    "数据集接入需要 scope studio:dataset-manage 且令牌所属账号为管理员；"
    "重新授权：medpowers studio auth --url <地址> --repo-key <仓库> "
    f"--scopes <原scope...> {DATASET_MANAGE_SCOPE} [--allow-insecure-http]"
)


def _dataset_guard(fn: Callable[[Any], int]) -> Callable[[Any], int]:
    """给 403 补上数据集接入的专属提示。"""

    def wrapper(args) -> int:
        try:
            return fn(args)
        except StudioError as exc:
            if " 403" in str(exc):
                raise StudioError(f"{exc}\n提示：{_DATASET_PERM_HINT}") from exc
            raise

    return wrapper


def _resolve_dataset_id(client: StudioClient, ref: str) -> str:
    """数据集引用支持 id 或 code：先按 id 直查，404 再按 code 匹配。"""
    try:
        return str(client.dataset(ref).get("id"))
    except StudioError as exc:
        if " 404" not in str(exc):
            raise
    for item in client.datasets():
        if item.get("code") == ref:
            return str(item["id"])
    raise StudioError(f"数据集不存在（id 或 code 都未命中）：{ref}")


def _resolve_protocol_id(client: StudioClient, ref: str, modality: str) -> str:
    """扫描流程引用支持 id 或 code；未命中时列出可选流程。"""
    protocols = client.scan_protocols()
    for item in protocols:
        if item.get("id") == ref or item.get("code") == ref:
            if modality in (item.get("modality_scope") or []):
                return str(item["id"])
            raise StudioError(
                f"扫描流程 {ref} 不适用于模态 {modality}（适用："
                f"{', '.join(item.get('modality_scope') or []) or '不限'}）"
            )
    available = "；".join(
        f"{item.get('code')}（{item.get('name')}，适用 {', '.join(item.get('modality_scope') or [])}）"
        for item in protocols
    )
    raise StudioError(f"扫描流程不存在：{ref}。现有流程：{available or '（无，先 protocol create）'}")


def _print_import_report(report: dict[str, Any]) -> int:
    print(
        f"导入完成：新增 {report.get('imported', 0)}，跳过 {report.get('skipped', 0)}，"
        f"转换 {report.get('converted', 0)}，存储 {report.get('stored_bytes', 0)} 字节"
    )
    cases = report.get("cases") or []
    if cases:
        shown = ", ".join(cases[:10]) + ("…" if len(cases) > 10 else "")
        print(f"病例（{len(cases)}）：{shown}")
    for warning in report.get("warnings") or []:
        print(f"[警告] {warning}")
    errors = report.get("errors") or []
    for issue in errors[:20]:
        print(f"[错误] {issue.get('source')}: {issue.get('message')}")
    if len(errors) > 20:
        print(f"…另有 {len(errors) - 20} 条错误未展开")
    return 1 if errors else 0


def _print_job(job: dict[str, Any], prefix: str = "") -> None:
    report = job.get("report") or {}
    line = (
        f"{prefix}任务 {job.get('id')}  status={job.get('status')}  "
        f"导入 {job.get('imported', 0)}  跳过 {job.get('skipped', 0)}"
    )
    print(line)
    error = job.get("error")
    if error:
        print(f"{prefix}错误：{error}")
    for issue in (report.get("errors") or [])[:10]:
        print(f"{prefix}[错误] {issue.get('source')}: {issue.get('message')}")


@_dataset_guard
def cmd_datasets(args) -> int:
    rows = _client(args).datasets(
        modality=args.modality or None,
        status=args.status or None,
        search=args.search or None,
    )
    if not rows:
        print("（无数据集）")
        return 0
    for item in rows:
        print(
            f"{item.get('id')}\t{item.get('code')}\t{item.get('name')}\t"
            f"{item.get('modality')}\t{item.get('status')}\t"
            f"病例={item.get('case_count', 0)}\t患者={item.get('patient_count', 0)}"
        )
    return 0


@_dataset_guard
def cmd_protocols(args) -> int:
    rows = _client(args).scan_protocols()
    if not rows:
        print("（无扫描流程：平台应已预置标准流程，若确无适用项再用 protocol create 创建）")
        return 0
    for item in rows:
        print(
            f"{item.get('id')}\t{item.get('code')}\t{item.get('name')}\t"
            f"适用={','.join(item.get('modality_scope') or []) or '不限'}\t"
            f"{'启用' if item.get('is_active') else '停用'}"
        )
    return 0


@_dataset_guard
def cmd_protocol_create(args) -> int:
    result = _client(args).create_scan_protocol(
        {
            "code": args.code,
            "name": args.name,
            "description": args.description or "",
            "modality_scope": args.modality,
        }
    )
    print(f"扫描流程已创建：{result.get('code')}（id={result.get('id')}）")
    return 0


@_dataset_guard
def cmd_dataset_create(args) -> int:
    client = _client(args)
    protocol_id = _resolve_protocol_id(client, args.protocol, args.modality)
    result = client.create_dataset(
        {
            "code": args.code,
            "name": args.name,
            "description": args.description or "",
            "modality": args.modality,
            "scan_protocol_id": protocol_id,
        }
    )
    print(
        f"数据集已创建：{result.get('code')}（id={result.get('id')}，status={result.get('status')}）"
    )
    print("下一步：dataset import 导入病例，全部导入并核对报告后 dataset ready")
    return 0


@_dataset_guard
def cmd_dataset_show(args) -> int:
    client = _client(args)
    item = client.dataset(_resolve_dataset_id(client, args.dataset))
    protocol = item.get("scan_protocol") or {}
    print(
        f"{item.get('code')}（{item.get('name')}）  id={item.get('id')}  "
        f"modality={item.get('modality')}  {item.get('dimension')}/{item.get('data_kind')}  "
        f"status={item.get('status')}"
    )
    print(
        f"病例={item.get('case_count', 0)}  患者={item.get('patient_count', 0)}  "
        f"检查={item.get('study_count', 0)}  流程={protocol.get('code', '?')}"
    )
    if item.get("description"):
        print(f"描述：{item['description']}")
    for job in client.import_jobs(str(item["id"]))[:5]:
        _print_job(job, prefix="  最近导入 ")
    return 0


@_dataset_guard
def cmd_dataset_import(args) -> int:
    if not args.upload and not args.server_path:
        raise StudioError("import 需要 --upload <本地路径> 或 --server-path <服务端可见路径> 之一")
    if args.upload and args.server_path:
        raise StudioError("--upload 与 --server-path 只能二选一")
    client = _client(args)
    dataset_id = _resolve_dataset_id(client, args.dataset)

    if args.upload:
        source = Path(args.upload).expanduser()
        if not source.exists():
            raise StudioError(f"数据源不存在：{source}")
        zip_path, is_temp = zip_for_upload(source, args.zip_dir or "")
        try:
            size_mb = zip_path.stat().st_size / (1024 * 1024)
            if size_mb > args.max_upload_mb:
                raise StudioError(
                    f"ZIP {size_mb:.0f}MB 超过上限 {args.max_upload_mb}MB：拆分数据，"
                    f"或调大服务端 MEDPOWERS_MAX_UPLOAD_MB，或改用 --server-path"
                    f"（数据放进 Studio 容器可见路径，如部署约定的 /imports）"
                )
            print(f"上传 {zip_path.name}（{size_mb:.1f}MB）…")
            report = client.import_upload(
                dataset_id,
                zip_path,
                acknowledge_source_discarded=args.confirm_backup,
                timeout=args.timeout,
            )
        finally:
            if is_temp:
                zip_path.unlink(missing_ok=True)
        return _print_import_report(report)

    # --server-path：同步导入或后台任务
    path = args.server_path
    if args.async_job:
        if not args.confirm_backup:
            raise StudioError(
                "后台导入任务要求显式 --confirm-backup（服务端要求确认已自行备份原始数据）"
            )
        job = client.create_import_job(
            dataset_id, path, recursive=not args.no_recursive, acknowledge_source_discarded=True
        )
        print(f"后台导入任务已创建：{job.get('id')}（status={job.get('status')}）")
        if args.wait:
            job = client.wait_import_job(
                dataset_id, str(job.get("id")), timeout=args.timeout, log=print
            )
        _print_job(job)
        return 1 if job.get("status") == "failed" or job.get("error") else 0
    report = client.import_path(
        dataset_id,
        path,
        recursive=not args.no_recursive,
        acknowledge_source_discarded=args.confirm_backup,
    )
    return _print_import_report(report)


@_dataset_guard
def cmd_dataset_jobs(args) -> int:
    client = _client(args)
    dataset_id = _resolve_dataset_id(client, args.dataset)
    if args.job:
        job = client.import_job(dataset_id, args.job)
        if args.wait:
            job = client.wait_import_job(dataset_id, args.job, timeout=args.timeout, log=print)
        _print_job(job)
        return 1 if job.get("status") == "failed" or job.get("error") else 0
    jobs = client.import_jobs(dataset_id)
    if not jobs:
        print("（该数据集暂无导入任务）")
        return 0
    for job in jobs:
        _print_job(job)
    return 0


@_dataset_guard
def cmd_dataset_ready(args) -> int:
    client = _client(args)
    dataset_id = _resolve_dataset_id(client, args.dataset)
    item = client.dataset_ready(dataset_id)
    print(
        f"数据集 {item.get('code')} 已标记就绪（status={item.get('status')}，"
        f"病例={item.get('case_count', 0)}）"
    )
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

    # ---- 数据集接入（scope studio:dataset-manage + 管理员账号） ----
    p = sub.add_parser("datasets", help="列出数据集")
    p.add_argument("--modality", help="按模态过滤（DR/CT/MR/TABLE）")
    p.add_argument("--status", help="按状态过滤（draft/ready/archived）")
    p.add_argument("--search", help="按名称/编码模糊搜索")
    _add_runtime(p)
    p.set_defaults(fn=cmd_datasets)

    p = sub.add_parser("protocols", help="列出扫描流程")
    _add_runtime(p)
    p.set_defaults(fn=cmd_protocols)

    p = sub.add_parser("protocol", help="扫描流程管理")
    protocol_sub = p.add_subparsers(dest="protocol_cmd", required=True)
    q = protocol_sub.add_parser(
        "create", help="新建扫描流程（优先用平台预置流程；仅当预置均不适用时新建）"
    )
    q.add_argument("--code", required=True, help="流程编码，小写下划线（如 mr_routine、knee_mr）")
    q.add_argument("--name", required=True, help="流程名称（建议「部位 模态 扫描」，如 膝关节 MR 扫描）")
    q.add_argument(
        "--modality",
        nargs="+",
        required=True,
        choices=DATASET_MODALITIES,
        help="适用模态（可多个）",
    )
    q.add_argument("--description", default="")
    _add_runtime(q)
    q.set_defaults(fn=cmd_protocol_create)

    p = sub.add_parser("dataset", help="数据集接入：建库/导入/就绪")
    dataset_sub = p.add_subparsers(dest="dataset_cmd", required=True)

    q = dataset_sub.add_parser("create", help="新建数据集（status=draft）")
    q.add_argument("--code", required=True, help="数据集编码（唯一）")
    q.add_argument("--name", required=True, help="数据集名称")
    q.add_argument("--modality", required=True, choices=DATASET_MODALITIES, help="模态")
    q.add_argument(
        "--protocol", required=True, help="扫描流程 id 或 code（protocols 查看，优先预置标准流程）"
    )
    q.add_argument("--description", default="")
    _add_runtime(q)
    q.set_defaults(fn=cmd_dataset_create)

    q = dataset_sub.add_parser("show", help="查看数据集与最近导入任务")
    q.add_argument("--dataset", required=True, help="数据集 id 或 code")
    _add_runtime(q)
    q.set_defaults(fn=cmd_dataset_show)

    q = dataset_sub.add_parser("import", help="导入病例（上传 ZIP 或服务端路径）")
    q.add_argument("--dataset", required=True, help="数据集 id 或 code")
    group = q.add_mutually_exclusive_group(required=True)
    group.add_argument("--upload", help="本地数据源（目录/文件自动打 ZIP 上传）")
    group.add_argument(
        "--server-path",
        help="Studio 容器内可见路径（部署约定：宿主机 <实例>/imports/ → 容器 /imports，只读）",
    )
    q.add_argument(
        "--async", dest="async_job", action="store_true",
        help="--server-path 走后台导入任务（大体量数据推荐；令牌并发 ≤3）",
    )
    q.add_argument("--wait", action="store_true", help="后台任务创建后轮询等待完成")
    q.add_argument("--no-recursive", action="store_true", help="不递归子目录")
    q.add_argument(
        "--confirm-backup", action="store_true",
        help="确认已自行备份原始数据（导入为复制，源数据不动；后台任务强制要求）",
    )
    q.add_argument("--zip-dir", default="", help="临时 ZIP 存放目录（默认系统临时目录）")
    q.add_argument(
        "--max-upload-mb", type=int, default=DEFAULT_MAX_UPLOAD_MB,
        help=f"上传体积上限提示（默认 {DEFAULT_MAX_UPLOAD_MB}，与服务端默认一致）",
    )
    q.add_argument("--timeout", type=float, default=1800.0, help="上传/等待超时秒数")
    _add_runtime(q)
    q.set_defaults(fn=cmd_dataset_import)

    q = dataset_sub.add_parser("jobs", help="查看导入任务（可等待完成）")
    q.add_argument("--dataset", required=True, help="数据集 id 或 code")
    q.add_argument("--job", help="只看/等待指定任务")
    q.add_argument("--wait", action="store_true", help="轮询等待直到 succeeded/failed")
    q.add_argument("--timeout", type=float, default=3600.0, help="等待超时秒数")
    _add_runtime(q)
    q.set_defaults(fn=cmd_dataset_jobs)

    q = dataset_sub.add_parser("ready", help="标记数据集就绪（需已有病例）")
    q.add_argument("--dataset", required=True, help="数据集 id 或 code")
    _add_runtime(q)
    q.set_defaults(fn=cmd_dataset_ready)

    args = ap.parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except StudioError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
