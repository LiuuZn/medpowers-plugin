#!/usr/bin/env python3
"""notify.py — 训练通知组件：训练开始/中断/结束等时点经 webhook 推送提醒（至少支持飞书）。

配置：任务仓库 <工作区>/runtime/notify.json（由 set/disable 写入）。文件不存在或
enabled=false 时通知组件不生效——send 打印跳过原因并 exit 0，不报错、不重试。

子命令：
    set --machine-name <本机名称> --url <webhook地址> [--type feishu|generic] [--secret <飞书加签密钥>]
                                写入配置并启用（重配即覆盖）
    disable [--reason <说明>]   停用（保留配置，enabled=false）
    status                      查看配置（URL 与密钥脱敏）
    send --event <start|finish|interrupt|cancel|test|自定义文本>
         [--gpus 0,1] [--experiment EXP-001] [--note <说明>] [--style card|text] [--dry-run]
                                推送一条通知（未配置则跳过）
    test [--note <说明>]        发一条测试消息，确认 webhook 可用

通知是尽力而为的旁路操作：失败重试 1 次后放弃（send exit 1），绝不阻断训练与实验
记录动作；调用方不得用 && 把 send 与训练启动/finish 串联。

函数级 API（任务仓库脚本内 import 调用，绝不抛异常）：
    sys.path.insert(0, "<mtskills>/skills/notify")
    import notify
    notify.send_notification("interrupt", gpus=[0], experiment="EXP-002", note="CUDA OOM")

runtime 目录解析与 init-env 相同：环境变量 MTSKILLS_RUNTIME（直接指向 runtime 目录）
> 任务仓库工作区（cwd 向上最近含 .git 的祖先）下的 runtime/。仅用标准库。
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import platform
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

NOTIFY_FILE = "notify.json"
DEFAULT_TYPE = "feishu"
TIMEOUT = 10   # 单次请求超时（秒）：通知不该拖住训练流程
ATTEMPTS = 2   # 尽力而为：失败重试 1 次后放弃

# 事件 → (标题, 飞书卡片模板色)；未登记的事件按自定义信息处理（蓝色）
EVENTS = {
    "start": ("🚀 训练开始", "green"),
    "finish": ("✅ 训练结束", "green"),
    "interrupt": ("⚠️ 训练中断", "red"),
    "cancel": ("🚫 实验取消", "grey"),
    "test": ("🔔 通知测试", "blue"),
}


class NotifyError(Exception):
    """用户可见错误：拒绝执行、非零退出。"""


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def workspace_root() -> Path:
    """任务仓库工作区根：从 cwd 逐级向上（含 cwd）找最近含 .git 的祖先
    （兼容 submodule/worktree）；找不到则用 cwd 本身。"""
    cwd = Path.cwd()
    for d in (cwd, *cwd.parents):
        if (d / ".git").is_dir() or (d / ".git").is_file():
            return d
    return cwd


def resolve_runtime(override: str | None = None) -> Path:
    """runtime 状态目录：--runtime 显式指定 > 环境变量 MTSKILLS_RUNTIME > <工作区根>/runtime。"""
    env = override or os.environ.get("MTSKILLS_RUNTIME", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return workspace_root() / "runtime"


def config_path(runtime: Path) -> Path:
    return runtime / NOTIFY_FILE


def load_config(runtime: Path) -> dict | None:
    """读配置；文件不存在返回 None（= 未配置，组件不生效）。"""
    p = config_path(runtime)
    if not p.exists():
        return None
    try:
        cfg = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        raise NotifyError(f"{p} 无法解析：{e}（用 set 重写，或删除后重新配置）")
    if not isinstance(cfg, dict):
        raise NotifyError(f"{p} 不是 JSON 对象")
    return cfg


def save_config(runtime: Path, cfg: dict) -> Path:
    """原子写 notify.json（先 .tmp 再 replace，防止别的进程读到半个文件）。"""
    runtime.mkdir(parents=True, exist_ok=True)
    p = config_path(runtime)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, p)
    return p


# ---------- 飞书 ----------

def feishu_sign(secret: str, timestamp: str) -> str:
    """飞书自定义机器人加签：HMAC-SHA256，key = "timestamp\\nsecret"，空消息体，base64。"""
    string_to_sign = f"{timestamp}\n{secret}"
    digest = hmac.new(string_to_sign.encode("utf-8"), digestmod=hashlib.sha256).digest()
    return base64.b64encode(digest).decode("utf-8")


def feishu_payload(cfg: dict, title: str, color: str, plain: str, md: str,
                   style: str, info: dict) -> dict:
    # info 与 generic_payload 签名对齐（SENDERS 注册表统一按 7 参调用），飞书用不到
    if style == "text":
        payload: dict = {"msg_type": "text", "content": {"text": f"{title}\n{plain}"}}
    else:
        payload = {
            "msg_type": "interactive",
            "card": {
                "header": {"template": color, "title": {"tag": "plain_text", "content": title}},
                "elements": [
                    {"tag": "div", "text": {"tag": "lark_md", "content": md}},
                    {"tag": "hr"},
                    {"tag": "note", "elements": [{"tag": "plain_text",
                                                  "content": f"MTSkills 训练通知 · {now_iso()}"}]},
                ],
            },
        }
    secret = (cfg.get("webhook") or {}).get("secret") or ""
    if secret:  # 加签校验：timestamp + sign 加在 payload 顶层
        ts = str(int(time.time()))
        payload["timestamp"] = ts
        payload["sign"] = feishu_sign(secret, ts)
    return payload


def generic_payload(cfg: dict, title: str, color: str, plain: str, md: str,
                    style: str, info: dict) -> dict:
    """通用 webhook：POST 一个 JSON 对象（字段扁平，便于任何接收端解析）。"""
    out = {"title": title, "message": f"{title}\n{plain}"}
    out.update({k: v for k, v in info.items() if v})
    return out


SENDERS = {"feishu": feishu_payload, "generic": generic_payload}


# ---------- 通知内容 ----------

def gpu_names(indices: list[int]) -> dict[int, str]:
    """best-effort 查 GPU 名称（index → name）；nvidia-smi 不可用/超时返回空表，不影响发送。"""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if r.returncode != 0:
        return {}
    out: dict[int, str] = {}
    for line in r.stdout.strip().splitlines():
        v = [x.strip() for x in line.split(",")]
        if len(v) >= 2:
            try:
                out[int(v[0])] = v[1]
            except ValueError:
                continue
    return out


def parse_gpus(s: str | None) -> list[int]:
    if not s:
        return []
    ids = []
    for tok in s.replace("，", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            ids.append(int(tok))
        except ValueError:
            raise NotifyError(f"GPU 编号「{tok}」不是整数（--gpus 用逗号分隔，如 0,1）")
    return ids


def render_lines(machine: str, gpu_txt: str, experiment: str, note: str, bold: bool) -> str:
    """通知正文：机器名称 + GPU 编号 + 实验编号 + 操作说明 + 时间。"""
    def fmt(k: str, v: str) -> str:
        return f"**{k}：** {v}" if bold else f"{k}：{v}"
    pairs = [("机器", machine)]
    if gpu_txt:
        pairs.append(("GPU", gpu_txt))
    if experiment:
        pairs.append(("实验", experiment))
    if note:
        pairs.append(("说明", note))
    pairs.append(("时间", now_iso()))
    return "\n".join(fmt(k, v) for k, v in pairs)


def deliver(url: str, payload: dict, *, strict_code: bool) -> dict:
    """POST JSON 并按 webhook 语义判成败；共 ATTEMPTS 次，返回 {"ok": bool, ...}。

    strict_code=True（飞书）：HTTP 2xx 且响应 JSON 的 code/StatusCode==0 才算成功；
    strict_code=False（generic）：HTTP 2xx 即成功（不约束响应体）。"""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    last = ""
    for _ in range(ATTEMPTS):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                body = resp.read().decode("utf-8", "replace")
            if not strict_code:
                return {"ok": True, "response": body[:200]}
            try:
                rj = json.loads(body) if body else {}
            except json.JSONDecodeError:
                rj = {}
            code = rj.get("code", rj.get("StatusCode"))
            if code == 0:
                return {"ok": True, "response": rj.get("msg", "success")}
            last = f"webhook 返回 code={code}：{rj.get('msg') or body[:200]}"
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}"
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            last = f"{type(e).__name__}: {e}"
    return {"ok": False, "error": last}


def send_notification(event: str, *, gpus=None, experiment: str = "", note: str = "",
                      style: str = "card", runtime: str | None = None,
                      dry_run: bool = False) -> dict:
    """函数级 API：推送一条通知。绝不抛异常、绝不重试到阻塞——任何失败折叠为
    {"sent": False, "skipped"/"error": ...}，调用方无须 try/except 即可安全嵌入
    训练开始/中断/结束的处理路径。"""
    try:
        ids = [int(g) for g in (gpus or [])]
        rt = resolve_runtime(runtime)
        cfg = load_config(rt)
        if cfg is None:
            return {"sent": False, "skipped": f"通知未配置（{config_path(rt)} 不存在）"}
        if not cfg.get("enabled"):
            reason = cfg.get("disabled_reason") or ""
            return {"sent": False,
                    "skipped": "通知未启用（enabled=false）" + (f"：{reason}" if reason else "")}
        hook = cfg.get("webhook") or {}
        wtype = hook.get("type", DEFAULT_TYPE)
        url = (hook.get("url") or "").strip()
        if wtype not in SENDERS:
            return {"sent": False, "skipped": f"未知 webhook 类型「{wtype}」（重跑 set 配置）"}
        if not url:
            return {"sent": False, "skipped": "webhook 地址为空（重跑 set 配置）"}
        machine = cfg.get("machine_name") or platform.node()
        names = gpu_names(ids) if ids else {}
        gpu_txt = "、".join(f"{i}（{names[i]}）" if i in names else str(i) for i in ids) if ids else ""
        title, color = EVENTS.get(event, (f"ℹ️ {event}", "blue"))
        title = f"{title}｜{machine}"
        plain = render_lines(machine, gpu_txt, experiment, note, bold=False)
        md = render_lines(machine, gpu_txt, experiment, note, bold=True)
        payload = SENDERS[wtype](cfg, title, color, plain, md, style,
                                 {"machine": machine, "event": event, "experiment": experiment,
                                  "gpus": ids, "note": note, "time": now_iso()})
        if dry_run:
            return {"sent": False, "skipped": "dry-run（未实际发送）",
                    "url": url, "payload": payload}
        result = deliver(url, payload, strict_code=(wtype == "feishu"))
        return {"sent": result["ok"], **({"response": result["response"]} if result["ok"]
                                         else {"error": result["error"]})}
    except NotifyError as e:
        return {"sent": False, "skipped": str(e)}
    except Exception as e:  # 通知绝不向上抛：训练路径里的通知调用必须无副作用
        return {"sent": False, "error": f"{type(e).__name__}: {e}"}


# ---------- CLI ----------

def mask_url(u: str) -> str:
    """脱敏 webhook 地址：保留 scheme+host，路径只留前 6 字符。"""
    if not u:
        return ""
    m = re.match(r"^(https?://[^/]+/)(.*)$", u)
    if not m:
        return u[:12] + "…"
    tail = m.group(2)
    return m.group(1) + (tail[:6] + "…" if len(tail) > 6 else tail)


def cmd_set(args) -> int:
    machine = args.machine_name.strip()
    url = args.url.strip()
    if not machine:
        raise NotifyError("--machine-name 不能为空（通知中标识本机的名称）")
    if not re.match(r"^https?://", url):
        raise NotifyError("--url 必须是 http(s) 地址")
    if args.type == "feishu" and not re.search(r"feishu|larksuite|lark", url, re.I):
        print("警告：type=feishu 但 URL 不像飞书 webhook（open.feishu.cn / open.larksuite.com）；仍按提供地址保存",
              file=sys.stderr)
    if args.secret and args.type != "feishu":
        print("警告：--secret 仅对 type=feishu（加签校验）生效，generic 将忽略", file=sys.stderr)
    cfg = {
        "enabled": True,
        "machine_name": machine,
        "webhook": {"type": args.type, "url": url, "secret": args.secret or ""},
        "configured_at": now_iso(),
    }
    p = save_config(resolve_runtime(args.runtime), cfg)
    print(f"[notify] 已写入 {p}（enabled=true，type={args.type}，url={mask_url(url)}）")
    print("[notify] 建议运行 test 发送一条测试消息，确认 webhook 可用")
    return 0


def cmd_disable(args) -> int:
    rt = resolve_runtime(args.runtime)
    cfg = load_config(rt) or {"machine_name": "", "configured_at": None,
                              "webhook": {"type": DEFAULT_TYPE, "url": "", "secret": ""}}
    cfg["enabled"] = False
    cfg["disabled_reason"] = args.reason or "用户选择不启用"
    p = save_config(rt, cfg)
    print(f"[notify] 已停用（保留配置）：{p}——send 将自动跳过")
    return 0


def cmd_status(args) -> int:
    rt = resolve_runtime(args.runtime)
    cfg = load_config(rt)
    if cfg is None:
        print(f"通知未配置（{config_path(rt)} 不存在）——send 自动跳过；init-env 问卷或 set 可配置")
        return 0
    hook = cfg.get("webhook") or {}
    view = {
        "enabled": bool(cfg.get("enabled")),
        "machine_name": cfg.get("machine_name") or "",
        "webhook": {"type": hook.get("type", DEFAULT_TYPE),
                    "url": mask_url(hook.get("url", "")),
                    "secret": "已配置，不显示" if hook.get("secret") else ""},
        "configured_at": cfg.get("configured_at"),
    }
    if not cfg.get("enabled"):
        view["disabled_reason"] = cfg.get("disabled_reason") or ""
    print(json.dumps(view, ensure_ascii=False, indent=2))
    return 0


def cmd_send(args) -> int:
    res = send_notification(args.event, gpus=parse_gpus(args.gpus), experiment=args.experiment,
                            note=args.note or "", style=args.style, runtime=args.runtime,
                            dry_run=args.dry_run)
    if args.dry_run and "payload" in res:
        print(json.dumps({"url": mask_url(res["url"]), "payload": res["payload"]},
                         ensure_ascii=False, indent=2))
        return 0
    if res.get("sent"):
        print(f"[notify] 已发送：{args.event}（webhook 响应：{res.get('response', '')}）")
        return 0
    if "skipped" in res:
        print(f"[notify] 跳过：{res['skipped']}")
        return 0
    print(f"[notify] 发送失败（已重试 {ATTEMPTS - 1} 次）：{res.get('error')}", file=sys.stderr)
    print("[notify] 通知是尽力而为：不阻断训练/记录动作，可稍后重试或忽略", file=sys.stderr)
    return 1


def cmd_test(args) -> int:
    res = send_notification("test", note=args.note or "通知组件配置成功", runtime=args.runtime)
    if res.get("sent"):
        print(f"[notify] 测试消息已发送（webhook 响应：{res.get('response', '')}）——请到 IM 里确认收到")
        return 0
    if "skipped" in res:
        print(f"[notify] 跳过：{res['skipped']}")
        return 0
    print(f"[notify] 测试发送失败（已重试 {ATTEMPTS - 1} 次）：{res.get('error')}", file=sys.stderr)
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="训练通知组件（详见 skills/notify/SKILL.md）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_runtime(p):
        p.add_argument("--runtime", help="runtime 目录（默认 $MTSKILLS_RUNTIME 或 <工作区根>/runtime）")

    p = sub.add_parser("set", help="写入 webhook 配置并启用")
    p.add_argument("--machine-name", required=True, help="本机名称（通知中标识这台机器）")
    p.add_argument("--url", required=True, help="webhook 地址（http/https）")
    p.add_argument("--type", default=DEFAULT_TYPE, choices=sorted(SENDERS),
                   help=f"webhook 类型（默认 {DEFAULT_TYPE}）")
    p.add_argument("--secret", default="", help="飞书加签密钥（机器人开启加签校验时必填）")
    add_runtime(p)
    p.set_defaults(fn=cmd_set)

    p = sub.add_parser("disable", help="停用通知（保留配置）")
    p.add_argument("--reason", default="", help="停用原因（记录在配置里）")
    add_runtime(p)
    p.set_defaults(fn=cmd_disable)

    p = sub.add_parser("status", help="查看当前配置（脱敏）")
    add_runtime(p)
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("send", help="推送一条通知")
    p.add_argument("--event", required=True,
                   help="start|finish|interrupt|cancel|test 或自定义文本（按信息卡发送）")
    p.add_argument("--gpus", default="", help="GPU 编号，逗号分隔（如 0,1）")
    p.add_argument("--experiment", default="", help="实验编号（如 EXP-001）")
    p.add_argument("--note", default="", help="操作说明（中断原因等）")
    p.add_argument("--style", default="card", choices=["card", "text"],
                   help="飞书消息形态（默认卡片）")
    p.add_argument("--dry-run", action="store_true", help="只打印 payload 不发送")
    add_runtime(p)
    p.set_defaults(fn=cmd_send)

    p = sub.add_parser("test", help="发送测试消息")
    p.add_argument("--note", default="", help="测试消息附言")
    add_runtime(p)
    p.set_defaults(fn=cmd_test)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except NotifyError as e:
        print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)
