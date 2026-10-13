#!/usr/bin/env python3
"""exp_mgmt.py — 实验记录管理：编号分配、三时点状态机、git 自动提交、INDEX 生成、校验、旧格式迁移。

规范正本：docs/superpowers/specs/2026-10-13-exp-mgmt-design.md；
上下文：AGENTS.md「实验记录规范」、templates/runtime/README.md lease 协议。
零第三方依赖。"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

STATUSES = ("planned", "running", "done", "interrupted", "cancelled")
END_STATUSES = ("done", "interrupted", "cancelled")

# 新格式 EXP-<num>-<slug>.md（num 三位起）；旧格式 EXP-<8位日期>-<seq>-<slug>.md
NEW_RE = re.compile(r"^EXP-(\d{3,})-([a-z0-9][a-z0-9-]*)\.md$")
OLD_RE = re.compile(r"^EXP-(\d{8})-(\d+)-([a-z0-9][a-z0-9-]*)\.md$")

FIELD_RES = {
    "status": re.compile(r"^-[ \t]+\*\*status\*\*:[ \t]*(\S+)[ \t]*$", re.M),
    "planned_at": re.compile(r"^-[ \t]+\*\*planned_at\*\*:[ \t]*(.*)$", re.M),
    "started_at": re.compile(r"^-[ \t]+\*\*started_at\*\*:[ \t]*(.*)$", re.M),
    "ended_at": re.compile(r"^-[ \t]+\*\*ended_at\*\*:[ \t]*(.*)$", re.M),
}


class ExpError(Exception):
    """用户可见错误：拒绝执行、非零退出。"""


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def normalize_id(s: str) -> str:
    m = re.fullmatch(r"(?i)exp[-_]?(\d+)", s.strip())
    if not m:
        raise ExpError(f"实验编号「{s}」格式应为 EXP-<NNN>")
    return f"EXP-{int(m.group(1)):03d}"


def parse_record(path: Path) -> dict:
    m = NEW_RE.match(path.name)
    if not m:
        raise ExpError(f"{path.name}: 文件名不符合 EXP-<NNN>-<slug>.md 格式")
    text = path.read_text(encoding="utf-8")
    rec = {"path": path, "num": int(m.group(1)), "id": f"EXP-{m.group(1)}", "slug": m.group(2)}
    for field, rx in FIELD_RES.items():
        fm = rx.search(text)
        rec[field] = fm.group(1).strip() if fm else ""
    if rec["status"] not in STATUSES:
        raise ExpError(f"{path.name}: 非法 status「{rec['status']}」（合法：{'/'.join(STATUSES)}）")
    return rec


def scan_records(exp_dir: Path) -> tuple[list[dict], list[Path]]:
    records, old_format = [], []
    for p in sorted(exp_dir.glob("EXP-*.md")):
        if OLD_RE.match(p.name):
            old_format.append(p)
        elif NEW_RE.match(p.name):
            records.append(parse_record(p))
        else:
            raise ExpError(f"{p.name}: 无法识别的实验记录文件名")
    return records, old_format


def check_duplicates(records: list[dict]) -> None:
    seen: dict[int, dict] = {}
    for r in records:
        if r["num"] in seen:
            raise ExpError(f"编号冲突：{seen[r['num']]['path'].name} 与 {r['path'].name} 同为 EXP-{r['num']:03d}")
        seen[r["num"]] = r


def check_time_consistency(r: dict) -> list[str]:
    errs = []
    st, sa, ea = r["status"], r["started_at"], r["ended_at"]
    if st == "planned" and (sa or ea):
        errs.append(f"{r['path'].name}: planned 不应已有 started_at/ended_at")
    if st == "running":
        if not sa:
            errs.append(f"{r['path'].name}: running 缺 started_at")
        if ea:
            errs.append(f"{r['path'].name}: running 不应已有 ended_at")
    if st in END_STATUSES and not (sa and ea):
        errs.append(f"{r['path'].name}: {st} 缺 started_at 或 ended_at")
    return errs


def next_num(records: list[dict]) -> int:
    return max((r["num"] for r in records), default=0) + 1


INDEX_HEADER = (
    "# 实验记录索引\n\n"
    "> 本文件由 skills/exp-mgmt 的 list 动作自动生成，请勿手工编辑；记录文件是唯一事实源。\n\n"
    "| 编号 | 状态 | 标题 | planned_at | started_at | ended_at |\n"
    "|---|---|---|---|---|---|\n"
)


def _day(iso: str) -> str:
    return iso[:10] if iso else "—"


def render_index(records: list[dict]) -> str:
    lines = [
        f"| {r['id']} | {r['status']} | {r['slug']} | {_day(r['planned_at'])} "
        f"| {_day(r['started_at'])} | {_day(r['ended_at'])} |"
        for r in sorted(records, key=lambda r: r["num"])
    ]
    return INDEX_HEADER + "\n".join(lines) + "\n"


def git(repo: Path, *args: str) -> str:
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if p.returncode != 0:
        raise ExpError(f"git {' '.join(args)} 失败：{p.stderr.strip()}")
    return p.stdout


def require_repo(exp_dir: Path) -> Path:
    p = subprocess.run(["git", "-C", str(exp_dir), "rev-parse", "--show-toplevel"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise ExpError(f"{exp_dir} 不在 git 仓库内：实验记录必须由任务仓库 git 管理")
    return Path(p.stdout.strip())


def commit_paths(repo: Path, message: str, *paths: Path) -> bool:
    """路径级 add + commit；目标路径无暂存变更时不产生空提交。绝不使用 -A。"""
    git(repo, "add", "--", *[str(p) for p in paths])
    staged = subprocess.run(["git", "-C", str(repo), "diff", "--cached", "--quiet", "--",
                             *[str(p) for p in paths]],
                            capture_output=True, text=True)
    if staged.returncode == 0:
        return False
    # pathspec 提交：只提交目标路径，无关预暂存内容不被卷入（已 add 的新文件仍按工作树内容入提交）
    git(repo, "commit", "-m", message, "--", *[str(p) for p in paths])
    return True


def set_field(path: Path, field: str, value: str) -> None:
    text = path.read_text(encoding="utf-8")
    rx = FIELD_RES[field]
    if not rx.search(text):
        raise ExpError(f"{path.name}: 缺少 - **{field}** 字段行，模板损坏？")
    path.write_text(rx.sub(f"- **{field}**: {value}", text, count=1), encoding="utf-8")


def default_template_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "experiments" / "TEMPLATE.md"


def resolve_dir(args) -> Path:
    if getattr(args, "dir", None):
        return Path(args.dir).resolve()
    env = os.environ.get("MTSKILLS_EXPERIMENTS")
    return Path(env).resolve() if env else (Path.cwd() / "experiments").resolve()


def render_new_record(template_text: str, exp_id: str, slug: str) -> str:
    text = re.sub(r"^# EXP-.*$", f"# {exp_id}-{slug}", template_text, count=1, flags=re.M)
    if not text.startswith(f"# {exp_id}-{slug}"):
        raise ExpError("模板缺少一级标题（# EXP-…），无法生成记录")
    text = FIELD_RES["status"].sub("- **status**: planned", text, count=1)
    text = FIELD_RES["planned_at"].sub(f"- **planned_at**: {now_iso()}", text, count=1)
    return text


def refresh_index(exp_dir: Path, records: list[dict], *updated: dict) -> None:
    by_path = {r["path"]: r for r in records}
    for u in updated:
        by_path[u["path"]] = parse_record(u["path"])
    (exp_dir / "INDEX.md").write_text(render_index(by_path.values()), encoding="utf-8")


def find_record(records: list[dict], exp_id: str) -> dict:
    hits = [r for r in records if r["id"] == exp_id]
    if not hits:
        known = "、".join(sorted(r["id"] for r in records)) or "（无）"
        raise ExpError(f"找不到 {exp_id}（现有：{known}）")
    if len(hits) > 1:
        raise ExpError(f"{exp_id} 命中多条记录，先运行 check 排查编号冲突")
    return hits[0]


def append_cancel_reason(path: Path, reason: str) -> None:
    text = path.read_text(encoding="utf-8")
    heading = "## 取消理由【仅 cancelled 填】"
    if heading in text:
        text = text.replace(heading, f"{heading}\n\n- {reason}", 1)
    else:
        text = text.rstrip("\n") + f"\n\n## 取消理由\n\n- {reason}\n"
    path.write_text(text, encoding="utf-8")


def cmd_new(args) -> int:
    exp_dir = resolve_dir(args)
    repo = require_repo(exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    records, old_format = scan_records(exp_dir)
    if old_format:
        names = "、".join(p.name for p in old_format)
        raise ExpError(f"存在 {len(old_format)} 个旧格式记录（{names}），先运行 migrate 再新建")
    check_duplicates(records)
    slug = args.slug
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", slug):
        raise ExpError(f"slug「{slug}」只允许小写字母/数字/短横线")
    num = next_num(records)
    exp_id = f"EXP-{num:03d}"
    path = exp_dir / f"{exp_id}-{slug}.md"
    if path.exists():
        raise ExpError(f"{path.name} 已存在")
    template = Path(args.template).read_text(encoding="utf-8") if args.template \
        else default_template_path().read_text(encoding="utf-8")
    path.write_text(render_new_record(template, exp_id, slug), encoding="utf-8")
    refresh_index(exp_dir, records, parse_record(path))
    commit_paths(repo, f"exp({exp_id}): plan", exp_dir)
    print(f"{path.name} 已创建（planned）并提交。规划内容（任务与数据/配置）请尽快填入记录。")
    return 0


def cmd_start(args) -> int:
    exp_dir = resolve_dir(args)
    repo = require_repo(exp_dir)
    exp_id = normalize_id(args.exp)
    records, old_format = scan_records(exp_dir)
    if old_format:
        names = "、".join(p.name for p in old_format)
        raise ExpError(f"存在旧格式记录（{names}），先运行 migrate 再 start")
    check_duplicates(records)
    rec = find_record(records, exp_id)
    if rec["status"] != "planned":
        raise ExpError(f"{rec['path'].name}: 状态为 {rec['status']}，只有 planned 可 start")
    set_field(rec["path"], "status", "running")
    set_field(rec["path"], "started_at", now_iso())
    refresh_index(exp_dir, records, rec)
    commit_paths(repo, f"exp({exp_id}): start", exp_dir)
    print(f"{exp_id} → running。接下来按 lease 协议写 lease（experiment 字段引用 {exp_id}），再后台分离启动训练。")
    return 0


def cmd_finish(args) -> int:
    exp_dir = resolve_dir(args)
    repo = require_repo(exp_dir)
    exp_id = normalize_id(args.exp)
    if args.status not in ("done", "interrupted"):
        raise ExpError("--status 只接受 done 或 interrupted")
    records, _ = scan_records(exp_dir)
    check_duplicates(records)
    rec = find_record(records, exp_id)
    if rec["status"] != "running":
        raise ExpError(f"{rec['path'].name}: 状态为 {rec['status']}，只有 running 可 finish")
    set_field(rec["path"], "status", args.status)
    set_field(rec["path"], "ended_at", now_iso())
    refresh_index(exp_dir, records, rec)
    commit_paths(repo, f"exp({exp_id}): {args.status}", exp_dir)
    print(f"{exp_id} → {args.status}（ended_at 已填）。确认结果/诊断/下一轮决策/产物已填入记录。")
    return 0


def cmd_cancel(args) -> int:
    exp_dir = resolve_dir(args)
    repo = require_repo(exp_dir)
    exp_id = normalize_id(args.exp)
    records, _ = scan_records(exp_dir)
    check_duplicates(records)
    rec = find_record(records, exp_id)
    if rec["status"] not in ("planned", "running"):
        raise ExpError(f"{rec['path'].name}: 状态为 {rec['status']}，只有 planned/running 可 cancel")
    set_field(rec["path"], "status", "cancelled")
    if not rec["started_at"]:
        # planned 直接取消：没有真实开始时间，以取消时刻补齐——check 要求 cancelled 必有 started_at 与 ended_at
        set_field(rec["path"], "started_at", now_iso())
    set_field(rec["path"], "ended_at", now_iso())
    append_cancel_reason(rec["path"], args.reason)
    refresh_index(exp_dir, records, rec)
    commit_paths(repo, f"exp({exp_id}): cancel", exp_dir)
    print(f"{exp_id} → cancelled。编号永久占用，不再复用。")
    return 0


def check_leases(runtime_dir: Path, records: list[dict]) -> list[str]:
    errs = []
    lease_dir = runtime_dir / "training"
    if not lease_dir.is_dir():
        return errs
    known = {r["id"] for r in records}
    running = {r["id"] for r in records if r["status"] == "running"}
    for lease in sorted(lease_dir.glob("*.json")):
        try:
            payload = json.loads(lease.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            errs.append(f"lease {lease.name} 无法解析：{e}")
            continue
        if not isinstance(payload, dict):
            errs.append(f"lease {lease.name} 不是 JSON 对象")
            continue
        exp = payload.get("experiment", "")
        if not isinstance(exp, str):
            errs.append(f"lease {lease.name} 的 experiment 不是字符串")
            continue
        if not exp:
            continue
        try:
            norm = normalize_id(exp)
        except ExpError:
            errs.append(f"lease {lease.name} 的 experiment「{exp}」不是合法 EXP-<NNN> 编号")
            continue
        if norm not in known:
            errs.append(f"lease {lease.name} 引用的 {norm} 不存在（幽灵记录）")
        elif norm not in running:
            errs.append(f"lease {lease.name} 引用的 {norm} 状态非 running：lease 未删或记录未更新，按 stale lease 协议核对")
    return errs


def cmd_list(args) -> int:
    exp_dir = resolve_dir(args)
    repo = require_repo(exp_dir)
    records, old_format = scan_records(exp_dir)
    check_duplicates(records)
    index = exp_dir / "INDEX.md"
    new_text = render_index(records)
    if not index.exists() or index.read_text(encoding="utf-8") != new_text:
        index.write_text(new_text, encoding="utf-8")
        commit_paths(repo, "exp: index 同步", exp_dir)
    for r in sorted(records, key=lambda r: r["num"]):
        print(f"{r['id']}\t{r['status']}\t{r['slug']}\tplanned={_day(r['planned_at'])} "
              f"started={_day(r['started_at'])} ended={_day(r['ended_at'])}")
    if old_format:
        print(f"警告：{len(old_format)} 个旧格式记录待迁移（migrate）", file=sys.stderr)
    return 0


def cmd_check(args) -> int:
    exp_dir = resolve_dir(args)
    repo = require_repo(exp_dir)
    errors, warnings = [], []
    try:
        records, old_format = scan_records(exp_dir)
        check_duplicates(records)
    except ExpError as e:
        records, old_format = [], []
        errors.append(str(e))
    for r in records:
        errors.extend(check_time_consistency(r))
    index = exp_dir / "INDEX.md"
    if not index.exists() or index.read_text(encoding="utf-8") != render_index(records):
        errors.append("INDEX.md 与记录不一致：运行 list 重新生成")
    if old_format:
        names = "、".join(p.name for p in old_format)
        warnings.append(f"{len(old_format)} 个旧格式记录（{names}）：运行 migrate 迁移")
    if getattr(args, "runtime", None):
        errors.extend(check_leases(Path(args.runtime), records))
    status = subprocess.run(["git", "-C", str(exp_dir), "status", "--porcelain", "--", "."],
                            capture_output=True, text=True)
    if status.returncode == 0 and status.stdout.strip():
        warnings.append("记录目录存在未提交变更：git 管理要求不留存未提交记录")
    for w in warnings:
        print(f"警告：{w}", file=sys.stderr)
    if errors:
        for e in errors:
            print(f"错误：{e}", file=sys.stderr)
        return 1
    print(f"check 通过：{len(records)} 条记录" + (f"，{len(warnings)} 条警告" if warnings else ""))
    return 0


EXP_TOKEN_RE = re.compile(r"EXP-\d{8}-\d+-[a-z0-9-]+|EXP-\d+-[a-z0-9-]+|EXP-\d{1,4}(?![\d-])")


def rewrite_exp_fields(node, mapping: dict) -> bool:
    """递归改写 dict/list 中 key=='exp' 且值命中旧名的字段。返回是否发生改写。"""
    changed = False
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "exp" and isinstance(v, str) and v in mapping:
                node[k] = mapping[v][1]
                changed = True
            else:
                changed = rewrite_exp_fields(v, mapping) or changed
    elif isinstance(node, list):
        for v in node:
            changed = rewrite_exp_fields(v, mapping) or changed
    return changed


def find_dangling_refs(texts: list[tuple[str, str]], known_names: set[str],
                       migrated: set[str] = frozenset()) -> list[str]:
    """报告悬空引用：文中出现的 EXP 编号在记录目录中找不到对应文件。
    migrated 中的编号是本次迁移刚改名的，跳过不报。只报告，不虚构记录。"""
    out = set()
    for src, text in texts:
        for tok in EXP_TOKEN_RE.findall(text):
            if tok in migrated:
                continue
            if not any(n == tok + ".md" or n.startswith(tok + "-") for n in known_names):
                out.add(f"{src}: 引用 {tok} 但无对应记录文件")
    return sorted(out)


def _is_tracked(repo: Path, p: Path) -> bool:
    r = subprocess.run(["git", "-C", str(repo), "ls-files", "--error-unmatch", str(p)],
                       capture_output=True, text=True)
    return r.returncode == 0


def cmd_migrate(args) -> int:
    exp_dir = resolve_dir(args)
    repo = require_repo(exp_dir)
    records, old_format = scan_records(exp_dir)
    if not old_format:
        print("无旧格式记录，无需迁移")
        return 0
    check_duplicates(records)
    start = next_num(records)
    ordered = sorted(old_format, key=lambda p: (int(OLD_RE.match(p.name).group(1)),
                                                int(OLD_RE.match(p.name).group(2))))
    mapping = {}
    for i, p in enumerate(ordered):
        m = OLD_RE.match(p.name)
        new_id = f"EXP-{start + i:03d}"
        mapping[p.stem] = (f"{new_id}-{m.group(3)}.md", new_id)
    # 原子性（spec §7：先全部校验后写盘）第一步：改名前预扫描全部旧记录——提取 status 行、
    # 校验 status 值合法、算好注入内容；任一文件缺 status 行或 status 值非法都一次性列出
    # 全部问题文件并拒绝，磁盘不动。
    plans = []
    missing = []
    invalid = []
    tz = datetime.now().astimezone().strftime('%z')  # 本机时区，如 +0800 → 下面补冒号成 +08:00
    tz = f"{tz[:3]}:{tz[3:]}"
    for old_id, (new_name, _) in mapping.items():
        m = OLD_RE.match(old_id + ".md")
        text = (exp_dir / (old_id + ".md")).read_text(encoding="utf-8")
        status_m = FIELD_RES["status"].search(text)
        if not status_m:
            missing.append(old_id)
            continue
        status = status_m.group(1)
        if status not in STATUSES:
            invalid.append(f"{old_id}（{status}）")
            continue
        day = f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:8]}"
        stamp = f"{day}T00:00:00{tz}"
        # 补齐 v2 时间字段行：planned_at 取文件名日期；终态记录缺 started/ended 时同以文件名日期填充
        #（文件名日期是唯一可信来源）；running 状态的陈旧记录留空，让 check 报错提示人工处置。
        inject = ""
        if not FIELD_RES["planned_at"].search(text):
            inject += f"\n- **planned_at**: {stamp}"
        if not FIELD_RES["started_at"].search(text):
            inject += f"\n- **started_at**: {stamp if status in END_STATUSES else ''}"
        if not FIELD_RES["ended_at"].search(text):
            inject += f"\n- **ended_at**: {stamp if status in END_STATUSES else ''}"
        plans.append((old_id, new_name, text, status_m.group(0), inject))
    if missing or invalid:
        problems = []
        if missing:
            problems.append(f"{'、'.join(missing)}: 缺 - **status** 行")
        if invalid:
            problems.append(f"{'、'.join(invalid)}: status 值非法（合法：{'/'.join(STATUSES)}）")
        raise ExpError("；".join(problems) + "；无法迁移，请人工修正后重试")
    # 原子性第二步：versions.json 提前加载（同样在任何写盘之前），解析失败包成 ExpError。
    versions_path = None
    versions_data = None
    dangling_texts = []
    if getattr(args, "runtime", None):
        runtime_dir = Path(args.runtime)
        versions_path = runtime_dir / "versions.json"
        if versions_path.exists():
            raw = versions_path.read_text(encoding="utf-8")
            try:
                versions_data = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ExpError(f"versions.json 无法解析：{e}")
            dangling_texts.append(("versions.json", raw))
        lease_dir = runtime_dir / "training"
        if lease_dir.is_dir():
            for lease in sorted(lease_dir.glob("*.json")):
                dangling_texts.append((f"training/{lease.name}", lease.read_text(encoding="utf-8")))
    # 校验全部通过才开始写盘：改名 → 标题改写与字段注入（用预扫描结果）→ versions.json → INDEX → 提交。
    for old_id, new_name, text, status_line, inject in plans:
        old_path, new_path = exp_dir / (old_id + ".md"), exp_dir / new_name
        if _is_tracked(repo, old_path):
            git(repo, "mv", str(old_path), str(new_path))
        else:
            old_path.rename(new_path)  # 未跟踪的旧记录直接改名，随后由 commit_paths 入库
        text = re.sub(r"^# EXP-.*$", f"# {new_name[:-3]}", text, count=1, flags=re.M)
        if inject:
            text = text.replace(status_line, status_line + inject, 1)
        new_path.write_text(text, encoding="utf-8")
    if versions_data is not None:
        rewrite_exp_fields(versions_data, mapping)
        versions_path.write_text(json.dumps(versions_data, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
    records2, _ = scan_records(exp_dir)
    (exp_dir / "INDEX.md").write_text(render_index(records2), encoding="utf-8")
    commit_args = [repo, f"exp: migrate 旧格式记录至新编号（{len(mapping)} 条）", exp_dir]
    if versions_path is not None and versions_path.exists():
        commit_args.append(versions_path)
    commit_paths(*commit_args)
    for line in find_dangling_refs(dangling_texts, {p.name for p in exp_dir.glob("EXP-*.md")},
                                   migrated=set(mapping)):
        print(f"悬空引用：{line}", file=sys.stderr)
    print("迁移映射：")
    for old_id, (new_name, _) in mapping.items():
        print(f"  {old_id} -> {new_name}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="实验记录管理（编号/状态机/git 提交/INDEX/校验/迁移）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("new", help="规划实验：分配编号建 planned 记录")
    p.add_argument("--dir", help="记录目录（默认 $MTSKILLS_EXPERIMENTS 或 ./experiments）")
    p.add_argument("--slug", required=True, help="短横线小写短语，如 unet3d-baseline")
    p.add_argument("--template", help="模板路径（默认插件 experiments/TEMPLATE.md）")
    p.set_defaults(fn=cmd_new)

    p = sub.add_parser("start", help="安排训练门禁：planned → running")
    p.add_argument("--dir", help="记录目录")
    p.add_argument("exp", help="实验编号，如 EXP-001")
    p.set_defaults(fn=cmd_start)

    p = sub.add_parser("finish", help="训练结束：running → done|interrupted")
    p.add_argument("--dir", help="记录目录")
    p.add_argument("exp", help="实验编号")
    p.add_argument("--status", required=True, choices=["done", "interrupted"])
    p.set_defaults(fn=cmd_finish)

    p = sub.add_parser("cancel", help="取消实验：planned|running → cancelled（编号不复用）")
    p.add_argument("--dir", help="记录目录")
    p.add_argument("exp", help="实验编号")
    p.add_argument("--reason", required=True, help="取消理由（写入记录）")
    p.set_defaults(fn=cmd_cancel)

    p = sub.add_parser("list", help="查询实验状态并重生成 INDEX.md")
    p.add_argument("--dir", help="记录目录")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("check", help="只读校验：编号/状态/INDEX/lease 引用")
    p.add_argument("--dir", help="记录目录")
    p.add_argument("--runtime", help="任务仓库 runtime 目录（校验 training/ 下 lease 引用）")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("migrate", help="旧格式 EXP-YYYYMMDD-序号-slug 迁移为新编号")
    p.add_argument("--dir", help="记录目录")
    p.add_argument("--runtime", help="任务仓库 runtime 目录（改写 versions.json 引用、报告 lease 悬空引用）")
    p.set_defaults(fn=cmd_migrate)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ExpError as e:
        print(f"错误：{e}", file=sys.stderr)
        sys.exit(1)
