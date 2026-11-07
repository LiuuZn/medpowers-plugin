"""exp_cli.py — 实验记录管理的 agent 侧 CLI（原 skills/exp-mgmt/exp_mgmt.py 迁入）。

全接管模式（训练框架经 medpowers.training_session）下，start/finish 由训练进程自己做；
本 CLI 的 start/finish 是**遗留兜底**，仅用于训练代码未集成包的过渡期。new/cancel/
list/check/migrate 仍是智能体的常备动作。核心逻辑在同包 records.py。"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from . import records as R
from .records import (ExpError, append_cancel_reason, check_duplicates, commit_paths,
                      default_template_path, find_record, next_num, normalize_id,
                      parse_record, refresh_index, render_index, render_new_record,
                      require_repo, scan_records, sync_pull, sync_push, upstream_of,
                      ahead_behind, check_leases, rewrite_exp_fields, find_dangling_refs,
                      finish_experiment, start_experiment, _is_tracked, OLD_RE)


def _dir(args) -> Path:
    return R.resolve_dir(getattr(args, "dir", None))


def _studio_push(exp_id: str, model: str, status: str, *, machine: str = "",
                 gpus=None, note: str = "", runtime=None) -> None:
    """把实验状态镜像到 Studio（尽力而为；未配置或失败都不影响 git 台账）。"""
    if not model:
        return
    try:
        from .studio import StudioClient, load_config
        cfg = load_config(runtime)
        if not cfg:
            return
        payload = {
            "exp_id": exp_id,
            "task_code": model,
            "status": status,
            "machine": machine,
            "notes": note,
        }
        if gpus:
            payload["config"] = {"gpus": list(gpus)}
        StudioClient.from_config(cfg).upsert_experiment(payload)
    except Exception as e:  # noqa: BLE001  镜像失败绝不回滚实验记录
        print(f"[警告] Studio 镜像失败（不影响实验记录）：{e}", file=sys.stderr)


def cmd_new(args) -> int:
    exp_dir = _dir(args)
    exp_dir.mkdir(parents=True, exist_ok=True)
    repo = require_repo(exp_dir)
    print(f"[同步] {sync_pull(repo, args.allow_offline)}")
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
    if getattr(args, "task", ""):
        R.set_field(path, "model", args.task.strip())
    refresh_index(exp_dir, records, parse_record(path))
    commit_paths(repo, f"exp({exp_id}): plan", exp_dir)
    print(f"[同步] {sync_push(repo, args.allow_offline)}")
    _studio_push(exp_id, getattr(args, "task", ""), "planned",
                 runtime=exp_dir.parent / "runtime")
    print(f"{path.name} 已创建（planned）并提交。规划内容（任务与数据/配置）请尽快填入记录。")
    return 0


def cmd_start(args) -> int:
    gpus = [int(g) for g in (args.gpus.split(",") if args.gpus else [])] or None
    res = start_experiment(args.exp, gpus=gpus, machine=args.machine or "",
                           exp_dir=getattr(args, "dir", None),
                           allow_offline=args.allow_offline)
    rec = parse_record(path) if (path := Path(res["path"])).exists() else {}
    runtime = (Path(args.dir).parent / "runtime") if getattr(args, "dir", None) else None
    _studio_push(res["id"], rec.get("model", ""), "running",
                 machine=args.machine or "", gpus=gpus, runtime=runtime)
    print("[遗留兜底] 正路是训练框架经 medpowers.training_session 在训练进程内 start（门禁失败训练不开始）；"
          "本命令仅用于训练代码未集成包的过渡期。")
    print(f"接下来按 lease 协议写 lease（experiment 字段引用 {normalize_id(args.exp)}），再后台分离启动训练。")
    return 0


def cmd_finish(args) -> int:
    res = finish_experiment(args.exp, args.status, note=args.note or "",
                            exp_dir=getattr(args, "dir", None), allow_offline=args.allow_offline)
    if not res["pushed"] and not res.get("offline"):
        raise ExpError(res["push_note"])  # 硬推送失败 = 动作未完成；--allow-offline 豁免除外
    rec = parse_record(Path(res["path"])) if Path(res["path"]).exists() else {}
    runtime = (Path(args.dir).parent / "runtime") if getattr(args, "dir", None) else None
    _studio_push(res["id"], rec.get("model", ""), res["status"],
                 machine=rec.get("machine", ""), note=args.note or "", runtime=runtime)
    print("[遗留兜底] 正路是训练框架经 training_session 在训练进程内 finish（指标/产物/时长自动入账）；"
          "本命令仅用于训练代码未集成包的过渡期。")
    print(f"{res['id']} → {res['status']}（ended_at 已填）。确认结果/诊断/下一轮决策/产物已填入记录。")
    return 0


def cmd_cancel(args) -> int:
    exp_dir = _dir(args)
    repo = require_repo(exp_dir)
    print(f"[同步] {sync_pull(repo, args.allow_offline)}")
    exp_id = normalize_id(args.exp)
    records, _ = scan_records(exp_dir)
    check_duplicates(records)
    rec = find_record(records, exp_id)
    if rec["status"] not in ("planned", "running"):
        raise ExpError(f"{rec['path'].name}: 状态为 {rec['status']}，只有 planned/running 可 cancel")
    R.set_field(rec["path"], "status", "cancelled")
    if not rec["started_at"]:
        # planned 直接取消：没有真实开始时间，以取消时刻补齐——check 要求 cancelled 必有 started_at 与 ended_at
        R.set_field(rec["path"], "started_at", R.now_iso())
    R.set_field(rec["path"], "ended_at", R.now_iso())
    append_cancel_reason(rec["path"], args.reason)
    refresh_index(exp_dir, records, rec)
    commit_paths(repo, f"exp({exp_id}): cancel", exp_dir)
    print(f"[同步] {sync_push(repo, args.allow_offline)}")
    _studio_push(exp_id, rec.get("model", ""), "cancelled",
                 machine=rec.get("machine", ""), note=args.reason,
                 runtime=exp_dir.parent / "runtime")
    print(f"{exp_id} → cancelled。编号永久占用，不再复用。")
    return 0


def cmd_list(args) -> int:
    exp_dir = _dir(args)
    repo = require_repo(exp_dir)
    print(f"[同步] {sync_pull(repo, args.allow_offline)}", file=sys.stderr)
    records, old_format = scan_records(exp_dir)
    check_duplicates(records)
    index = exp_dir / "INDEX.md"
    new_text = render_index(records)
    if not index.exists() or index.read_text(encoding="utf-8") != new_text:
        index.write_text(new_text, encoding="utf-8")
        commit_paths(repo, "exp: index 同步", exp_dir)
    print(f"[同步] {sync_push(repo, args.allow_offline)}", file=sys.stderr)
    for r in sorted(records, key=lambda r: r["num"]):
        print(f"{r['id']}\t{r['status']}\t{r.get('machine') or '—'}\t{r['slug']}\t"
              f"planned={R._day(r['planned_at'])} started={R._day(r['started_at'])} "
              f"ended={R._day(r['ended_at'])}")
    if old_format:
        print(f"警告：{len(old_format)} 个旧格式记录待迁移（migrate）", file=sys.stderr)
    return 0


def cmd_check(args) -> int:
    exp_dir = _dir(args)
    repo = require_repo(exp_dir)
    errors, warnings = [], []
    try:
        records, old_format = scan_records(exp_dir)
        check_duplicates(records)
    except ExpError as e:
        records, old_format = [], []
        errors.append(str(e))
    for r in records:
        errors.extend(R.check_time_consistency(r))
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
    up = upstream_of(repo)
    if up:
        ref = f"{up[0]}/{up[1]}"
        probe = subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", "--quiet", ref],
                               capture_output=True, text=True)
        if probe.returncode == 0:
            ahead, behind = ahead_behind(repo, ref)
            if ahead:
                warnings.append(f"本地有 {ahead} 个提交未推送：其他机器看不到这些变更（推送成功才算动作完成）")
            if behind:
                warnings.append(f"按上次 fetch，本地落后 {ref} {behind} 个提交：先拉取同步再安排训练")
    for w in warnings:
        print(f"警告：{w}", file=sys.stderr)
    if errors:
        for e in errors:
            print(f"错误：{e}", file=sys.stderr)
        return 1
    print(f"check 通过：{len(records)} 条记录" + (f"，{len(warnings)} 条警告" if warnings else ""))
    return 0


def cmd_migrate(args) -> int:
    exp_dir = _dir(args)
    repo = require_repo(exp_dir)
    print(f"[同步] {sync_pull(repo, args.allow_offline)}")
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
        status_m = R.FIELD_RES["status"].search(text)
        if not status_m:
            missing.append(old_id)
            continue
        status = status_m.group(1)
        if status not in R.STATUSES:
            invalid.append(f"{old_id}（{status}）")
            continue
        day = f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:8]}"
        stamp = f"{day}T00:00:00{tz}"
        # 补齐 v2 时间字段行：planned_at 取文件名日期；终态记录缺 started/ended 时同以文件名日期填充
        #（文件名日期是唯一可信来源）；running 状态的陈旧记录留空，让 check 报错提示人工处置。
        inject = ""
        if not R.FIELD_RES["planned_at"].search(text):
            inject += f"\n- **planned_at**: {stamp}"
        if not R.FIELD_RES["started_at"].search(text):
            inject += f"\n- **started_at**: {stamp if status in R.END_STATUSES else ''}"
        if not R.FIELD_RES["ended_at"].search(text):
            inject += f"\n- **ended_at**: {stamp if status in R.END_STATUSES else ''}"
        plans.append((old_id, new_name, text, status_m.group(0), inject))
    if missing or invalid:
        problems = []
        if missing:
            problems.append(f"{'、'.join(missing)}: 缺 - **status** 行")
        if invalid:
            problems.append(f"{'、'.join(invalid)}: status 值非法（合法：{'/'.join(R.STATUSES)}）")
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
            R.git(repo, "mv", str(old_path), str(new_path))
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
    print(f"[同步] {sync_push(repo, args.allow_offline)}")
    for line in find_dangling_refs(dangling_texts, {p.name for p in exp_dir.glob("EXP-*.md")},
                                   migrated=set(mapping)):
        print(f"悬空引用：{line}", file=sys.stderr)
    print("迁移映射：")
    for old_id, (new_name, _) in mapping.items():
        print(f"  {old_id} -> {new_name}")
    return 0


def _add_sync_flag(p) -> None:
    p.add_argument("--allow-offline", action="store_true",
                   help="远端不可达时仍按本地状态继续（跳过拉取与推送，撞号风险自担；恢复网络后请 check 并补推送）")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="实验记录管理（编号/状态机/git 提交与多机推送同步/INDEX/校验/迁移）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("new", help="规划实验：分配编号建 planned 记录")
    p.add_argument("--dir", help="记录目录（默认 $MEDPOWERS_EXPERIMENTS 或 ./experiments）")
    p.add_argument("--slug", required=True, help="短横线小写短语，如 unet3d-baseline")
    p.add_argument("--task", default="", help="Studio 任务 code（= 模型）；写入 model 字段并镜像 planned")
    p.add_argument("--template", help="模板路径（默认插件 experiments/TEMPLATE.md）")
    _add_sync_flag(p)
    p.set_defaults(fn=cmd_new)

    p = sub.add_parser("start", help="[遗留兜底] 安排训练门禁：planned → running（正路：训练框架 training_session）")
    p.add_argument("--dir", help="记录目录")
    p.add_argument("exp", help="实验编号，如 EXP-001")
    p.add_argument("--gpus", default="", help="GPU 编号，逗号分隔（写入记录 gpus 字段）")
    p.add_argument("--machine", default="", help="机器名称（写入记录 machine 字段；默认不写）")
    _add_sync_flag(p)
    p.set_defaults(fn=cmd_start)

    p = sub.add_parser("finish", help="[遗留兜底] 训练结束：running → done|interrupted（正路：训练框架 training_session）")
    p.add_argument("--dir", help="记录目录")
    p.add_argument("exp", help="实验编号")
    p.add_argument("--status", required=True, choices=["done", "interrupted"])
    p.add_argument("--note", default="", help="说明（追加进记录 ## 结果 节）")
    _add_sync_flag(p)
    p.set_defaults(fn=cmd_finish)

    p = sub.add_parser("cancel", help="取消实验：planned|running → cancelled（编号不复用）")
    p.add_argument("--dir", help="记录目录")
    p.add_argument("exp", help="实验编号")
    p.add_argument("--reason", required=True, help="取消理由（写入记录）")
    _add_sync_flag(p)
    p.set_defaults(fn=cmd_cancel)

    p = sub.add_parser("list", help="查询实验状态并重生成 INDEX.md")
    p.add_argument("--dir", help="记录目录")
    _add_sync_flag(p)
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("check", help="只读校验：编号/状态/INDEX/lease 引用")
    p.add_argument("--dir", help="记录目录")
    p.add_argument("--runtime", help="任务仓库 runtime 目录（校验 training/ 下 lease 引用）")
    p.set_defaults(fn=cmd_check)

    p = sub.add_parser("migrate", help="旧格式 EXP-YYYYMMDD-序号-slug 迁移为新编号")
    p.add_argument("--dir", help="记录目录")
    p.add_argument("--runtime", help="任务仓库 runtime 目录（改写 versions.json 引用、报告 lease 悬空引用）")
    _add_sync_flag(p)
    p.set_defaults(fn=cmd_migrate)

    args = ap.parse_args(argv)
    return args.fn(args)
