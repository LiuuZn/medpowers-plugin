"""records.py — 实验记录核心：编号分配、三时点状态机、git 同步红线、INDEX、校验原语，
以及供训练进程内嵌调用的库 API（start_experiment / finish_experiment）。

规范正本：docs/superpowers/specs/2026-10-13-exp-mgmt-design.md（多机同步见 §13）；
上下文：AGENTS.md「实验记录规范」、templates/runtime/README.md lease 协议。
零第三方依赖（外部仅 git CLI）。

多机同步（拉取-确认-提交）：任务仓库远端是多台机器唯一的实验台账事实源。全部变更动作
强制「fetch+快进拉取 → 在最新记录上确认编号与状态 → 提交后立即推送」；推送被拒 = 动作
未完成（他机刚推送，可能撞号/撞状态），按报错恢复。远端不可达默认拒绝执行
（allow_offline=True 显式豁免）；分支无上游远端视为单机仓库，跳过同步。

两级纪律（与 notify 的"尽力而为"相对）：start_experiment 是**门禁语义**——失败抛
ExpError，调用方（训练进程）必须中止，训练不得开始；finish_experiment 在收尾路径上
拉取/校验失败抛 ExpError（由调用方决定降级处理），推送失败不抛、返回 pushed=False
（进程即将退出，check 兜底报告未推送提交）。"""
from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

STATUSES = ("planned", "running", "done", "interrupted", "cancelled")
END_STATUSES = ("done", "interrupted", "cancelled")

# 新格式 EXP-<num>-<slug>.md（num 三位起）；旧格式 EXP-<8位日期>-<seq>-<slug>.md
NEW_RE = re.compile(r"^EXP-(\d{3,})-([a-z0-9][a-z0-9-]*)\.md$")
OLD_RE = re.compile(r"^EXP-(\d{8})-(\d+)-([a-z0-9][a-z0-9-]*)\.md$")

FIELD_ORDER = ("status", "model", "planned_at", "started_at", "ended_at", "machine", "gpus")
FIELD_RES = {
    "status": re.compile(r"^-[ \t]+\*\*status\*\*:[ \t]*(\S+)[ \t]*$", re.M),
    # Studio 任务（= 模型）标识；可选，旧记录缺行属正常。
    "model": re.compile(r"^-[ \t]+\*\*model\*\*:[ \t]*(.*)$", re.M),
    "planned_at": re.compile(r"^-[ \t]+\*\*planned_at\*\*:[ \t]*(.*)$", re.M),
    "started_at": re.compile(r"^-[ \t]+\*\*started_at\*\*:[ \t]*(.*)$", re.M),
    "ended_at": re.compile(r"^-[ \t]+\*\*ended_at\*\*:[ \t]*(.*)$", re.M),
    # machine/gpus 为可选结构化字段（训练进程经包在 start 时写入）；旧记录缺行，
    # set_field 会自动补插，check 不因缺失报错（空值视为未记录）。
    "machine": re.compile(r"^-[ \t]+\*\*machine\*\*:[ \t]*(.*)$", re.M),
    "gpus": re.compile(r"^-[ \t]+\*\*gpus\*\*:[ \t]*(.*)$", re.M),
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
    "> 本文件由 exp-mgmt 的 list 动作自动生成，请勿手工编辑；记录文件是唯一事实源。\n\n"
    "| 编号 | 状态 | 机器 | 标题 | planned_at | started_at | ended_at |\n"
    "|---|---|---|---|---|---|---|\n"
)


def _day(iso: str) -> str:
    return iso[:10] if iso else "—"


def render_index(records: list[dict]) -> str:
    lines = [
        f"| {r['id']} | {r['status']} | {r.get('machine') or '—'} | {r['slug']} "
        f"| {_day(r['planned_at'])} | {_day(r['started_at'])} | {_day(r['ended_at'])} |"
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


GIT_NET_TIMEOUT = 30  # 秒；fetch/push 网络操作超时
NO_UPSTREAM_HINT = "分支无上游远端，跳过同步（多机共享需先 `git push -u <远端> <分支>` 建立上游）"


def upstream_of(repo: Path) -> tuple[str, str] | None:
    """当前分支上游 (remote, branch)；无上游或 detached HEAD 返回 None（视为单机仓库）。"""
    r = subprocess.run(["git", "-C", str(repo), "rev-parse", "--abbrev-ref",
                        "--symbolic-full-name", "@{u}"], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    remote, _, branch = r.stdout.strip().partition("/")
    return (remote, branch) if remote and branch else None


def ahead_behind(repo: Path, ref: str) -> tuple[int, int]:
    out = git(repo, "rev-list", "--left-right", "--count", f"HEAD...{ref}").split()
    return int(out[0]), int(out[1])


RECOVERY_RECIPE = ("恢复步骤：1) git pull --rebase；冲突时 git checkout --ours -- <文件>（rebase 中 --ours=远端侧），"
                   "INDEX.md 取任意侧、恢复后用 list 重新生成；new 撞号则把自己的记录改名到下一个空闲编号（文件名+一级标题）；"
                   "2) git add <记录目录>（须覆盖改名前后两个路径）&& git rebase --continue；3) 重新推送。")


def require_syncable(repo: Path) -> None:
    """rebase 恢复进行中时拒绝执行：HEAD 游离、上游不可解析，静默跳过同步会让本机台账悄悄脱队。"""
    for name in ("rebase-merge", "rebase-apply"):
        p = subprocess.run(["git", "-C", str(repo), "rev-parse", "--git-path", name],
                           capture_output=True, text=True)
        if p.returncode == 0 and p.stdout.strip() and Path(p.stdout.strip()).exists():
            raise ExpError(f"检测到未完成的 rebase（{name} 存在），同步状态不可信，拒绝执行——"
                           "先 `git rebase --continue`（冲突解决后 git add 记录目录）或 `git rebase --abort`，再重跑本动作")


def sync_pull(repo: Path, allow_offline: bool = False) -> str:
    """多机同步第一步「拉取」：fetch + 快进到上游，让编号/状态确认基于全仓库最新记录。"""
    require_syncable(repo)
    up = upstream_of(repo)
    if not up:
        return NO_UPSTREAM_HINT
    remote, branch = up
    ref = f"{remote}/{branch}"
    try:
        r = subprocess.run(["git", "-C", str(repo), "fetch", remote],
                           capture_output=True, text=True, timeout=GIT_NET_TIMEOUT)
    except subprocess.TimeoutExpired:
        r = None
    if r is None or r.returncode != 0:
        why = "fetch 超时" if r is None else (r.stderr.strip().splitlines() or ["fetch 失败"])[0]
        if allow_offline:
            return f"远端不可达（{why}），--allow-offline 按本地状态继续"
        raise ExpError(f"无法从远端 {remote} 拉取最新记录（多台机器共用台账时必须先拉取再确认）：{why}\n"
                       "网络恢复后重试；确要离线操作加 --allow-offline（撞号风险自担）")
    ahead, behind = ahead_behind(repo, ref)
    if ahead and behind:
        raise ExpError(f"本地与 {ref} 分叉（本地多 {ahead} 个提交、远端多 {behind} 个），拒绝执行。{RECOVERY_RECIPE}")
    if behind:
        try:
            git(repo, "merge", "--ff-only", ref)
        except ExpError as e:
            raise ExpError(f"快进到 {ref} 失败（常见原因：记录目录有未提交改动与远端改动相交，先以 exp(EXP-NNN): update 提交或 stash）\n{e}")
        return f"已快进 {behind} 个提交（{ref}）"
    return f"与 {ref} 一致"


def sync_push(repo: Path, allow_offline: bool = False) -> str:
    """多机同步第三步「推送」：提交后立即推送，他机才能看到本次占号/状态；推送被拒 = 动作未完成。"""
    up = upstream_of(repo)
    if not up:
        return NO_UPSTREAM_HINT
    if allow_offline:
        return "离线模式：跳过推送——恢复网络后运行 check 并补推送，本次变更他机暂不可见"
    remote, branch = up
    try:
        r = subprocess.run(["git", "-C", str(repo), "push", remote, f"HEAD:refs/heads/{branch}"],
                           capture_output=True, text=True, timeout=GIT_NET_TIMEOUT)
    except subprocess.TimeoutExpired:
        r = None
    if r is None or r.returncode != 0:
        why = "push 超时" if r is None else (r.stderr.strip() or "push 失败")
        raise ExpError("推送未成功——本次变更只在本地，其他机器看不到，**动作未完成**；推送成功前不得做依赖本次时点的"
                       "后续动作（start 后不得写 GPU lease；finish 后不得更新 versions.json）。\n"
                       f"{RECOVERY_RECIPE}\n  git 输出：{why}")
    return f"已推送到 {remote}/{branch}"


def set_field(path: Path, field: str, value: str) -> None:
    text = path.read_text(encoding="utf-8")
    rx = FIELD_RES[field]
    if rx.search(text):
        path.write_text(rx.sub(f"- **{field}**: {value}", text, count=1), encoding="utf-8")
        return
    # 字段行缺失（典型：旧记录没有 machine/gpus 行）——锚定到正文中最后一个存在的字段行，
    # 在其后补插，而不是报错拒绝：库路径要能升级旧格式记录。
    ends = [m.end() for f in FIELD_ORDER if (m := FIELD_RES[f].search(text))]
    if not ends:
        raise ExpError(f"{path.name}: 找不到任何结构化字段行（- **status** 等），模板损坏？")
    pos = max(ends)
    path.write_text(text[:pos] + f"\n- **{field}**: {value}" + text[pos:], encoding="utf-8")


def default_template_path() -> Path:
    """实验记录模板：优先仓库副本（源仓库/插件缓存/.agents/medpowers 都有 experiments/）；
    pip 安装场景回落到随包分发的 medpowers/TEMPLATE.md。"""
    root = Path(__file__).resolve().parents[1]
    p = root / "experiments" / "TEMPLATE.md"
    if p.exists():
        return p
    pkg = Path(__file__).resolve().parent / "TEMPLATE.md"
    if pkg.exists():
        return pkg
    raise ExpError("找不到实验记录模板 experiments/TEMPLATE.md")


def resolve_dir(dir_arg: str | None = None) -> Path:
    if dir_arg:
        return Path(dir_arg).resolve()
    env = os.environ.get("MEDPOWERS_EXPERIMENTS")
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


def _insert_after_heading(text: str, heading_prefix: str, block: str) -> str:
    """把 block 插到以 heading_prefix 开头的标题行之后（追加在节首，不覆盖人工内容）；
    标题不存在则在文末补节。"""
    m = re.search(rf"^{re.escape(heading_prefix)}.*$", text, re.M)
    if not m:
        return text.rstrip("\n") + f"\n\n{heading_prefix}\n\n{block}\n"
    pos = m.end()
    return text[:pos] + "\n\n" + block + text[pos:]


def append_facts(path: Path, *, results=None, artifacts=None, note: str = "",
                 duration_s: float | None = None) -> None:
    """把训练进程自动收集的事实追加进记录：指标/时长/说明进 ## 结果，产物路径进 ## 产物。
    诊断与下一轮决策章节不动——那是智能体结合训练曲线的分析工作。"""
    result_lines = ["- **训练进程自动落账**：" +
                    (f"时长 {timedelta(seconds=int(duration_s))}" if duration_s is not None else "") +
                    (f"；说明：{note}" if note else "")]
    if isinstance(results, dict) and results:
        result_lines.append(f"  - 指标：`{json.dumps(results, ensure_ascii=False)}`")
    text = path.read_text(encoding="utf-8")
    text = _insert_after_heading(text, "## 结果", "\n".join(result_lines))
    if artifacts:
        arts = "\n".join(f"- {a}" for a in artifacts)
        text = _insert_after_heading(text, "## 产物", arts)
    path.write_text(text, encoding="utf-8")


# ---------- 训练进程内嵌库 API ----------

def start_experiment(exp: str, *, gpus=None, machine: str = "",
                     exp_dir: str | None = None, allow_offline: bool = False,
                     log=print) -> dict:
    """训练进程入口的门禁动作（全接管模式正路）：planned → running，写 started_at 与可选
    machine/gpus，提交并推送。**门禁语义：任何失败抛 ExpError，调用方（训练进程）必须
    中止，训练不得开始**——撞号、状态不符、远端不可达、分叉都属此列。"""
    d = resolve_dir(exp_dir)
    repo = require_repo(d)
    log(f"[同步] {sync_pull(repo, allow_offline)}")
    exp_id = normalize_id(exp)
    records, old_format = scan_records(d)
    if old_format:
        names = "、".join(p.name for p in old_format)
        raise ExpError(f"存在旧格式记录（{names}），先运行 migrate 再 start")
    check_duplicates(records)
    rec = find_record(records, exp_id)
    if rec["status"] != "planned":
        raise ExpError(f"{rec['path'].name}: 状态为 {rec['status']}，只有 planned 可 start")
    started = now_iso()
    set_field(rec["path"], "status", "running")
    set_field(rec["path"], "started_at", started)
    if machine:
        set_field(rec["path"], "machine", machine)
    if gpus:
        set_field(rec["path"], "gpus", ",".join(str(int(g)) for g in gpus))
    refresh_index(d, records, rec)
    commit_paths(repo, f"exp({exp_id}): start", d)
    log(f"[同步] {sync_push(repo, allow_offline)}")  # 推送失败抛 ExpError：门禁未过，训练不得开始
    log(f"{exp_id} → running（started_at={started}）。")
    return {"id": exp_id, "path": rec["path"], "started_at": started, "pushed": True}


def finish_experiment(exp: str, status: str, *, results=None, artifacts=None,
                      note: str = "", duration_s: float | None = None,
                      exp_dir: str | None = None, allow_offline: bool = False,
                      log=print) -> dict:
    """训练进程收尾（全接管模式正路）：running → done|interrupted，自动把事实（指标/
    产物路径/时长/说明）追加进记录正文，提交并推送；诊断与下一轮决策留给智能体补写。

    拉取/校验失败抛 ExpError（调用方决定降级）；**推送失败不抛**——返回
    {"pushed": False, "push_note": …}，进程即将退出，未推送提交由 check 兜底报告。"""
    if status not in ("done", "interrupted"):
        raise ExpError("status 只接受 done 或 interrupted")
    d = resolve_dir(exp_dir)
    repo = require_repo(d)
    log(f"[同步] {sync_pull(repo, allow_offline)}")
    exp_id = normalize_id(exp)
    records, _ = scan_records(d)
    check_duplicates(records)
    rec = find_record(records, exp_id)
    if rec["status"] != "running":
        raise ExpError(f"{rec['path'].name}: 状态为 {rec['status']}，只有 running 可 finish")
    if results or artifacts or note or duration_s is not None:
        append_facts(rec["path"], results=results, artifacts=artifacts,
                     note=note, duration_s=duration_s)
    set_field(rec["path"], "status", status)
    set_field(rec["path"], "ended_at", now_iso())
    refresh_index(d, records, rec)
    commit_paths(repo, f"exp({exp_id}): {status}", d)
    pushed, offline = True, False
    try:
        push_note = sync_push(repo, allow_offline)
        if allow_offline and upstream_of(repo):
            pushed, offline = False, True  # 离线豁免：主动跳过了推送，变更他机不可见
    except ExpError as e:
        pushed = False
        push_note = str(e)
        log(f"[警告] 落账已本地提交但{e}")
    if not pushed:
        return {"id": exp_id, "path": rec["path"], "status": status,
                "pushed": False, "offline": offline, "push_note": push_note}
    log(f"[同步] {push_note}")
    log(f"{exp_id} → {status}（ended_at 已填）。诊断/下一轮决策由智能体结合曲线补写。")
    return {"id": exp_id, "path": rec["path"], "status": status,
            "pushed": True, "offline": False}


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


# ---------- 旧格式迁移原语（exp_cli 的 migrate 使用） ----------

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
