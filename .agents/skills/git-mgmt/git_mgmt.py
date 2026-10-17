#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""git_mgmt.py — git-mgmt 技能脚本：medpowers 仓库自身的双远端状态概览与安全推送。

定位：放在 .agents/skills/（ZCode 等智能体的自动发现路径），面向本仓库自身的开发管理，
不随 skills/ 分发给训练仓库当算法开发技能用（安装副本只拉不推）。
远端约定（细则见同目录 SKILL.md）：
  origin  私有 GitLab —— 日常开发远端，默认推送/拉取目标；
  github  公有 GitHub —— 开源发布远端，必须用户当次明确指示（脚本层还要 --publish 双确认）才允许推送。

用法（在 medpowers 仓库任一副本内执行）：
  python3 .agents/skills/git-mgmt/git_mgmt.py status
  python3 .agents/skills/git-mgmt/git_mgmt.py push [--remote origin|github] [--publish] [--dry-run]
  python3 .agents/skills/git-mgmt/git_mgmt.py add-github --url <URL> [--force]
  python3 .agents/skills/git-mgmt/git_mgmt.py check
  python3 .agents/skills/git-mgmt/git_mgmt.py validate

安全设计：
  1) push 默认只认 origin；--remote github 不带 --publish 直接拒绝（双确认）；
  2) 不提供任何 force 选项，绝不改写远端历史；
  3) 安装副本（训练仓库 .agents/medpowers）只拉不推，检测到安装形态即拒绝 push；
  4) 推 github 前自动跑泄露面扫描（启发式），把发现摆在明面上再动手。
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# .agents/skills/git-mgmt/git_mgmt.py → parents[3] = 仓库根（源副本与安装副本深度一致）
REPO = Path(__file__).resolve().parents[3]
DEFAULT_REMOTE = "origin"   # 私有 GitLab：日常开发，未指明时一律推这里
GITHUB_REMOTE = "github"    # 公有 GitHub：开源发布，须用户当次明确指示

# 泄露面扫描的启发式模式（标签, 正则, 是否忽略大小写）。命中只代表"需要人工看"，不代表不能公开。
LEAK_PATTERNS: list[tuple[str, str, bool]] = [
    ("内网 GitLab 域名", r"gitlab\.suoyin\.info", True),
    ("本机绝对路径 /ssd/lx/", r"/ssd/lx/", False),
    ("家目录路径 /home/<user>/", r"/home/[^/\s'\"<>]+/", False),
]

# 插件名规范（Claude Code 插件约束：小写字母/数字开头，可含 . _ -，总长 ≤128）
PLUGIN_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
# runtime 模板的 5 个固定文件（README + 4 个 example.json）
RUNTIME_TEMPLATES = ("README.md", "data.example.json", "env.example.json", "notify.example.json", "versions.example.json")


def die(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def git(*args: str, check: bool = True):
    r = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        die(f"git {' '.join(args)} 失败：{(r.stderr or r.stdout).strip()}")
    return r


def current_branch() -> str:
    b = git("symbolic-ref", "--short", "HEAD", check=False).stdout.strip()
    if not b:
        die("当前处于 detached HEAD，没有可推送的分支")
    return b


def is_installed_copy() -> str | None:
    """返回安装形态描述；源副本返回 None。判据从严：疑似安装形态即按安装副本处理（宁拒勿漏）。"""
    sp = git("rev-parse", "--show-superproject-working-tree", check=False).stdout.strip()
    if sp:
        return f"submodule，隶属 {sp}"
    if REPO.name == "medpowers" and REPO.parent.name == ".agents":
        return "嵌套 clone（<训练仓库>/.agents/medpowers）"
    return None


def require_remote(name: str) -> str:
    r = git("remote", "get-url", name, check=False)
    if r.returncode != 0 or not r.stdout.strip():
        if name == GITHUB_REMOTE:
            die(f"没有 {GITHUB_REMOTE} 远端。首次配置："
                f"python3 .agents/skills/git-mgmt/git_mgmt.py add-github --url <GitHub仓库URL>")
        die(f"没有 {name} 远端。源副本请先 git remote add {name} <URL>")
    return r.stdout.strip()


def leak_scan() -> list[tuple[str, int, list[str]]]:
    """扫描 HEAD 跟踪文件中的敏感线索，返回 (标签, 命中文件数, 示例文件)。"""
    out: list[tuple[str, int, list[str]]] = []
    for label, pattern, ignore_case in LEAK_PATTERNS:
        args = ["grep", "-I", "-l", "-E", pattern, "HEAD"]
        if ignore_case:
            args.insert(1, "-i")
        r = git(*args, check=False)
        files = [l.removeprefix("HEAD:") for l in r.stdout.splitlines() if l.strip()]
        if files:
            out.append((label, len(files), files[:5]))
    return out


def print_findings(findings: list[tuple[str, int, list[str]]]) -> None:
    if not findings:
        print("[INFO] 泄露面扫描未命中启发式模式（不代表没有敏感内容，人工复核仍是必须）")
        return
    for label, n, examples in findings:
        print(f"[WARN] {label}：{n} 个文件命中，如：")
        for f in examples:
            print(f"         {f}")
        if n > len(examples):
            print(f"         ……等共 {n} 个")


def cmd_status() -> None:
    shape = is_installed_copy()
    print(f"[INFO] 副本 {REPO}" + (f"（安装副本：{shape}，只拉不推）" if shape else "（源副本）"))
    branch = current_branch()
    head = git("rev-parse", "--short", "HEAD").stdout.strip()
    print(f"[INFO] 分支 {branch} @ {head}")
    dirty = [l for l in git("status", "--porcelain=v1").stdout.splitlines() if l.strip()]
    print(f"[INFO] 工作区：{len(dirty)} 个未提交改动")

    remotes: dict[str, str] = {}
    for line in git("remote", "-v").stdout.splitlines():
        name, rest = line.split("\t", 1)
        remotes.setdefault(name, rest.rsplit(" ", 1)[0])
    if not remotes:
        print("[WARN] 没有配置任何远端（源副本可 git remote add origin <URL>）")
        return

    fr = git("fetch", "--all", "--quiet", check=False)
    if fr.returncode != 0:
        print(f"[WARN] fetch 失败（离线？），以下 ahead/behind 基于陈旧的远端引用：{(fr.stderr or fr.stdout).strip()}")

    for name in (DEFAULT_REMOTE, GITHUB_REMOTE):
        if name not in remotes:
            hint = (f"python3 .agents/skills/git-mgmt/git_mgmt.py add-github --url <URL> 配置"
                    if name == GITHUB_REMOTE else f"git remote add {name} <URL> 配置")
            print(f"[WARN] {name} 远端未配置（{hint}）")
            continue
        print(f"[INFO] {name} ← {remotes[name]}")
        if git("rev-parse", "--verify", f"refs/remotes/{name}/{branch}", check=False).returncode != 0:
            print(f"[WARN]   远端还没有 {branch} 分支（首次推送后出现）")
            continue
        cnt = git("rev-list", "--left-right", "--count", f"{name}/{branch}...HEAD").stdout.split()
        behind, ahead = int(cnt[0]), int(cnt[1])
        rel = "与远端一致" if (ahead, behind) == (0, 0) else f"本地领先 {ahead} 个提交 / 落后 {behind} 个提交"
        print(f"[INFO]   {rel}")


def cmd_push(remote: str, publish: bool, dry_run: bool) -> None:
    if remote not in (DEFAULT_REMOTE, GITHUB_REMOTE):
        die(f"远端只允许 {DEFAULT_REMOTE}（私有 GitLab，默认）或 {GITHUB_REMOTE}（公有 GitHub），收到：{remote}")
    shape = is_installed_copy()
    if shape:
        die(f"本副本是安装副本（{shape}），只拉不推——要发布内容请回源副本操作后经 {DEFAULT_REMOTE} 分发")
    if remote == GITHUB_REMOTE and not publish:
        die("推 GitHub 属于对外发布，必须同时满足：\n"
            "  1) 用户在当前对话中明确指示发布到 GitHub（提醒过、默认过、上一次同意过都不算授权）；\n"
            "  2) 命令加 --publish 双确认。\n"
            "发布前建议先跑 check 动作做泄露面扫描并与用户过一遍结果。")
    url = require_remote(remote)
    branch = current_branch()

    if remote == GITHUB_REMOTE:
        print(f"[INFO] 对外发布：将把 {branch} 的全部历史推送到公开仓库 {url}")
        print_findings(leak_scan())

    if dry_run:
        git("push", "--dry-run", remote, f"{branch}:{branch}")
        print(f"[OK] 试运行通过：{branch} → {remote}（未实际推送）")
        return
    git("push", remote, f"{branch}:{branch}")
    print(f"[OK] 已推送 {branch} → {remote}（{url}）")
    if remote == GITHUB_REMOTE:
        print("[INFO] GitHub 是公开仓库，本次已发布该分支全部历史；之后的每次推送仍需用户再次明确指示。")


def cmd_add_github(url: str, force: bool) -> None:
    if not (url.startswith("https://") or url.startswith("git@")):
        die(f"URL 形如 https://github.com/<user>/<repo>.git 或 git@github.com:<user>/<repo>.git，收到：{url}")
    r = git("remote", "get-url", GITHUB_REMOTE, check=False)
    if r.returncode == 0 and r.stdout.strip():
        cur = r.stdout.strip()
        if not force:
            die(f"{GITHUB_REMOTE} 远端已存在：{cur}\n       确认要更换请加 --force")
        git("remote", "set-url", GITHUB_REMOTE, url)
        print(f"[OK] 已更新 {GITHUB_REMOTE}：{cur} → {url}")
    else:
        git("remote", "add", GITHUB_REMOTE, url)
        print(f"[OK] 已添加 {GITHUB_REMOTE} → {url}")
    print(f"[INFO] {GITHUB_REMOTE} 是公开发布远端：只有用户当次明确指示（push --remote {GITHUB_REMOTE} --publish）才会推送。")


def cmd_check() -> None:
    print(f"[INFO] 发布前泄露面扫描（HEAD 全部跟踪文件，启发式）：{REPO}")
    print_findings(leak_scan())
    emails = sorted({e.strip()
                     for e in git("log", "--format=%ae").stdout.splitlines()
                        + git("log", "--format=%ce").stdout.splitlines()
                     if e.strip()})
    print(f"[INFO] 历史中出现的作者/提交者邮箱（push 会发布全部历史）：{'、'.join(emails)}")
    rt = [l for l in git("ls-files", "runtime/").stdout.splitlines() if l.strip()]
    if rt:
        print(f"[INFO] runtime/ 内跟踪文件 {len(rt)} 个（机器本地状态，是否随仓库公开由用户决定）：")
        for f in rt[:10]:
            print(f"         {f}")
        if len(rt) > 10:
            print(f"         ……等共 {len(rt)} 个")
    print("[INFO] 以上为启发式参考；是否公开、公开哪些内容，最终由用户决定。")


def read_json(path: Path) -> tuple[dict, str]:
    """读 JSON 文件，返回 (数据, 错误说明)；合法时错误说明为空串，数据为 dict。"""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}, "文件不存在"
    except json.JSONDecodeError as e:
        return {}, f"JSON 解析失败：{e}"
    except OSError as e:
        return {}, f"读取失败：{e}"
    if not isinstance(data, dict):
        return {}, "顶层不是 JSON 对象"
    return data, ""


def parse_frontmatter(text: str) -> dict[str, str]:
    """极简 frontmatter 解析（自包含，不引第三方库）：--- 围栏内每行取 key: value。"""
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end < 0:
        return {}
    fields: dict[str, str] = {}
    for line in text[3:end].splitlines():
        m = re.match(r"^([A-Za-z][A-Za-z0-9_-]*):\s*(.*)$", line)
        if m:
            fields[m.group(1)] = m.group(2).strip()
    return fields


def cmd_validate() -> None:
    """发布前校验：插件清单与收录边界。只读——不改任何文件、不写 git 状态，任一项 fail 退出码非 0。"""
    print(f"[INFO] 发布前校验（插件清单与收录边界，只读，逐项列出）：{REPO}")
    results: list[tuple[str, bool, str]] = []  # (检查项, 是否通过, 失败原因)

    def record(item: str, ok: bool, why: str = "") -> None:
        results.append((item, ok, why))
        print(f"{'[ok]' if ok else '[FAIL]'} {item}" + (f"——{why}" if why else ""))

    # 1) 两份插件清单存在且 JSON 合法
    plugin, plugin_err = read_json(REPO / ".claude-plugin" / "plugin.json")
    record("清单 .claude-plugin/plugin.json 存在且 JSON 合法", not plugin_err, plugin_err)
    market, market_err = read_json(REPO / ".claude-plugin" / "marketplace.json")
    record("清单 .claude-plugin/marketplace.json 存在且 JSON 合法", not market_err, market_err)

    # 2) plugin name 规范、清单一致性与组件目录
    pname = plugin.get("name")
    if plugin_err:
        record("plugin name 匹配 ^[a-z0-9][a-z0-9._-]{0,127}$", False, "plugin.json 缺失或非法，无法核对")
    else:
        ok = isinstance(pname, str) and bool(PLUGIN_NAME_RE.fullmatch(pname))
        record(f"plugin name {pname!r} 匹配 ^[a-z0-9][a-z0-9._-]{{0,127}}$", ok,
               "" if ok else "须小写字母/数字开头，仅含 a-z0-9._-，总长 ≤128")

    plugins_list = market.get("plugins")
    entry = plugins_list[0] if isinstance(plugins_list, list) and plugins_list and isinstance(plugins_list[0], dict) else None
    if market_err:
        record("marketplace 条目 name 与 plugin.json 一致", False, "marketplace.json 缺失或非法，无法核对")
    elif entry is None:
        record("marketplace 条目 name 与 plugin.json 一致", False, "plugins 为空或首条目不是对象")
    else:
        ename = entry.get("name")
        same = not plugin_err and ename == pname
        record(f"marketplace 条目 name（{ename!r}）与 plugin.json 一致", same,
               "" if same else f"plugin.json name 为 {pname!r}")

    skills_dir = REPO / "skills"
    why = []
    if market_err:
        why.append("marketplace.json 缺失或非法")
    elif entry is None:
        why.append("plugins 无有效条目")
    elif entry.get("source") != "./":
        why.append(f"source 实际为 {entry.get('source')!r}（应为 \"./\"）")
    if not skills_dir.is_dir():
        why.append("仓库根没有 skills/ 目录")
    record('marketplace 条目 source 为 "./" 且仓库根存在 skills/', not why, "；".join(why))

    # 3) skills/*/SKILL.md frontmatter：含 name/description、name 与目录名一致、description ≤1024
    item = "skills/*/SKILL.md frontmatter（含 name/description、name 与目录一致、description ≤1024 字符）"
    if not skills_dir.is_dir():
        record(item, False, "skills/ 目录不存在")
    else:
        dirs = sorted(d for d in skills_dir.iterdir() if d.is_dir())
        if not dirs:
            record(item, False, "skills/ 下没有任何技能目录")
        for d in dirs:
            fm: dict[str, str] = {}
            why = ""
            md = d / "SKILL.md"
            if not md.is_file():
                why = "SKILL.md 不存在"
            else:
                try:
                    fm = parse_frontmatter(md.read_text(encoding="utf-8"))
                except OSError as e:
                    why = f"读取失败：{e}"
            if not why and not fm:
                why = "缺少 --- 围栏 frontmatter"
            if not why:
                name, desc = fm.get("name", ""), fm.get("description", "")
                if not name:
                    why = "frontmatter 缺 name 或为空"
                elif not desc:
                    why = "frontmatter 缺 description 或为空"
                elif name != d.name:
                    why = f"name {name!r} 与目录名 {d.name!r} 不一致"
                elif len(desc) > 1024:
                    why = f"description {len(desc)} 字符 > 1024"
            record(f"skills/{d.name}/SKILL.md frontmatter", not why,
                   why or f"name={fm.get('name', '')}，description {len(fm.get('description', ''))} 字符")

    # 4) 实验记录边界：experiments/ 只有模板进 git
    tracked = [l.strip() for l in git("ls-files", "experiments/").stdout.splitlines() if l.strip()]
    ok = tracked == ["experiments/TEMPLATE.md"]
    record("git ls-files experiments/ 仅 experiments/TEMPLATE.md（实验记录不进 git）", ok,
           "" if ok else f"实际跟踪：{'、'.join(tracked) if tracked else '（无，模板也不在 git？）'}")

    # 5) runtime 模板齐全
    rt = REPO / "templates" / "runtime"
    missing = [f for f in RUNTIME_TEMPLATES if not (rt / f).is_file()] if rt.is_dir() else list(RUNTIME_TEMPLATES)
    record("templates/runtime/ 存在且 5 个文件齐全（README.md + 4 个 example.json）",
           rt.is_dir() and not missing, "" if not missing else f"缺少：{'、'.join(missing)}")

    n_fail = sum(1 for _, ok, _ in results if not ok)
    if n_fail:
        print(f"[FAIL] 校验未通过：{n_fail}/{len(results)} 项 fail，修复后再发布（本命令只读，未改动任何文件）")
        sys.exit(1)
    print(f"[INFO] 校验通过：{len(results)} 项全部 ok。本脚本为自包含启发式，发布前还可手动运行 claude plugin validate . 做官方清单校验。")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="git-mgmt：本仓库双远端状态与安全推送（规则见同目录 SKILL.md）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="双远端概览：地址、当前分支、ahead/behind、未提交改动")
    p = sub.add_parser("push", help="推送当前分支；默认 origin（私有 GitLab）")
    p.add_argument("--remote", default=DEFAULT_REMOTE, choices=[DEFAULT_REMOTE, GITHUB_REMOTE],
                   help=f"{DEFAULT_REMOTE}=私有 GitLab（默认），{GITHUB_REMOTE}=公有 GitHub（须 --publish）")
    p.add_argument("--publish", action="store_true",
                   help="确认对外发布（推 github 必带；推 origin 无需）")
    p.add_argument("--dry-run", action="store_true", help="试运行：不实际推送")
    p = sub.add_parser("add-github", help="配置/更换 github 公有远端")
    p.add_argument("--url", required=True, help="GitHub 仓库 URL（https://… 或 git@…）")
    p.add_argument("--force", action="store_true", help="远端已存在时更换其 URL")
    sub.add_parser("check", help="发布前泄露面扫描（内网域名/本机路径/邮箱/runtime 状态，启发式）")
    sub.add_parser("validate", help="发布前校验：插件清单与收录边界（只读，逐项 ok/FAIL，任一 FAIL 退出码非 0）")
    a = ap.parse_args()
    if a.cmd == "status":
        cmd_status()
    elif a.cmd == "push":
        cmd_push(a.remote, a.publish, a.dry_run)
    elif a.cmd == "add-github":
        cmd_add_github(a.url, a.force)
    elif a.cmd == "check":
        cmd_check()
    elif a.cmd == "validate":
        cmd_validate()


if __name__ == "__main__":
    main()
