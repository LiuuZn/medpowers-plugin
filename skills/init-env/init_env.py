#!/usr/bin/env python3
"""MedPowers init-env 技能脚本：GPU 静态清单 / Python 训练环境检查 / uv 初始化 / 记录保存。

子命令：
    gpu-static                  查询 GPU 静态信息（index/name/显存总量）→ JSON
    check-python [解释器]       检查 PyTorch 与 CUDA 张量可用性 → JSON（通过则退出码 0）
    init-uv [--python 版本] [--index URL]
                                用 uv 初始化共享训练环境（pyproject + .venv + torch；
                                默认走阿里 PyPI 镜像，--index 可换源）
    save                        从 stdin 读完整 env.json 内容，校验 task_profile（任务画像，
                                必答）与 gpus/gpu_availability/python_env 必填后原子写入 runtime/env.json

初始化只查静态信息；利用率/温度/进程等动态信息不在本技能范围，需要时实时 nvidia-smi。
仅用标准库。

状态写入位置：runtime 目录解析优先级为环境变量 MEDPOWERS_RUNTIME（直接指向 runtime
目录）> 任务仓库工作区（cwd 向上最近含 .git 的祖先）下的 runtime/。插件根（脚本
parents[2]）仅用于读取仓库自带资源（模板等），不承载任何状态写入。
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def workspace_root() -> Path:
    """任务仓库工作区根：从 cwd 逐级向上（含 cwd）找最近含 .git 的祖先
    （.git 为目录或文件，兼容 submodule/worktree）；找不到则用 cwd 本身。
    插件形态下本脚本位于用户级插件缓存目录，训练环境与 runtime 状态
    都必须落在任务仓库工作区，而不是脚本所在的缓存目录。"""
    cwd = Path.cwd()
    for d in (cwd, *cwd.parents):
        if (d / ".git").is_dir() or (d / ".git").is_file():
            return d
    return cwd


def resolve_runtime() -> Path:
    """runtime 状态目录：环境变量 MEDPOWERS_RUNTIME（直接指向 runtime 目录）最高；
    否则 <工作区根>/runtime。"""
    env = os.environ.get("MEDPOWERS_RUNTIME", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    return workspace_root() / "runtime"


PLUGIN_ROOT = Path(__file__).resolve().parents[2]  # 插件根：只读仓库自带资源（模板等），不写状态
WORKSPACE = workspace_root()  # 任务仓库工作区根：训练环境（pyproject/.venv）所在
RUNTIME = resolve_runtime()   # runtime 状态目录：env.json、training/、gpu/ 等
ENV_FILE = RUNTIME / "env.json"
VENV_PY = WORKSPACE / ".venv" / "bin" / "python"

# 默认 PyPI 镜像：直连 pypi.org 下载 torch 动辄数 GB 且慢，国内镜像默认必走
DEFAULT_PYPI_INDEX = "https://mirrors.aliyun.com/pypi/simple/"


def ensure_runtime() -> None:
    """初始化落盘前确保 runtime 目录结构完整（含 training/、gpu/ 子目录）；
    插件根若自带 templates/runtime/README.md 则拷入（模板缺失或目标已存在则静默
    跳过——各任务仓库以自己的副本为准，不覆盖其定制）。"""
    for d in (RUNTIME, RUNTIME / "training", RUNTIME / "gpu"):
        d.mkdir(parents=True, exist_ok=True)
    tpl = PLUGIN_ROOT / "templates" / "runtime" / "README.md"
    if tpl.is_file() and not (RUNTIME / "README.md").exists():
        shutil.copyfile(tpl, RUNTIME / "README.md")


CHECK_CODE = r'''
import json, sys
out = {"python": sys.version.split()[0], "executable": sys.executable}
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda_available"] = bool(torch.cuda.is_available())
    out["cuda_tensor_ok"] = False
    if out["cuda_available"]:
        t = torch.ones(3, device="cuda")
        out["cuda_tensor_ok"] = bool((t + t).sum().item() == 6.0)
except Exception as e:
    out["torch"] = None
    out["cuda_available"] = False
    out["cuda_tensor_ok"] = False
    out["error"] = f"{type(e).__name__}: {e}"
print(json.dumps(out, ensure_ascii=False))
'''


def now():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def cmd_gpu_static(_args):
    if subprocess.run(["which", "nvidia-smi"], capture_output=True).returncode != 0:
        sys.exit("[init-env] 未找到 nvidia-smi：本机无 GPU 环境或驱动未装。")
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"[init-env] nvidia-smi 查询失败：{r.stderr.strip()}")
    gpus = []
    for line in r.stdout.strip().splitlines():
        v = [x.strip() for x in line.split(",")]
        if len(v) < 3:
            continue
        try:
            mem = int(float(v[2]))
        except ValueError:
            mem = None
        gpus.append({"index": int(v[0]), "name": v[1], "mem_total_mb": mem})
    print(json.dumps(gpus, ensure_ascii=False, indent=2))


def check(interpreter):
    """在指定解释器上检查 torch + CUDA 张量，返回报告 dict（不退出）。"""
    interp = interpreter or (str(VENV_PY) if VENV_PY.exists() else "python3")
    try:
        r = subprocess.run([interp, "-c", CHECK_CODE], capture_output=True, text=True)
        rep = json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else {}
    except (OSError, json.JSONDecodeError, IndexError):
        rep = {}
    report = {
        "interpreter": interp,
        "default_target": "user-specified" if interpreter else "uv-venv(.venv)",
        "python": rep.get("python"),
        "torch": rep.get("torch"),
        "cuda_available": bool(rep.get("cuda_available")),
        "cuda_tensor_ok": bool(rep.get("cuda_tensor_ok")),
        "error": rep.get("error"),
    }
    return report


def cmd_check_python(args):
    report = check(args.interpreter)
    if "/" in report["interpreter"] and not Path(report["interpreter"]).exists():
        report["error"] = "解释器不存在"
    print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(0 if (report["torch"] and report["cuda_tensor_ok"]) else 1)


def cmd_init_uv(args):
    if subprocess.run(["which", "uv"], capture_output=True).returncode != 0:
        sys.exit("[init-env] 未找到 uv，请先安装：https://docs.astral.sh/uv/")
    index = args.index or DEFAULT_PYPI_INDEX
    # 环境变量双写：UV_DEFAULT_INDEX 是现行名，UV_INDEX_URL 兼容旧版 uv；
    # 同时写进 pyproject 的 [tool.uv.index]，后续 uv add/sync 不带环境变量也走同一镜像
    env = dict(os.environ, UV_DEFAULT_INDEX=index, UV_INDEX_URL=index)
    pyproject = WORKSPACE / "pyproject.toml"
    if not pyproject.exists():
        pyproject.write_text(
            '[project]\nname = "medpowers-env"\nversion = "0.1.0"\n'
            'description = "MedPowers 共享训练环境（由 skills/init-env 初始化）"\n'
            'requires-python = ">=3.11"\ndependencies = []\n'
            "\n[[tool.uv.index]]\n"
            f'url = "{index}"\n'
            "default = true\n", encoding="utf-8")
        print(f"[init-env] 已创建 pyproject.toml（PyPI 源: {index}）")
    cmds = []
    if args.python:
        cmds.append(["uv", "venv", "--python", args.python])
    cmds.append(["uv", "add", "torch"])
    for c in cmds:
        print(f"[init-env] 运行: {' '.join(c)}（PyPI 源: {index}；torch 下载量大，可能需要几分钟）")
        r = subprocess.run(c, cwd=WORKSPACE, capture_output=True, text=True, env=env)
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr, file=sys.stderr)
            sys.exit(f"[init-env] 失败: {' '.join(c)}")
    print("[init-env] uv 环境初始化完成，正在验证…")
    report = check(str(VENV_PY))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(0 if (report["torch"] and report["cuda_tensor_ok"]) else 1)


def cmd_save(_args):
    try:
        data = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        sys.exit(f"[init-env] stdin 不是合法 JSON：{e}")
    for key in ("task_profile", "gpus", "gpu_availability", "python_env"):
        if key not in data:
            sys.exit(f"[init-env] env.json 缺少必填字段: {key}（task_profile 缺失说明任务画像问卷未做，先问用户）")
    tp = data["task_profile"]
    if not isinstance(tp, dict):
        sys.exit("[init-env] task_profile 必须是对象")
    for k in ("goal", "modality", "dimension", "task_type"):
        v = tp.get(k)
        if not isinstance(v, (str, list)) or not v:
            sys.exit(f"[init-env] task_profile.{k} 缺失或为空：任务画像未答复，不得完成初始化（先向用户提问）")
    data["saved_at"] = now()
    ensure_runtime()
    tmp = ENV_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, ENV_FILE)
    print(f"[init-env] 已写入 {ENV_FILE}")


def main():
    p = argparse.ArgumentParser(description="MedPowers 初始化环境工具（详见 skills/init-env/SKILL.md）")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("gpu-static", help="查询 GPU 静态信息（index/name/显存总量）")
    cp = sub.add_parser("check-python", help="检查 PyTorch + CUDA 张量可用性")
    cp.add_argument("interpreter", nargs="?", help="解释器路径；缺省时优先 .venv/bin/python，否则 python3")
    iu = sub.add_parser("init-uv", help="用 uv 初始化共享训练环境并安装 torch（默认阿里 PyPI 镜像）")
    iu.add_argument("--python", help="venv 使用的 Python 版本（如 3.11）")
    iu.add_argument("--index", help=f"PyPI 镜像源 URL（默认 {DEFAULT_PYPI_INDEX}）")
    sub.add_parser("save", help="从 stdin 读 env.json 内容并原子写入 runtime/env.json")
    args = p.parse_args()
    {"gpu-static": cmd_gpu_static, "check-python": cmd_check_python,
     "init-uv": cmd_init_uv, "save": cmd_save}[args.cmd](args)


if __name__ == "__main__":
    main()
