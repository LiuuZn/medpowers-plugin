#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""manage_data.py — data-mgmt 技能脚本：数据根目录设定、数据集建立/登记/校验/预览/空间报告。

用法（在任务仓库工作区执行；runtime 状态落工作区 runtime/，数据默认 <工作区>/data）：
  python3 skills/data-mgmt/manage_data.py set-root <path> [--source user|default] [--answer "答复原文"]
  python3 skills/data-mgmt/manage_data.py scaffold <name> --task-type <t> [--recipe r1] [--modality CT] [--root DIR] [--force]
                                     [--link-source DIR]
  python3 skills/data-mgmt/manage_data.py import-planes <mapping.json> --dataset <ds> [--recipe r1]
                                     [--labels L1,L2] [--seg-dir DIR] [--seg-suffix S] [--size-mm H,W]
  python3 skills/data-mgmt/manage_data.py register <path>
  python3 skills/data-mgmt/manage_data.py check <path> [--recipe r1] [--deep]
  python3 skills/data-mgmt/manage_data.py preview <path> [--recipe r1] [--limit 20]
  python3 skills/data-mgmt/manage_data.py du [<path>]

scaffold / register / du 只依赖标准库；check / preview 的增强功能按需使用
numpy、nrrd（pynrrd）、Pillow，缺失时降级为文件级检查并在输出中明示。
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import re
import sys
import tempfile
from pathlib import Path

def _workspace_root() -> Path:
    """任务仓库工作区根：从 cwd 逐级向上（含 cwd）找最近含 .git 的祖先
    （.git 为目录或文件，兼容 submodule/worktree）；找不到则用 cwd 本身。
    插件形态下本脚本位于用户级插件缓存目录，runtime/data.json 等状态
    必须落在任务仓库工作区，而不是脚本所在的缓存目录。"""
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
    return _workspace_root() / "runtime"


RUNTIME = resolve_runtime()
DATA_JSON = RUNTIME / "data.json"
DEFAULT_ROOT = str(_workspace_root() / "data")

# task_type -> 布局定义（dirs 为 processed/<recipe>/ 下按任务建的结构）
TASKS = {
    "seg3d": {
        "label": "3D 分割",
        "dim": "3d",
        "ext": ".nrrd",
        "dirs": {"images": "每通道一个文件 case_0000.nrrd（gzip 压缩）",
                 "labels": "uint8 NRRD 掩码，几何与影像一致"},
        "ann": None,
        "ann_desc": "labels/<case>.nrrd：uint8，值=classes 下标（0 背景），sizes/space directions/space origin 与影像一致",
    },
    "seg2d": {
        "label": "2D 分割",
        "dim": "2d",
        "ext": ".png",
        "dirs": {"images": "2D 影像 PNG（16-bit 灰度可用），切片命名 <case>_<0123>.png",
                 "masks": "uint8 索引 PNG（mode L，值=类别号，0 背景）"},
        "ann": None,
        "ann_desc": "masks/<同名>.png：uint8 索引值，不是调色板色号",
    },
    "cls2d": {
        "label": "2D 分类",
        "dim": "2d",
        "ext": ".png",
        "dirs": {"images": "2D 影像 PNG"},
        "ann": "labels.csv",
        "ann_desc": "labels.csv：case_id,label（多标签加列；划分只看 splits/）",
    },
    "cls3d": {
        "label": "3D 分类",
        "dim": "3d",
        "ext": ".nrrd",
        "dirs": {"images": "3D 影像 NRRD（gzip 压缩，多模态用通道后缀 _0000）"},
        "ann": "labels.csv",
        "ann_desc": "labels.csv：case_id,label（多标签加列；划分只看 splits/）",
    },
    "det3d": {
        "label": "3D 包围框检测",
        "dim": "3d",
        "ext": ".nrrd",
        "dirs": {"images": "3D 影像 NRRD（gzip 压缩，多模态用通道后缀 _0000）"},
        "ann": "boxes.json",
        "ann_desc": "boxes.json：每例 [{label, center_world, size_world}]，RAS 物理 mm",
    },
    "det2d": {
        "label": "2D 包围框检测",
        "dim": "2d",
        "ext": ".png",
        "dirs": {"images": "2D 影像 PNG"},
        "ann": "annotations.json",
        "ann_desc": "annotations.json：COCO 格式（images/annotations/categories）",
    },
    "keyframe3d": {
        "label": "3D 关键帧/关键平面（与 Studio 统一）",
        "dim": "3d",
        "ext": ".nrrd",
        "dirs": {"images": "3D 影像 NRRD（gzip 压缩，多模态用通道后缀 _0000）"},
        "ann": "planes.json",
        "ann_desc": "planes.json：每例 [{label, center_world, normal_world, 可选 in_plane_up_world/size_world}]，RAS mm",
    },
    # 历史别名：旧数据集仍可 check/preview；新建一律用 keyframe3d。
    "plane3d": {
        "label": "3D 关键平面检出（历史别名）",
        "dim": "3d",
        "ext": ".nrrd",
        "dirs": {"images": "3D 影像 NRRD（gzip 压缩，多模态用通道后缀 _0000）"},
        "ann": "planes.json",
        "ann_desc": "planes.json：每例 [{label, center_world, normal_world, 可选 in_plane_up_world/size_world}]，RAS mm",
    },
}

TASK_ALIASES = {"plane3d": "keyframe3d"}

PALETTE = [(228, 26, 28), (55, 126, 184), (77, 175, 74), (152, 78, 163),
           (255, 127, 0), (255, 255, 51), (166, 86, 40), (247, 129, 191),
           (153, 153, 153), (0, 255, 255), (128, 0, 128), (0, 128, 128)]


# ---------- 基础工具 ----------

def now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


def die(msg: str) -> None:
    print(f"[FAIL] {msg}")
    sys.exit(1)


def load_json(path: Path, default=None):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        die(f"读 {path} 失败：{e}")


def atomic_write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _dep(module_name: str):
    """可选依赖：缺失时打印降级提示并返回 None（调用方必须判空）。"""
    try:
        return __import__(module_name, fromlist=["__doc__"])
    except ImportError:
        print(f"[WARN] 缺少可选依赖 {module_name}，相关检查/预览降级或跳过")
        return None


def human(nbytes: int) -> str:
    x = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if x < 1024 or unit == "TB":
            return f"{x:.1f}{unit}" if unit != "B" else f"{int(x)}B"
        x /= 1024
    return f"{x:.1f}TB"


def dir_size(p: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(p):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def get_data_root() -> str:
    dj = load_json(DATA_JSON, {}) or {}
    dr = dj.get("data_root") or {}
    path = dr.get("path")
    if not path:
        print(f"[WARN] data_root 未设定，使用默认 {DEFAULT_ROOT}（可 set-root 修改）")
        return DEFAULT_ROOT
    return path


def case_key(task: str, filename: str) -> str:
    """影像文件名 -> case 键。仅 3D NRRD 剥离多模态通道后缀 _0000（2D 切片号不剥离）。"""
    stem = Path(filename).stem
    if TASKS[task]["ext"] == ".nrrd" and re.fullmatch(r".*_\d{4}", stem):
        return stem[:-5]
    return stem


def recipes_of(ds: Path):
    pdir = ds / "processed"
    if not pdir.is_dir():
        return []
    return sorted(q.name for q in pdir.iterdir() if q.is_dir())


def images_of(rdir: Path, task: str):
    idir = rdir / "images"
    if not idir.is_dir():
        return []
    return sorted(q for q in idir.iterdir() if q.is_file() and q.suffix == TASKS[task]["ext"])


# ---------- set-root ----------

def cmd_set_root(a) -> None:
    p = Path(a.path).expanduser()
    if not p.is_absolute():
        die("data_root 必须是绝对路径")
    p.mkdir(parents=True, exist_ok=True)
    data = load_json(DATA_JSON, {}) or {}
    old = data.get("data_root") or {}
    data["data_root"] = {
        "path": str(p),
        "source": a.source,
        "answer": a.answer or old.get("answer", ""),
        "set_at": now(),
    }
    data.setdefault("datasets", {})
    atomic_write_json(DATA_JSON, data)
    print(f"[OK] data_root = {p}（source={a.source}，已写入 {DATA_JSON}）")


# ---------- scaffold ----------

ANN_TEMPLATES = {
    "labels.csv": "case_id,label\n",
    "boxes.json": {"coordinate_system": "RAS, physical mm",
                   "boxes": {"<case_id>": [{"label": "示例类别",
                                            "center_world": [0.0, 0.0, 0.0],
                                            "size_world": [10.0, 10.0, 10.0]}]}},
    "planes.json": {"coordinate_system": "RAS, physical mm",
                    "plane_def": "center_world + normal_world + 可选 in_plane_up_world / size_world",
                    "planes": {"<case_id>": [{"label": "示例平面",
                                              "center_world": [0.0, 0.0, 0.0],
                                              "normal_world": [0.0, 0.0, 1.0],
                                              "in_plane_up_world": None,
                                              "size_world": None}]}},
    "annotations.json": {"images": [], "annotations": [], "categories": []},
}

README_TMPL = """# {name} — 数据集

由 medpowers 的 data-mgmt 技能创建。层级规则：

- `raw/` 只读（原始数据，永不修改）
- `processed/<recipe>/` 训练就绪正本（3D 压缩 NRRD / 2D 无损 PNG），recipe = 预处理配方名
- `processed/<recipe>/cache/` npy 解包缓存，可随时删除（训练时自动重建）
- `splits/` train/val/test 的 case_id JSON 数组（划分唯一事实源，不复制文件）
- `preview/` 标注叠加 PNG

权威元数据见 `dataset.json`；全局登记见工作区 `runtime/data.json`。
任务类型：{label}。标注格式：{ann_desc}
"""


def cmd_scaffold(a) -> None:
    task = TASK_ALIASES.get(a.task_type, a.task_type)
    T = TASKS[task]
    root = Path(a.root).expanduser() if a.root else Path(get_data_root())
    ds = root / a.name
    rec = ds / "processed" / a.recipe
    if ds.exists() and any(ds.iterdir()) and not a.force:
        die(f"{ds} 已存在且非空；确认后在原有结构上补建可用 --force（只补缺失目录并覆盖模板文件）")
    for sub in ["raw", "splits", "preview"] + [f"processed/{a.recipe}/{d}" for d in T["dirs"]]:
        (ds / sub).mkdir(parents=True, exist_ok=True)
    if getattr(a, "link_source", None):
        src = Path(a.link_source).expanduser()
        ext = T["ext"]
        files = sorted(q for q in src.iterdir() if q.is_file() and q.suffix == ext)
        idir = rec / "images"
        n_link = 0
        for q in files:
            link = idir / q.name
            if not link.exists():
                os.symlink(str(q), str(link))
            n_link += 1
        print(f"[OK] --link-source：{n_link} 个 {ext} 已软链至 {idir}（单一正本，不复制）")
    if T["ann"]:
        annf = rec / T["ann"]
        tpl = ANN_TEMPLATES[T["ann"]]
        if isinstance(tpl, str):
            annf.write_text(tpl, encoding="utf-8")
        else:
            annf.write_text(json.dumps(tpl, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (rec / "cache").mkdir(exist_ok=True)
    (ds / "splits" / "train.json").write_text("[]\n", encoding="utf-8")
    (ds / "splits" / "val.json").write_text("[]\n", encoding="utf-8")
    (ds / "splits" / "test.json").write_text("[]\n", encoding="utf-8")
    (ds / "dataset.json").write_text(json.dumps({
        "name": a.name,
        "task_type": task,
        "task_type_label": T["label"],
        "dimension": T["dim"],
        "modality": [a.modality] if a.modality else [],
        "classes": [],
        "coordinate": "RAS, physical mm",
        "annotation_format": T["ann_desc"],
        "n_cases": None,
        "recipes": {a.recipe: "TODO: 预处理配方说明（反链资产卡 / EXP 编号）"},
        "source": "TODO: 数据来源与许可",
        "created": now(),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (ds / "README.md").write_text(README_TMPL.format(name=a.name, label=T["label"], ann_desc=T["ann_desc"]), encoding="utf-8")
    print(f"[OK] 已创建 {ds}")
    print(f"     任务类型 {task}（{T['label']}），recipe {a.recipe}")
    print(f"     下一步：放入 raw/ 或直接产出 processed/{a.recipe}/，补全 dataset.json 的 TODO 字段，"
          f"然后 check -> preview -> register")


# ---------- import-planes ----------

def cmd_import_planes(a) -> None:
    """「每例每标签一个 z 切片索引」mapping JSON -> keyframe3d planes.json。

    mapping：{case_id: {"T12": 9, ...}}，z 沿影像第 3 轴。几何由影像头换算，
    统一输出 RAS mm（LPS 影像自动翻转 x,y）。面内中心缺省影像中心，
    --seg-dir/--seg-suffix 提供时按 --labels 顺序取对应标签值掩码质心（值=1..）。
    需要 numpy + nrrd（建议任务仓库 venv 执行）。
    """
    np_ = _dep("numpy"); nrrd_ = _dep("nrrd")
    if np_ is None or nrrd_ is None:
        die("import-planes 需要 numpy 与 nrrd（任务仓库 venv 内执行）")
    ds = Path(a.dataset).expanduser().resolve()
    if not load_json(ds / "dataset.json"):
        die(f"{ds} 缺 dataset.json（先 scaffold）")
    rec = ds / "processed" / a.recipe
    idir = rec / "images"
    mapping = load_json(Path(a.mapping).expanduser())
    if not mapping:
        die(f"mapping 文件无效：{a.mapping}")
    labels = ([s.strip() for s in a.labels.split(",")] if a.labels
              else sorted({k for v in mapping.values() for k in v}))
    size_hw = [float(x) for x in a.size_mm.split(",")] if a.size_mm else [240.0, 240.0]
    seg_dir = Path(a.seg_dir).expanduser() if a.seg_dir else None
    lps2ras = np_.diag([-1.0, -1.0, 1.0])

    planes, n_c, n_p, n_fb, miss = {}, 0, 0, 0, 0
    for case, labels_z in mapping.items():
        img = idir / f"{case}.nrrd"
        if not img.exists():
            miss += 1
            continue
        hi_ = nrrd_.read_header(str(img))
        sizes = np_.asarray(hi_["sizes"], float)
        S = np_.asarray(hi_["space directions"], float)
        origin = np_.asarray(hi_.get("space origin", [0, 0, 0]), float)
        seg = None
        if seg_dir is not None:
            sp = seg_dir / f"{case}{a.seg_suffix}.nrrd"
            if sp.exists():
                arr, _ = nrrd_.read(str(sp))
                seg = arr.transpose(2, 1, 0)  # -> (Z,Y,X)
        entries = []
        for name in labels:
            if name not in labels_z:
                continue
            z_idx = int(labels_z[name])
            cy, cx = sizes[1] / 2.0, sizes[0] / 2.0
            used_seg = False
            if seg is not None:
                lab_val = 1 + labels.index(name)  # 约定：seg 值按 --labels 顺序 1..
                mm = seg == lab_val
                if mm.any():
                    _, ys_, xs_ = np_.nonzero(mm)
                    cy, cx = float(ys_.mean()), float(xs_.mean())
                    used_seg = True
            if not used_seg:
                n_fb += 1
            ras = lambda v: (lps2ras @ np_.asarray(v, float)).tolist()
            center_lps = origin + cx * S[:, 0] + cy * S[:, 1] + z_idx * S[:, 2]
            nrm = S[:, 2] / np_.linalg.norm(S[:, 2])
            up = S[:, 1] - np_.dot(S[:, 1], nrm) * nrm
            up = up / np_.linalg.norm(up)
            entries.append({"label": name,
                            "center_world": ras(center_lps),
                            "normal_world": ras(nrm),
                            "in_plane_up_world": ras(up),
                            "size_world": list(size_hw)})
            n_p += 1
        if entries:
            planes[case] = entries
            n_c += 1
    atomic_write_json(rec / "planes.json",
                      {"coordinate_system": "RAS, physical mm",
                       "plane_def": "center_world + normal_world + 可选 in_plane_up_world / size_world",
                       "planes": planes})
    print(f"[OK] import-planes：{n_c} 例 / {n_p} 平面 -> {rec / 'planes.json'}"
          f"（面内中心回退影像中心 {n_fb}，缺影像 {miss}）；"
          f"补全 dataset.json 的 annotation 字段后执行 check")

# ---------- register ----------

def cmd_register(a) -> None:
    ds = Path(a.path).expanduser().resolve()
    if not ds.is_dir():
        die(f"{ds} 不是目录")
    meta = load_json(ds / "dataset.json")
    if not meta:
        die(f"{ds} 缺 dataset.json（先 scaffold）")
    name = meta.get("name") or ds.name
    recipes = recipes_of(ds)
    data = load_json(DATA_JSON, {}) or {}
    datasets = data.setdefault("datasets", {})
    ent = datasets.get(name, {})
    ent.update({
        "path": str(ds),
        "task_type": meta.get("task_type"),
        "task_type_label": meta.get("task_type_label"),
        "dimension": meta.get("dimension"),
        "modality": meta.get("modality", []),
        "recipes": recipes,
        "dataset_json": str(ds / "dataset.json"),
    })
    ent.setdefault("first_registered_at", now())
    ent["registered_at"] = now()
    datasets[name] = ent
    atomic_write_json(DATA_JSON, data)
    print(f"[OK] 已登记数据集 {name} -> {ds}（recipes: {recipes or '无'}），写入 {DATA_JSON}")


# ---------- check ----------

def _check_vec3(v) -> bool:
    return isinstance(v, list) and len(v) == 3 and all(isinstance(q, (int, float)) for q in v)


def cmd_check(a) -> None:
    ds = Path(a.path).expanduser().resolve()
    meta = load_json(ds / "dataset.json")
    if not meta:
        die(f"{ds} 缺 dataset.json（先 scaffold）")
    task = meta.get("task_type")
    if task not in TASKS:
        die(f"dataset.json 的 task_type={task} 不在支持列表：{list(TASKS)}")
    T = TASKS[task]
    all_recipes = recipes_of(ds)
    if a.recipe:
        if a.recipe not in all_recipes:
            die(f"recipe {a.recipe} 不存在（现有：{all_recipes}）")
        targets = [a.recipe]
    else:
        targets = all_recipes
    fails, warns = [], []
    all_stems = set()

    for rec in targets:
        rdir = ds / "processed" / rec
        imgs = images_of(rdir, task)
        if not imgs:
            warns.append(f"{rec}: images/ 为空或不存在（还没放数据则忽略）")
        keys = {}
        for p in imgs:
            keys.setdefault(case_key(task, p.name), []).append(p)
        all_stems |= set(keys)

        # 分割：影像↔标注配对 + 几何
        if task in ("seg3d", "seg2d"):
            ldir = rdir / ("labels" if task == "seg3d" else "masks")
            ext = ".nrrd" if task == "seg3d" else ".png"
            labs = sorted(q for q in ldir.iterdir() if q.is_file() and q.suffix == ext) if ldir.is_dir() else []
            lab_keys = {case_key(task, q.name): q for q in labs}
            for k in keys:
                if k not in lab_keys:
                    fails.append(f"{rec}: 影像 {k} 缺对应标注")
            for k in lab_keys:
                if k not in keys:
                    fails.append(f"{rec}: 标注 {k} 无对应影像（孤儿标注）")
            if task == "seg3d":
                np_ = _dep("numpy") if any(keys) else None
                nrrd_ = _dep("nrrd") if any(keys) else None
                if np_ is None or nrrd_ is None:
                    warns.append(f"{rec}: 缺 numpy/pynrrd，跳过 NRRD 几何一致性校验（仅文件配对）")
                else:
                    for k, lp in lab_keys.items():
                        if k not in keys:
                            continue
                        ip = keys[k][0]
                        try:
                            hi = nrrd_.read_header(str(ip))
                            hl = nrrd_.read_header(str(lp))
                        except Exception as e:  # noqa: BLE001 损坏文件要逐例报告而不是整体崩
                            fails.append(f"{rec}: {k} 读 NRRD 头失败：{e}")
                            continue
                        if not np_.allclose(np_.asarray(hi.get("sizes")), np_.asarray(hl.get("sizes"))):
                            fails.append(f"{rec}: {k} sizes 不一致 {[int(q) for q in hi.get('sizes')]} vs {[int(q) for q in hl.get('sizes')]}")
                        di, dl = hi.get("space directions"), hl.get("space directions")
                        if di is None or dl is None:
                            warns.append(f"{rec}: {k} 头缺 space directions，无法比对方向")
                        elif not np_.allclose(np_.asarray(di, float), np_.asarray(dl, float)):
                            fails.append(f"{rec}: {k} space directions 不一致")
                        oi = np_.asarray(hi.get("space origin", 0), float)
                        ol = np_.asarray(hl.get("space origin", 0), float)
                        if not np_.allclose(oi, ol):
                            fails.append(f"{rec}: {k} space origin 不一致 {oi.tolist()} vs {ol.tolist()}")
                        si, sl = hi.get("space"), hl.get("space")
                        if si and sl and si != sl:
                            fails.append(f"{rec}: {k} space 不一致 {si} vs {sl}")
                        if str(hl.get("type", "")).lower() != "uint8":
                            fails.append(f"{rec}: {k} 标注类型 {hl.get('type')} ≠ uint8")
            elif task == "seg2d":
                pil = _dep("PIL.Image") if any(lab_keys) else None
                if pil is None:
                    warns.append(f"{rec}: 缺 Pillow，跳过 PNG 尺寸/mode 校验")
                else:
                    for k, lp in lab_keys.items():
                        if k not in keys:
                            continue
                        ii, ll = pil.open(keys[k][0]), pil.open(lp)
                        if ii.size != ll.size:
                            fails.append(f"{rec}: {k} 尺寸不一致 {ii.size} vs {ll.size}")
                        if ll.mode != "L":
                            fails.append(f"{rec}: {k} 掩码 mode={ll.mode}，应为 'L'（uint8 索引值，非调色板/RGB）")

        # 分类：labels.csv
        if task in ("cls2d", "cls3d"):
            annf = rdir / T["ann"]
            if not annf.is_file():
                fails.append(f"{rec}: 缺 {T['ann']}")
            else:
                with open(annf, newline="", encoding="utf-8") as f:
                    rdr = csv.DictReader(f)
                    cols = set(rdr.fieldnames or [])
                    if not {"case_id", "label"} <= cols:
                        fails.append(f"{rec}: {T['ann']} 表头缺 case_id/label 列")
                        ids = set()
                    else:
                        ids = {row["case_id"] for row in rdr}
                for k in keys:
                    if k not in ids:
                        fails.append(f"{rec}: 影像 {k} 在 {T['ann']} 无标签行")
                for k in ids - set(keys):
                    fails.append(f"{rec}: {T['ann']} 的 {k} 无对应影像")

        # det3d：boxes.json
        if task == "det3d":
            j = load_json(rdir / T["ann"])
            if not j:
                fails.append(f"{rec}: 缺 {T['ann']}")
            else:
                boxes = j.get("boxes", {})
                for k in keys:
                    if k not in boxes:
                        fails.append(f"{rec}: 影像 {k} 在 boxes.json 缺框标注")
                for k, bl in boxes.items():
                    if k not in keys:
                        fails.append(f"{rec}: boxes.json 的 {k} 无对应影像")
                        continue
                    if not isinstance(bl, list) or not bl:
                        warns.append(f"{rec}: {k} 框列表为空")
                        continue
                    for b in bl:
                        if not isinstance(b, dict) or "label" not in b \
                                or not _check_vec3(b.get("center_world")) or not _check_vec3(b.get("size_world")):
                            fails.append(f"{rec}: {k} 存在字段不全的框（需 label/center_world/size_world 各 3 数）")
                            break

        # keyframe3d（历史别名 plane3d）：planes.json
        if task in ("plane3d", "keyframe3d"):
            j = load_json(rdir / T["ann"])
            if not j:
                fails.append(f"{rec}: 缺 {T['ann']}")
            else:
                planes = j.get("planes", {})
                for k in keys:
                    if k not in planes:
                        fails.append(f"{rec}: 影像 {k} 在 planes.json 缺平面标注")
                for k, pl_list in planes.items():
                    if k not in keys:
                        fails.append(f"{rec}: planes.json 的 {k} 无对应影像")
                        continue
                    if not isinstance(pl_list, list) or not pl_list:
                        warns.append(f"{rec}: {k} 平面列表为空")
                        continue
                    for pl in pl_list:
                        if not isinstance(pl, dict) or not _check_vec3(pl.get("center_world")) \
                                or not _check_vec3(pl.get("normal_world")):
                            fails.append(f"{rec}: {k} 存在字段不全的平面（需 center_world/normal_world 各 3 数）")
                            break
                        n = pl["normal_world"]
                        norm2 = (n[0] ** 2 + n[1] ** 2 + n[2] ** 2) ** 0.5
                        if not 0.9 <= norm2 <= 1.1:
                            warns.append(f"{rec}: {k} normal_world 模长 {norm2:.3f} 偏离 1（应为单位向量）")
                        up = pl.get("in_plane_up_world")
                        if up is not None and not _check_vec3(up):
                            fails.append(f"{rec}: {k} in_plane_up_world 不是 3 数")
                        sz = pl.get("size_world")
                        if sz is not None and (not isinstance(sz, list) or len(sz) != 2):
                            fails.append(f"{rec}: {k} size_world 应为 [高mm, 宽mm]")

        # det2d：COCO
        if task == "det2d":
            j = load_json(rdir / T["ann"])
            if not j:
                fails.append(f"{rec}: 缺 {T['ann']}")
            else:
                files = {p.name for p in imgs}
                entries = j.get("images", [])
                ids = set()
                for e in entries:
                    ids.add(e.get("id"))
                    if e.get("file_name") not in files:
                        fails.append(f"{rec}: COCO images 引用了不存在的文件 {e.get('file_name')}")
                for f_ in files:
                    if f_ not in {e.get("file_name") for e in entries}:
                        warns.append(f"{rec}: {f_} 未登记进 COCO images")
                for an in j.get("annotations", []):
                    if an.get("image_id") not in ids:
                        fails.append(f"{rec}: annotation {an.get('id')} 引用不存在的 image_id")
                if not j.get("categories"):
                    warns.append(f"{rec}: COCO categories 为空")

        # --deep：分割标注值域
        if a.deep and task in ("seg3d", "seg2d") and keys:
            np_ = _dep("numpy")
            nrrd_ = _dep("nrrd") if task == "seg3d" else None
            pil = _dep("PIL.Image") if task == "seg2d" else None
            classes = meta.get("classes") or []
            vals_all = set()
            try:
                if task == "seg3d" and np_ is not None and nrrd_ is not None:
                    for k in list(keys)[:200]:
                        lp = rdir / "labels" / f"{k}.nrrd"
                        if lp.is_file():
                            arr, _ = nrrd_.read(str(lp))
                            vals_all |= set(np_.unique(arr).tolist())
                elif task == "seg2d" and pil is not None:
                    for p in imgs[:500]:
                        mp = rdir / "masks" / p.name
                        if mp.is_file():
                            vals_all |= set(pil.open(mp).getdata())
                else:
                    warns.append("--deep 需要对应依赖（3D: numpy+nrrd；2D: Pillow），已跳过")
                    vals_all = None
            except Exception as e:  # noqa: BLE001
                warns.append(f"--deep 读标注失败：{e}")
                vals_all = None
            if vals_all is not None and vals_all:
                extra = sorted(v for v in vals_all if not isinstance(v, (int,)) or v < 0 or v > 255)
                if extra:
                    fails.append(f"{rec}: 标注出现非 uint8 值 {extra[:8]}")
                if classes:
                    bad = [v for v in vals_all if isinstance(v, int) and v >= len(classes)]
                    if bad:
                        fails.append(f"{rec}: 标注值 {sorted(set(bad))} 超出 classes 下标范围（0..{len(classes) - 1}）")
                print(f"[INFO] {rec}: 标注值域 {sorted(v for v in vals_all if isinstance(v, int))[:20]}")

    # splits 引用有效性
    spdir = ds / "splits"
    if spdir.is_dir():
        for f_ in sorted(spdir.glob("*.json")):
            arr = load_json(f_)
            if not isinstance(arr, list) or not all(isinstance(x, str) for x in arr):
                fails.append(f"splits/{f_.name} 应为 case_id 字符串数组")
                continue
            bad = [x for x in arr if x not in all_stems]
            if bad:
                fails.append(f"splits/{f_.name} 引用不存在的 case：{bad[:5]}{'...' if len(bad) > 5 else ''}")

    for w in warns:
        print(f"[WARN] {w}")
    for f_ in fails:
        print(f"[FAIL] {f_}")
    print(f"\n校验完成：{len(fails)} FAIL / {len(warns)} WARN（任务 {task}，recipes {targets}）")
    if fails:
        sys.exit(1)
    print("[OK] 通过")


# ---------- preview ----------

def _to8(np_, a):
    lo, hi = np_.percentile(a, [1, 99])
    if hi <= lo:
        hi = lo + 1.0
    return ((np_.clip(a, lo, hi) - lo) / (hi - lo) * 255.0).astype("uint8")


def _overlay(np_, sl, ls, pil_image):
    rgb = np_.dstack([_to8(np_, sl)] * 3).astype("float32")
    if ls is not None:
        for v in np_.unique(ls):
            if v == 0:
                continue
            m = ls == v
            c = np_.array(PALETTE[int(v) % len(PALETTE)], "float32")
            rgb[m] = rgb[m] * 0.6 + c * 0.4
    return pil_image.fromarray(rgb.astype("uint8"))


def _pv_seg2d(pv, rdir, imgs, task, limit):
    pil = _dep("PIL.Image")
    ops = _dep("PIL.ImageOps")
    if pil is None or ops is None:
        return 0
    n = 0
    for ip in imgs[:limit]:
        mp = rdir / "masks" / ip.name
        if not mp.is_file():
            continue
        base = ops.autocontrast(pil.open(ip).convert("L")).convert("RGB")
        m = pil.open(mp)
        for v in sorted(set(m.getdata())):
            if v == 0:
                continue
            solid = pil.new("RGB", m.size, PALETTE[int(v) % len(PALETTE)])
            sel = m.point(lambda x, v=v: 255 if x == v else 0)
            base = pil.composite(pil.blend(base, solid, 0.4), base, sel)
        base.save(pv / f"seg2d-{ip.stem}.png")
        n += 1
    return n


def _pv_cls(pv, rdir, imgs, task, limit):
    pil = _dep("PIL.Image")
    ops = _dep("PIL.ImageOps")
    draw_mod = _dep("PIL.ImageDraw")
    if pil is None or ops is None or draw_mod is None:
        return 0
    labels = {}
    annf = rdir / "labels.csv"
    if annf.is_file():
        with open(annf, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                labels[row["case_id"]] = row.get("label", "?")
    cells = [(ops.autocontrast(pil.open(ip).convert("L")).resize((128, 128)),
              labels.get(case_key(task, ip.name), "?")) for ip in imgs[:limit]]
    if not cells:
        return 0
    cols = 4
    rows = (len(cells) + cols - 1) // cols
    canvas = pil.new("RGB", (cols * 132, rows * 150), (24, 24, 24))
    d = draw_mod.Draw(canvas)
    for i, (im, lab) in enumerate(cells):
        x, y = (i % cols) * 132 + 2, (i // cols) * 150 + 2
        canvas.paste(im, (x, y))
        d.text((x, y + 130), str(lab)[:20], fill=(255, 255, 0))
    canvas.save(pv / "cls-grid.png")
    return 1


def _pv_det2d(pv, rdir, imgs, task, limit):
    pil = _dep("PIL.Image")
    draw_mod = _dep("PIL.ImageDraw")
    if pil is None or draw_mod is None:
        return 0
    j = load_json(rdir / "annotations.json") or {}
    cats = {c.get("id"): c.get("name", "?") for c in j.get("categories", [])}
    by_img = {}
    for an in j.get("annotations", []):
        by_img.setdefault(an.get("image_id"), []).append(an)
    n = 0
    for e in j.get("images", [])[:limit]:
        fp = rdir / "images" / e.get("file_name", "")
        if not fp.is_file():
            continue
        im = pil.open(fp).convert("RGB")
        d = draw_mod.Draw(im)
        for an in by_img.get(e.get("id"), []):
            x, y, w, h = an.get("bbox", [0, 0, 0, 0])
            d.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=2)
            d.text((x + 2, max(y - 10, 0)), cats.get(an.get("category_id"), "?"), fill=(255, 255, 0))
        im.save(pv / f"det2d-{Path(e.get('file_name', 'x')).stem}.png")
        n += 1
    return n


def _pv_seg3d(pv, rdir, imgs, task, limit):
    np_ = _dep("numpy")
    nrrd_ = _dep("nrrd")
    pil = _dep("PIL.Image")
    if np_ is None or nrrd_ is None or pil is None:
        return 0
    n = 0
    seen = set()
    for ip in imgs[:limit]:
        k = case_key(task, ip.name)
        if k in seen:
            continue
        seen.add(k)
        lp = rdir / "labels" / f"{k}.nrrd"
        if not lp.is_file():
            continue
        arr, _ = nrrd_.read(str(ip))
        lab, _ = nrrd_.read(str(lp))
        axis = int(np_.argmax(arr.shape))
        idx = arr.shape[axis] // 2
        sl = np_.take(arr, idx, axis=axis)
        ls = np_.take(lab, idx, axis=axis) if lab.shape == arr.shape else None
        _overlay(np_, sl, ls, pil).save(pv / f"seg3d-{k}.png")
        n += 1
    return n


def _world_to_img_space(v, header):
    """世界坐标向量 RAS → 影像 header 声明的坐标系（标注统一存 RAS；LPS 影像取 x,y 取反）。"""
    space = str(header.get("space") or "").lower()
    v = list(v)
    if "left-posterior-superior" in space or space == "lps":
        v = [-v[0], -v[1], v[2]]
    return v


def _pv_det3d(pv, rdir, imgs, task, limit):
    np_ = _dep("numpy")
    nrrd_ = _dep("nrrd")
    pil = _dep("PIL.Image")
    draw_mod = _dep("PIL.ImageDraw")
    if np_ is None or nrrd_ is None or pil is None or draw_mod is None:
        return 0
    j = load_json(rdir / "boxes.json") or {}
    n = 0
    seen = set()
    for ip in imgs[:limit]:
        k = case_key(task, ip.name)
        if k in seen:
            continue
        seen.add(k)
        bl = j.get("boxes", {}).get(k)
        if not bl:
            continue
        arr, h = nrrd_.read(str(ip))
        S = np_.asarray(h["space directions"], float)  # 每行 = 该轴每体素的毫米向量
        origin = np_.asarray(h.get("space origin", np_.zeros(3)), float)
        Sinv = np_.linalg.inv(S)
        vox_c = np_.array([_world_to_img_space(b["center_world"], h) for b in bl], float) - origin
        vox_c = vox_c @ Sinv
        # 世界轴对齐框的近似：各轴半长（体素）= size/2 / 该轴体素毫米长度
        vox_half = np_.abs(np_.array([b["size_world"] for b in bl], float)) / 2.0 / np_.linalg.norm(S, axis=1)
        for axis in range(3):
            mid = arr.shape[axis] // 2
            sl = _to8(np_, np_.take(arr, mid, axis=axis))
            im = pil.fromarray(np_.dstack([sl] * 3))
            others = [q for q in range(3) if q != axis]  # 切片数组 行=others[0] 列=others[1]
            scale = max(1.0, max(im.size) / 512.0)
            if scale > 1.0:
                im = im.resize((int(im.size[0] / scale), int(im.size[1] / scale)))
            d = draw_mod.Draw(im)
            for i in range(len(bl)):
                if abs(vox_c[i, axis] - mid) > vox_half[i, axis]:
                    continue
                y0 = (vox_c[i, others[0]] - vox_half[i, others[0]]) / scale
                y1 = (vox_c[i, others[0]] + vox_half[i, others[0]]) / scale
                x0 = (vox_c[i, others[1]] - vox_half[i, others[1]]) / scale
                x1 = (vox_c[i, others[1]] + vox_half[i, others[1]]) / scale
                d.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
            im.save(pv / f"det3d-{k}-ax{axis}.png")
        n += 1
    return n


def _pv_plane3d(pv, rdir, imgs, task, limit):
    np_ = _dep("numpy")
    nrrd_ = _dep("nrrd")
    pil = _dep("PIL.Image")
    if np_ is None or nrrd_ is None or pil is None:
        return 0
    j = load_json(rdir / "planes.json") or {}
    H = W = 256
    n = 0
    seen = set()
    for ip in imgs[:limit]:
        k = case_key(task, ip.name)
        if k in seen:
            continue
        seen.add(k)
        pl_list = j.get("planes", {}).get(k)
        if not pl_list:
            continue
        arr, h = nrrd_.read(str(ip))
        S = np_.asarray(h["space directions"], float)
        origin = np_.asarray(h.get("space origin", np_.zeros(3)), float)
        Sinv = np_.linalg.inv(S)
        for pi, pl in enumerate(pl_list):
            c = np_.array(_world_to_img_space(pl["center_world"], h), float)
            nv = np_.array(_world_to_img_space(pl["normal_world"], h), float)
            nv = nv / np_.linalg.norm(nv)
            up = pl.get("in_plane_up_world")
            if up is not None:
                u = np_.array(_world_to_img_space(up, h), float)
                u = u - np_.dot(u, nv) * nv
            else:
                e = np_.zeros(3)
                e[int(np_.argmin(np_.abs(nv)))] = 1.0
                u = e - np_.dot(e, nv) * nv
            un = np_.linalg.norm(u)
            if un < 1e-6:
                continue
            u = u / un
            v = np_.cross(nv, u)
            eh, ew = pl.get("size_world") or (200.0, 200.0)
            a = (np_.arange(H) / (H - 1) - 0.5) * float(eh)
            b = (np_.arange(W) / (W - 1) - 0.5) * float(ew)
            G = c[None, None, :] + a[:, None, None] * u[None, None, :] + b[None, :, None] * v[None, None, :]
            vox = (G - origin) @ Sinv
            idx = np_.rint(vox).astype(int)
            shape = np_.array(arr.shape)
            valid = np_.all((idx >= 0) & (idx < shape), axis=-1)
            out = np_.zeros((H, W), arr.dtype)
            out[valid] = arr[idx[valid, 0], idx[valid, 1], idx[valid, 2]]
            pil.fromarray(_to8(np_, out)).save(pv / f"keyframe3d-{k}-{pl.get('label') or pi}.png")
        n += 1
    return n


def cmd_preview(a) -> None:
    ds = Path(a.path).expanduser().resolve()
    meta = load_json(ds / "dataset.json")
    if not meta:
        die(f"{ds} 缺 dataset.json")
    task = meta.get("task_type")
    if task not in TASKS:
        die(f"task_type={task} 不支持")
    all_recipes = recipes_of(ds)
    if not all_recipes:
        die("processed/ 下没有 recipe 目录")
    rec = a.recipe or all_recipes[0]
    if rec not in all_recipes:
        die(f"recipe {rec} 不存在（现有：{all_recipes}）")
    rdir = ds / "processed" / rec
    imgs = images_of(rdir, task)
    if not imgs:
        die(f"{rdir / 'images'} 为空")
    pv = ds / "preview"
    pv.mkdir(exist_ok=True)
    fn = {"seg2d": _pv_seg2d, "cls2d": _pv_cls, "cls3d": _pv_cls,
          "det2d": _pv_det2d, "seg3d": _pv_seg3d, "det3d": _pv_det3d,
          "keyframe3d": _pv_plane3d, "plane3d": _pv_plane3d}[task]
    n = fn(pv, rdir, imgs, task, a.limit)
    print(f"[OK] preview：{n} 例已输出到 {pv}（recipe {rec}，任务 {task}）" if n
          else f"[WARN] 没有生成任何预览（数据或依赖缺失，见上方提示）")


# ---------- du ----------

def cmd_du(a) -> None:
    target = Path(a.path).expanduser() if a.path else Path(get_data_root())
    if not target.exists():
        die(f"{target} 不存在")
    dsets = [target] if (target / "dataset.json").is_file() else \
        sorted(q for q in target.iterdir() if q.is_dir() and (q / "dataset.json").is_file())
    if not dsets:
        print(f"[WARN] {target} 下没有含 dataset.json 的数据集目录")
    cache_total = 0
    for ds in dsets:
        print(f"\n{ds}")
        if (ds / "raw").is_dir():
            print(f"  raw            {human(dir_size(ds / 'raw'))}")
        if (ds / "processed").is_dir():
            for r in sorted(q for q in (ds / "processed").iterdir() if q.is_dir()):
                c = r / "cache"
                if c.is_dir():
                    cs = dir_size(c)
                    cache_total += cs
                    print(f"  processed/{r.name:<6} {human(dir_size(r))}（其中 cache {human(cs)}，可删）")
                else:
                    print(f"  processed/{r.name:<6} {human(dir_size(r))}")
        for extra in ("preview", "splits"):
            if (ds / extra).is_dir():
                print(f"  {extra:<14} {human(dir_size(ds / extra))}")
        print(f"  合计            {human(dir_size(ds))}")
    print(f"\n可回收 cache 合计：{human(cache_total)}（删除后训练时自动重建）")


# ---------- main ----------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("set-root", help="设定数据根目录（写 runtime/data.json）")
    sp.add_argument("path")
    sp.add_argument("--source", default="user", choices=["user", "default"])
    sp.add_argument("--answer", default="", help="用户答复原文（溯源用）")
    sp.set_defaults(func=cmd_set_root)

    sp = sub.add_parser("scaffold", help="建数据集骨架 + dataset.json + 标注模板")
    sp.add_argument("name", help="数据集名（如 lung_seg）")
    sp.add_argument("--task-type", required=True, choices=list(TASKS))
    sp.add_argument("--recipe", default="r0")
    sp.add_argument("--modality", default=None, help="如 CT / MRI")
    sp.add_argument("--root", default=None, help="覆盖 data_root（默认取 runtime/data.json）")
    sp.add_argument("--force", action="store_true", help="目录已存在时补建缺失结构并覆盖模板文件")
    sp.add_argument("--link-source", default=None,
                    help="软链登记既有影像目录（<case>.nrrd）至 processed/<recipe>/images，不复制")
    sp.set_defaults(func=cmd_scaffold)

    sp = sub.add_parser("import-planes", help="z 索引 mapping JSON -> keyframe3d planes.json")
    sp.add_argument("mapping", help="mapping JSON：{case: {label: z_index}}")
    sp.add_argument("--dataset", required=True, help="数据集目录（需已 scaffold 且 images 就位）")
    sp.add_argument("--recipe", default="r0")
    sp.add_argument("--labels", default=None, help="逗号分隔标签序（缺省=mapping 内标签并集，排序）")
    sp.add_argument("--seg-dir", default=None, help="可选：每例分割 nrrd 目录（面内中心取掩码质心）")
    sp.add_argument("--seg-suffix", default="", help="分割文件名 = <case><suffix>.nrrd")
    sp.add_argument("--size-mm", default=None, help="平面范围 [高mm,宽mm]，缺省 240,240")
    sp.set_defaults(func=cmd_import_planes)

    sp = sub.add_parser("register", help="登记数据集到 runtime/data.json")
    sp.add_argument("path")
    sp.set_defaults(func=cmd_register)

    sp = sub.add_parser("check", help="校验清单/配对/几何一致性/标注格式/splits")
    sp.add_argument("path")
    sp.add_argument("--recipe", default=None)
    sp.add_argument("--deep", action="store_true", help="额外扫描分割标注值域")
    sp.set_defaults(func=cmd_check)

    sp = sub.add_parser("preview", help="生成标注叠加 PNG 到 preview/")
    sp.add_argument("path")
    sp.add_argument("--recipe", default=None)
    sp.add_argument("--limit", type=int, default=20)
    sp.set_defaults(func=cmd_preview)

    sp = sub.add_parser("du", help="空间报告（raw/processed/cache/preview）")
    sp.add_argument("path", nargs="?", default=None, help="数据集目录或根目录（默认 data_root）")
    sp.set_defaults(func=cmd_du)

    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
