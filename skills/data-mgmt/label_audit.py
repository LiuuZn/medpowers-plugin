"""label_audit — 标注上限诊断工具集 [工具]

配套方法论：methods/label-ceiling-diagnosis.md（模型平台期先测标注一致性上限）。
本文件是技能工具而非资产条目：无论文支撑四问，规范按 assets/CODE_STYLE.md 的
自包含/类型标注/自测要求执行（自测块见文件末，裸跑本文件即完成自检）。

三个子命令：
  profile      柱内逐片封闭面积剖面：沿 z 累加提取目标柱（最大连通域+填洞），
               柱内逐片取最大连通域填洞面积。适用于近似平行于某轴的管状/柱状
               结构（脊柱、长骨、血管等），剖面呈周期峰（结构最大截面）谷（间隙）。
  rank-test    排名检验：标注索引在特征曲线邻域窗内的排名分布（含 z 翻转对照）。
  review-sheet 审片图：邻域切片渲染成带绝对索引编号的拼条 PNG，供人眼指认校准。

布局约定（numpy 数组）：
  体数据读入后经 `np.moveaxis(v, axis, -1)` 把『厚度轴』移到末轴，
  记为 `(X, Y, S)`——S 为逐片剖面的索引轴（下标 s 对应输出剖面 p[s]）。
  剖面/特征曲线为一维 `(S,)`，取值非负（面积计数）；zscored 通道均值 0、标准差 1。

数值稳定：逐例标准化使用 `(ps - mean) / (std + 1e-6)`——1e-6 远小于正常剖面
标准差（数十至数百像素²），仅防常数剖面的零除退化。

依赖：numpy、scipy 必需；nrrd（profile/review-sheet 读文件）与 Pillow
（review-sheet）为惰性导入，缺失时报错并指明。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

__all__ = ["load_vol", "column_profile", "rank_stats", "cmd_profile",
           "cmd_rank_test", "cmd_review_sheet", "run_selftest", "main"]

_EPS_STD = 1e-6  # 逐例标准化的零除保护（见文件头数值稳定说明）


def load_vol(path: str | Path, axis: int = 2) -> np.ndarray:
    """读 nrrd 体数据并把『厚度轴』移到末轴。

    参数：path 为 nrrd 文件路径；axis 为原数组中厚度轴的轴号（nrrd 读入
    (X, Y, Z) 时缺省 2）。
    返回：np.ndarray，布局 (X, Y, S)——S 为原厚度轴，逐片剖面沿 S 索引。
    """
    try:
        import nrrd
    except ImportError as e:
        raise SystemExit("[label_audit] 读 nrrd 需要 nrrd 库（任务仓库 venv 内执行）") from e
    vol, _ = nrrd.read(str(path))
    return np.moveaxis(vol, axis, -1)


def column_profile(vol: np.ndarray, threshold: float, column_frac: float = 0.25,
                   smooth_k: int = 4) -> np.ndarray | None:
    """柱内逐片封闭面积剖面（周期峰谷特征）。

    参数：
      vol: (X, Y, S) 体数据（z/厚度轴在末轴）；
      threshold: 组织阈值（与 vol 同单位；HU 影像常用 80-300，取低值以包含骨髓，
        使目标结构为实心连通域）；
      column_frac: 柱判定阈值——XY 投影计数 ≥ column_frac×S 的像素视为柱（0.25
        表示结构需在四分之一以上切片出现，滤除斜行/短暂经过的伴行结构）；
      smooth_k: 剖面滑动平均窗宽（片），压制逐片噪声、保留结构周期峰谷。
    返回：(S, 3) 数组，列 = [raw, smoothed, zscored]；raw/smoothed ≥ 0（面积计数），
      zscored 均值 0、标准差 1（eps 保护）。无法提取柱（全空/阈值过高）时返回 None。
    """
    s_n = vol.shape[-1]
    mask = vol > threshold
    proj = mask.sum(axis=-1).astype(float)  # 沿 z 累加 → (X, Y) 二维投影
    col = proj > column_frac * s_n
    lab, _ = ndimage.label(col)
    if lab.max() == 0:
        return None
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0  # 背景不参与最大域竞选
    column = ndimage.binary_fill_holes(lab == np.argmax(sizes))
    p = np.zeros(s_n)
    for s in range(s_n):
        mm = mask[:, :, s] & column
        if not mm.any():
            continue
        lab, _ = ndimage.label(mm)
        if lab.max() == 0:
            continue
        sz = np.bincount(lab.ravel())
        sz[0] = 0
        p[s] = ndimage.binary_fill_holes(lab == np.argmax(sz)).sum()
    ker = np.ones(smooth_k) / smooth_k
    ps = np.convolve(p, ker, mode="same")
    pz = (ps - ps.mean()) / (ps.std() + _EPS_STD)
    return np.stack([p, ps, pz], 1)


def rank_stats(profile: np.ndarray, labeled_z: int,
               window: int = 5) -> tuple[int, bool, bool]:
    """单个标注片的排名检验（含 z 翻转对照在外层完成）。

    参数：profile (S,) 特征曲线；labeled_z 标注索引；window 邻域半宽。
    返回：(rank, is_argmax, in_range)——rank 为标注片在窗内的严格降序排名
      （0=窗内最大）；is_argmax 指示标注片是否为窗内 argmax；in_range 指示
      标注索引是否落在数组有效范围内。
    """
    n = len(profile)
    if not (0 <= labeled_z < n):
        return -1, False, False
    lo, hi = max(0, labeled_z - window), min(n - 1, labeled_z + window)
    seg = profile[lo:hi + 1]
    if len(seg) == 0:
        return -1, False, False
    rank = int((seg > profile[labeled_z]).sum())
    return rank, lo + int(np.argmax(seg)) == labeled_z, True


def cmd_profile(a: argparse.Namespace) -> None:
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.images).glob("*.nrrd"))
    if not files:
        sys.exit(f"[profile] {a.images} 下无 nrrd")
    done = 0
    for f in files:
        case = f.stem
        of = out / f"{case}.npy"
        if of.exists():
            continue
        vol = load_vol(f, a.axis)
        r = column_profile(vol.astype(np.float32), a.threshold, a.column_frac, a.smooth_k)
        if r is None:
            print(f"[WARN] {case}: 未提取到柱 ROI（阈值过高或结构过小）")
            continue
        r.astype(np.float16).tofile(of)
        done += 1
        if done % 200 == 0:
            print(f"  {done}/{len(files)}", flush=True)
    print(f"[OK] profile：{done}/{len(files)} 例 -> {out}")


def cmd_rank_test(a: argparse.Namespace) -> None:
    labels = json.loads(Path(a.labels).read_text())
    fd = Path(a.feats_dir)
    dist: dict[str, dict[str, list[int]]] = {}
    arg_hits: dict[str, dict[str, list[bool]]] = {}
    for case, targets in labels.items():
        f = fd / f"{case}.npy"
        if not f.exists():
            continue
        prof = np.fromfile(f, np.float16).reshape(-1, 3)[:, 1].astype(np.float32)
        for t, z in targets.items():
            for tag, zz in (("正", int(z)), ("翻", len(prof) - 1 - int(z))):
                rank, hit, ok = rank_stats(prof, zz, a.window)
                if not ok:
                    continue
                dist.setdefault(tag, {}).setdefault(t, []).append(rank)
                arg_hits.setdefault(tag, {}).setdefault(t, []).append(hit)
    for tag in dist:
        for t, v in sorted(dist[tag].items()):
            v = np.array(v)
            ah = float(np.mean(arg_hits[tag][t])) if arg_hits[tag].get(t) else float("nan")
            print(f"[{tag}] {t}: 排名 mean {v.mean():.2f}（均匀期望≈{a.window}）| "
                  f"rank0 {np.mean(v == 0):.3f} | 邻域 argmax 命中 {ah:.3f} (n={len(v)})")
    print("判读：排名≈均匀 → 该特征与标注无关；rank0 高但 argmax 低 → 平台平票（需平台中心解码/仲裁）。")


def cmd_review_sheet(a: argparse.Namespace) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as e:
        raise SystemExit("[review-sheet] 需要 Pillow（任务仓库 venv 内执行）") from e
    vp = Path(a.images) / f"{a.case}.nrrd"
    if not vp.exists():
        sys.exit(f"[review-sheet] {vp} 不存在")
    vol = load_vol(vp, a.axis)
    n = vol.shape[-1]
    labels = json.loads(Path(a.labels).read_text()).get(a.case, {}) if a.labels else {}
    lo = a.z0 if a.z0 is not None else 0
    hi = a.z1 if a.z1 is not None else n - 1
    tiles = []
    for s in range(lo, hi + 1):
        img = np.clip(vol[:, :, s].astype(np.float32) /
                      max(float(np.percentile(vol, 99.5)), 1.0), 0.0, 1.0)
        img = (img * 255).astype(np.uint8)
        im = Image.fromarray(img).resize((300, 300), Image.LANCZOS).convert("RGB")
        canvas = Image.new("RGB", (304, 330), (20, 20, 20))
        canvas.paste(im, (2, 2))
        d = ImageDraw.Draw(canvas)
        d.text((105, 302), f"slice {s}", fill=(255, 255, 0))
        for t, z in labels.items():
            if z == s:
                d.rectangle([0, 0, 303, 303], outline=(255, 80, 80), width=3)
                d.text((210, 302), t, fill=(255, 80, 80))
        tiles.append(canvas)
    row = Image.new("RGB", (304 * len(tiles), 330), (0, 0, 0))
    for i, tl in enumerate(tiles):
        row.paste(tl, (i * 304, 0))
    out = Path(a.out) if a.out else Path(f"review_{a.case}.png")
    row.save(out)
    print(f"[OK] 审片图 {len(tiles)} 片（{lo}-{hi}）-> {out}"
          + ("（红框=标注片）" if labels else ""))


def run_selftest() -> None:
    """内置自检：合成『周期椎体柱』体数据验证剖面峰谷相位与标准化，验证排名检验。

    CPU、无文件 IO、固定种子，覆盖退化输入（全空体积）。
    """
    rng = np.random.default_rng(0)

    # 1) 合成体数据：48 片，5 个『椎体』实心盘（中心 z=6+8k，半径 6，值 200），
    #    间隙片为低值背景 5（低于阈值=无组织）。柱应被提取，剖面峰=椎体中心。
    vol = np.full((32, 32, 48), 5.0, np.float32)
    centers = [6, 14, 22, 30, 38]
    yy, xx = np.mgrid[0:32, 0:32]
    disk = (yy - 16) ** 2 + (xx - 16) ** 2 <= 36
    for c in centers:
        vol[:, :, c - 2:c + 3] = 5.0
        vol[:, :, c - 2:c + 3][disk] = 200.0
    vol = vol + rng.normal(0, 1, vol.shape).astype(np.float32)  # 轻噪声

    r = column_profile(vol, threshold=100.0)
    assert r is not None and np.isfinite(r).all(), "剖面存在 NaN/Inf"
    assert r.shape == (48, 3)
    print("PASS 剖面形状 (S,3) 且有限")

    smoothed = r[:, 1]
    peak_z = int(np.argmax(smoothed))
    assert min(centers, key=lambda c: abs(c - peak_z)) == peak_z and peak_z in centers, \
        f"剖面主峰 {peak_z} 未落在椎体中心 {centers}"
    gaps = [c + 4 for c in centers[:-1]]
    assert all(smoothed[g] < smoothed[max(g - 2, 0)] for g in gaps), "间隙片未呈谷"
    print("PASS 峰=椎体中心、谷=间隙（周期峰谷相位正确）")

    zc = r[:, 2]  # zscore 对全 S 片计算，这里核对全通道
    assert abs(float(zc.mean())) < 1e-3 and abs(float(zc.std()) - 1.0) < 1e-3, "zscored 未标准化"
    print("PASS zscored 通道均值 0 标准差 1（eps 保护下有限）")

    empty = column_profile(np.zeros((32, 32, 48), np.float32), threshold=100.0)
    assert empty is None, "全空体积应返回 None 而非 NaN"
    print("PASS 退化输入（全空体积）返回 None")

    # 2) 排名检验：标注片在窗内最大 → rank0 命中；非最大 → 正排名
    prof = np.array([5.0, 10.0, 90.0, 20.0, 15.0, 30.0])
    rank, hit, ok = rank_stats(prof, 2, window=2)   # 值 90：窗 [10,90,20] 内最大
    assert ok and rank == 0 and hit, "窗内最大片的排名应为 0 且 argmax 命中"
    rank, hit, ok = rank_stats(prof, 0, window=2)   # 值 5：窗 [5,10,90] 内最大是 90
    assert ok and rank == 2 and not hit, "非最大片排名应为正且 argmax 不命中"
    rank, hit, ok = rank_stats(prof, 3, window=2)   # 值 20：窗 [10,90,20,15,30] 内排第 2
    assert ok and rank == 2 and not hit, "次大片排名应为 2"
    _, _, ok = rank_stats(prof, 99, window=2)
    assert not ok, "越界标注应 in_range=False"
    print("PASS 排名检验（最大片/非最大片/越界）")

    print("PASS 全部自测通过")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    sub.required = False

    sp = sub.add_parser("profile", help="柱内逐片封闭面积剖面（周期峰谷特征）")
    sp.add_argument("--images", required=True, help="nrrd 目录（<case>.nrrd）")
    sp.add_argument("--threshold", type=float, default=150.0,
                    help="组织阈值（与影像同单位；HU 影像常用 80-300）")
    sp.add_argument("--axis", type=int, default=2,
                    help="体数据中『厚度轴』的轴号（nrrd 读入缺省 2）")
    sp.add_argument("--column-frac", type=float, default=0.25,
                    help="柱判定：投影≥frac×切片数")
    sp.add_argument("--smooth-k", type=int, default=4)
    sp.add_argument("--out-dir", default="profile_feats")
    sp.set_defaults(func=cmd_profile)

    sp = sub.add_parser("rank-test", help="标注片在特征曲线邻域窗内的排名分布")
    sp.add_argument("--labels", required=True, help="labels JSON：{case:{target:z}}")
    sp.add_argument("--feats-dir", required=True, help="profile 输出目录")
    sp.add_argument("--window", type=int, default=5)
    sp.set_defaults(func=cmd_rank_test)

    sp = sub.add_parser("review-sheet", help="邻域切片审片拼条 PNG")
    sp.add_argument("--images", required=True)
    sp.add_argument("--case", required=True)
    sp.add_argument("--labels", default=None, help="labels JSON（提供时红框标出标注片）")
    sp.add_argument("--z0", type=int, default=None)
    sp.add_argument("--z1", type=int, default=None)
    sp.add_argument("--anchor", type=int, default=None, help="缺省范围 = 锚点±6")
    sp.add_argument("--axis", type=int, default=2)
    sp.add_argument("--out", default=None)
    sp.set_defaults(func=cmd_review_sheet)

    sp = sub.add_parser("selftest", help="运行内置自检（CPU，无文件 IO）")
    sp.set_defaults(func=lambda a: run_selftest())

    a = ap.parse_args()
    if a.cmd is None:
        run_selftest()
        return
    if a.cmd == "review-sheet" and a.z0 is None and a.anchor is None:
        sys.exit("[review-sheet] 需要 --z0/--z1 或 --anchor")
    if a.cmd == "review-sheet" and a.z0 is None:
        a.z0, a.z1 = a.anchor - 6, a.anchor + 6
    a.func(a)


if __name__ == "__main__":
    main()
