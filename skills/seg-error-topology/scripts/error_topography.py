"""分割误差空间分布画像（error topography）：FP/FN 沿各轴相对坐标的分布定位。

回答调优中的问题：过分割（FP）与欠分割（FN）发生在目标结构的**哪个部位**——
沿每个物理轴（如 z 头-足、y 腹-背、x 左-右），误差质量集中在 GT（标注）的
中心带还是端部、偏向哪一端、端部之外外溢多少。任务无关：只吃对齐的
pred/gt 二值掩码对 + 体素间距（mm）+ 任务仓库给定的轴语义标签。

口径定义（逐例）：
  相对坐标 r：GT 沿轴 a 的体素范围 [v_min, v_max]，中心 c=(v_min+v_max)/2、
  半长 h=(v_max-v_min)/2（h<0.5 体素时按 0.5 护底），r=(v-c)/h。GT 内部
  |r|<=1；FP 可越界到 |r|>1（沿该轴溢出 GT 实际范围）。
  分箱：[-inf,-1), [-1,-0.75), [-0.75,-0.5), [-0.5,-0.25), [-0.25,0),
        [0,0.25), [0.25,0.5), [0.5,0.75), [0.75,1), [1,+inf)  —— 10 箱。
  每箱计 FN / FP / TP / GT 体素质量（ml）；FN 只可能落在内侧 8 箱。
  局部漏检率 = FN / GT（逐箱）：该部位标注体素被漏掉的比例，直接可比
  全局漏检率（总体 FN/GT）——高者为误差富集部位。

聚合纪律（两套口径并报，结论以逐例平均为主）：
  逐例平均（primary）：每例 FN/FP 箱质量先归一为该例误差的份额，再跨例
  取均值——避免一两个大误差病例淹没群体规律；参与该误差类型平均的例
  须满足最小误差体量门槛（--min-err-ml，默认 0.05ml），排除数如实报告；
  汇总体（pooled）：全部误差体素直接累加——反映误差总体质量落点。
  μ_r（FP/FN 质量加权平均相对坐标，逐例）：跨例分布（中位/四分位），
  正负号直接给出偏向（须配任务方向语义解读，如 z 轴 μ_r>0 = 偏头侧）。
  端部外溢：r>1 / r<-1 的 FP 质量份额（沿轴两端）。

纪律与红线：
  - 本脚本只做掩码的程序化测量，不读影像、不渲染影像、不做视觉判读；
  - 轴的方向语义（哪端是头侧/左侧等）由任务仓库根据影像几何（方向矩阵
    符号）程序化核验后提供，本脚本只按数组轴序输出中性标签 + 调用方标签；
  - 图内文字一律 ASCII（matplotlib 缺 CJK 字形），中文结论写进 JSON/报告。

输出（--out 目录）：
  per_case.jsonl   逐例全量信号（含每轴箱质量 ml 与份额、μ_r、mm 偏移、dice）
  summary.json     群体聚合（两口径箱画像、局部漏检率、μ_r 分布、外溢、相关性）
  profiles.png     每轴 FN/FP 份额 + GT 质量基线 + 局部漏检率
  overview.png     μ_r 与 dice 的关系、FN/FP 体量与 dice 的关系
  manifest_copy.json  输入清单回写（可追溯）

CLI（配对清单模式，掩码文件路径均可相对清单文件）：
  python error_topography.py --pairs pairs.json --out OUT \
      [--axis-names z,y,x] \
      [--axis-labels "z:+S/-I;y:+P/-A;x:+L/-R"] \
      [--min-err-ml 0.05] [--groups val,test]
pairs.json 条目：{"case": str, "gt": path|null, "pred": path,
                 "spacing": [mm]（头缺省时必须给，数组轴序）,
                 "gt_label": int?, "pred_label": int?, "group": str?}
支持格式：.nii/.nii.gz（nibabel）、.nrrd（pynrrd，index_order=F）、
.npz（键 gt/pred，可选 spacing）。gt 与 pred 必须同形同网格（对齐由调用方负责）。

函数级 API（任务仓库包装脚本 import 用）：
  compute_case(gt, pred, spacing, case, axis_names) -> dict
  aggregate(rows, axis_names, min_err_ml) -> dict
  plot_profiles(summary, rows, out, axis_labels_ascii) -> None
用法示例：分析当前最优 checkpoint 的 val 预测 / 离线评估集自带的模型预测。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.stats import spearmanr

BIN_EDGES = (-np.inf, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, np.inf)
BIN_LABELS = ("lt-1", "-1..-0.75", "-0.75..-0.5", "-0.5..-0.25", "-0.25..0",
              "0..0.25", "0.25..0.5", "0.5..0.75", "0.75..1", "gt1")
N_BINS = len(BIN_LABELS)
INNER = range(1, 9)  # GT 范围内的 8 箱（FN 只落这里）


def _rel_coords(idx: np.ndarray, vmin: np.ndarray, vmax: np.ndarray) -> np.ndarray:
    """体素索引 → 逐轴相对坐标 r（GT 范围归一，半长按 0.5 体素护底）。"""
    c = (vmin + vmax) / 2.0
    h = np.maximum((vmax - vmin) / 2.0, 0.5)
    return (idx - c) / h


def _bin_mass(idx_sel: np.ndarray, vmin, vmax, ml_per_vox: float) -> np.ndarray:
    """选中体素索引 → 每轴每箱质量 ml。"""
    out = np.zeros((3, N_BINS))
    if idx_sel.shape[0] == 0:
        return out
    for a in range(3):
        r = _rel_coords(idx_sel[:, a], vmin[a], vmax[a])
        b = np.digitize(r, BIN_EDGES[1:-1])  # 0..9 箱
        # 边界口径：r=±1.0（GT 沿轴最端体素的精确值）属于内侧端箱而非外溢箱；
        # 外溢箱只收严格越界（|r|>1）的体素
        b = np.where((r <= 1.0) & (b == 9), 8, b)
        b = np.where((r >= -1.0) & (b == 0), 1, b)
        cnt = np.bincount(b, minlength=N_BINS)
        out[a] = cnt * ml_per_vox
    return out


def compute_case(gt: np.ndarray, pred: np.ndarray, spacing,
                 case: str, axis_names=("axis0", "axis1", "axis2")) -> dict:
    """单例误差空间画像。gt/pred 同形二值（bool 或 0/1）；spacing 数组轴序 mm。

    红线提醒：调用方必须保证 gt/pred 在同一网格上对齐（含轴序、方向、裁剪）。
    """
    gt = np.asarray(gt) > 0
    pred = np.asarray(pred) > 0
    assert gt.shape == pred.shape, f"{case}: gt/pred 形状不一致 {gt.shape} vs {pred.shape}"
    sp = np.asarray(spacing, float)
    assert sp.shape == (3,), f"{case}: spacing 须为 3 元（数组轴序 mm）"
    ml = float(np.prod(sp)) / 1000.0

    inter = gt & pred
    denom = gt.sum() + pred.sum()
    dice = 2.0 * inter.sum() / denom if denom else float("nan")

    idx_g = np.argwhere(gt)
    assert idx_g.shape[0] > 0, f"{case}: GT 为空，无法定义相对坐标（跳过由调用方处理）"
    vmin, vmax = idx_g.min(0), idx_g.max(0)

    fn = gt & ~pred
    fp = pred & ~gt
    idx_all = np.argwhere(gt | pred)
    sel_g = gt[tuple(idx_all.T)]
    sel_p = pred[tuple(idx_all.T)]
    m_gt = _bin_mass(idx_all[sel_g], vmin, vmax, ml)
    m_fn = _bin_mass(idx_all[sel_g & ~sel_p], vmin, vmax, ml)
    m_fp = _bin_mass(idx_all[~sel_g & sel_p], vmin, vmax, ml)
    m_tp = m_gt - m_fn

    gt_ml = float(gt.sum() * ml)
    pred_ml = float(pred.sum() * ml)
    fn_ml = float(fn.sum() * ml)
    fp_ml = float(fp.sum() * ml)

    per_axis = {}
    for a, name in enumerate(axis_names):
        tot_fn, tot_fp = m_fn[a].sum(), m_fp[a].sum()
        # μ_r：FP/FN 质量加权平均相对坐标（体素等权=质量等权）
        mu_fn = float(_rel_coords(np.argwhere(fn)[:, a], vmin[a], vmax[a]).mean()) if fn.any() else None
        mu_fp = float(_rel_coords(np.argwhere(fp)[:, a], vmin[a], vmax[a]).mean()) if fp.any() else None
        per_axis[name] = {
            "gt_extent_mm": round(float(2 * ((vmax[a] - vmin[a]) / 2.0) * sp[a]), 1),
            "fn_bin_ml": [round(x, 4) for x in m_fn[a]],
            "fp_bin_ml": [round(x, 4) for x in m_fp[a]],
            "gt_bin_ml": [round(x, 4) for x in m_gt[a]],
            "fn_share": [round(x / tot_fn, 4) if tot_fn > 0 else None for x in m_fn[a]],
            "fp_share": [round(x / tot_fp, 4) if tot_fp > 0 else None for x in m_fp[a]],
            "local_miss_rate": [round(float(m_fn[a][i] / m_gt[a][i]), 4)
                                if m_gt[a][i] > 0 else None for i in INNER],
            "mu_r_fn": round(mu_fn, 3) if mu_fn is not None else None,
            "mu_r_fp": round(mu_fp, 3) if mu_fp is not None else None,
            "fp_beyond_pos_share": round(float(m_fp[a][9] / tot_fp), 4) if tot_fp > 0 else None,
            "fp_beyond_neg_share": round(float(m_fp[a][0] / tot_fp), 4) if tot_fp > 0 else None,
        }

    # mm 口径：FP/FN 质心相对 GT 质心的逐轴偏移
    com_g = np.array(ndimage.center_of_mass(gt))
    row = {
        "case": case, "dice": round(dice, 4) if np.isfinite(dice) else None,
        "gt_ml": round(gt_ml, 3), "pred_ml": round(pred_ml, 3),
        "fn_ml": round(fn_ml, 3), "fp_ml": round(fp_ml, 3),
        "vol_ratio": round(pred_ml / gt_ml, 3) if gt_ml else None,
        "fn_rate": round(fn_ml / gt_ml, 4) if gt_ml else None,
        "axes": per_axis,
        "com_shift_mm": None,  # 下方填充
    }
    if fp.any():
        com_fp = np.array(ndimage.center_of_mass(fp))
        row["com_fp_shift_mm"] = [round(float(x), 2) for x in (com_fp - com_g) * sp]
    else:
        row["com_fp_shift_mm"] = None
    if fn.any():
        com_fn = np.array(ndimage.center_of_mass(fn))
        row["com_fn_shift_mm"] = [round(float(x), 2) for x in (com_fn - com_g) * sp]
    else:
        row["com_fn_shift_mm"] = None
    return row


def aggregate(rows: list[dict], axis_names, min_err_ml: float = 0.05) -> dict:
    """群体聚合：两口径箱画像 + 局部漏检率 + μ_r 分布 + 外溢 + 相关性。"""
    dice = np.array([r["dice"] if r["dice"] is not None else np.nan for r in rows], float)
    ok = np.isfinite(dice)
    out = {
        "n_cases": len(rows),
        "dice": {"mean": round(float(np.nanmean(dice)), 4),
                 "median": round(float(np.nanmedian(dice)), 4),
                 "p10": round(float(np.nanpercentile(dice, 10)), 3),
                 "p90": round(float(np.nanpercentile(dice, 90)), 3)},
        "err_ml_total": round(float(sum(r["fn_ml"] + r["fp_ml"] for r in rows)), 1),
        "fn_ml_share": None,
        "min_err_ml": min_err_ml,
        "axes": {},
    }
    tot_fn = sum(r["fn_ml"] for r in rows)
    tot_fp = sum(r["fp_ml"] for r in rows)
    out["fn_ml_share"] = round(tot_fn / (tot_fn + tot_fp), 3) if (tot_fn + tot_fp) else None

    def _corr(v):
        v = np.array(v, float)
        m = np.isfinite(v) & ok & np.isfinite(dice)
        if m.sum() < 5 or np.std(v[m]) == 0:
            return None
        rho, p = spearmanr(dice[m], v[m])
        return {"rho": round(float(rho), 3), "p": round(float(p), 4)}

    out["corr_dice_vs"] = {
        "fn_ml": _corr([r["fn_ml"] for r in rows]),
        "fp_ml": _corr([r["fp_ml"] for r in rows]),
        "vol_ratio": _corr([r["vol_ratio"] for r in rows]),
    }

    for name in axis_names:
        fn_s = np.array([[x if x is not None else np.nan
                          for x in r["axes"][name]["fn_share"]] for r in rows])
        fp_s = np.array([[x if x is not None else np.nan
                          for x in r["axes"][name]["fp_share"]] for r in rows])
        fn_ml = np.array([r["fn_ml"] for r in rows])
        fp_ml = np.array([r["fp_ml"] for r in rows])
        gt_b = np.array([r["axes"][name]["gt_bin_ml"] for r in rows])
        fn_b = np.array([r["axes"][name]["fn_bin_ml"] for r in rows])
        fp_b = np.array([r["axes"][name]["fp_bin_ml"] for r in rows])
        tp_b = gt_b - fn_b

        # 逐例平均（≥min_err_ml 才参与对应误差类型）
        m_fn = fn_ml >= min_err_ml
        m_fp = fp_ml >= min_err_ml
        avg_fn = np.nanmean(fn_s[m_fn], axis=0) if m_fn.any() else np.full(N_BINS, np.nan)
        avg_fp = np.nanmean(fp_s[m_fp], axis=0) if m_fp.any() else np.full(N_BINS, np.nan)
        # 汇总体（全误差体素直接累加）
        pool_fn = fn_b.sum(0) / fn_b.sum() if fn_b.sum() > 0 else np.full(N_BINS, np.nan)
        pool_fp = fp_b.sum(0) / fp_b.sum() if fp_b.sum() > 0 else np.full(N_BINS, np.nan)
        # 局部漏检率（汇总体：逐箱 ΣFN/ΣGT，仅内侧 8 箱）
        with np.errstate(invalid="ignore", divide="ignore"):
            lmr = fn_b[:, 1:9].sum(0) / np.maximum(gt_b[:, 1:9].sum(0), 1e-9)
        gt_share = gt_b.sum(0)
        gt_share = gt_share / gt_share.sum() if gt_share.sum() > 0 else gt_share

        mu_fn = np.array([r["axes"][name]["mu_r_fn"] for r in rows], float)
        mu_fp = np.array([r["axes"][name]["mu_r_fp"] for r in rows], float)

        def _q(v):
            m = np.isfinite(v)
            if m.sum() == 0:
                return None
            return {"median": round(float(np.median(v[m])), 3),
                    "p25": round(float(np.percentile(v[m], 25)), 3),
                    "p75": round(float(np.percentile(v[m], 75)), 3),
                    "frac_pos": round(float((v[m] > 0).mean()), 3),
                    "n": int(m.sum())}

        out["axes"][name] = {
            "fn_share_case_avg": [round(float(x), 3) for x in avg_fn],
            "fp_share_case_avg": [round(float(x), 3) for x in avg_fp],
            "fn_share_pooled": [round(float(x), 3) for x in pool_fn],
            "fp_share_pooled": [round(float(x), 3) for x in pool_fp],
            "gt_share_pooled": [round(float(x), 3) for x in gt_share],
            "local_miss_rate_pooled_inner8": [round(float(x), 3) for x in lmr],
            "global_miss_rate": round(float(fn_b.sum() / max(gt_b.sum(), 1e-9)), 3),
            "mu_r_fn": _q(mu_fn), "mu_r_fp": _q(mu_fp),
            "fp_beyond_pos_pooled_ml": round(float(fp_b[:, 9].sum()), 2),
            "fp_beyond_neg_pooled_ml": round(float(fp_b[:, 0].sum()), 2),
            "n_cases_fn_included": int(m_fn.sum()), "n_cases_fp_included": int(m_fp.sum()),
            "corr_dice_vs_mu_r_fn": _corr(mu_fn), "corr_dice_vs_mu_r_fp": _corr(mu_fp),
        }
    out["bin_labels"] = list(BIN_LABELS)
    return out


def plot_profiles(summary: dict, rows: list[dict], out: Path,
                  axis_labels_ascii: dict[str, str]) -> None:
    x = np.arange(N_BINS)
    labels = list(BIN_LABELS)
    names = list(summary["axes"].keys())
    fig, axs = plt.subplots(len(names), 1, figsize=(10, 3.2 * len(names)), squeeze=False)
    for ax, name in zip(axs[:, 0], names):
        s = summary["axes"][name]
        w = 0.38
        ax.bar(x - w / 2, np.nan_to_num(s["fn_share_case_avg"]), w,
               label="FN share (case-avg)", color="tab:red")
        ax.bar(x + w / 2, np.nan_to_num(s["fp_share_case_avg"]), w,
               label="FP share (case-avg)", color="tab:blue")
        ax.plot(x, np.nan_to_num(s["gt_share_pooled"]), "k--o", ms=3, lw=1,
                label="GT mass share (pooled)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, fontsize=8)
        ax.set_ylabel("share of error mass")
        ttl = axis_labels_ascii.get(name, name)
        mu_f = s["mu_r_fn"]; mu_p = s["mu_r_fp"]
        ax.set_title(f"{name} ({ttl})  mu_r FN={mu_f['median'] if mu_f else None} "
                     f"FP={mu_p['median'] if mu_p else None}")
        ax.legend(fontsize=8, loc="upper left")
        ax2 = ax.twinx()
        lmr = s["local_miss_rate_pooled_inner8"]
        ax2.plot(list(INNER), np.nan_to_num(lmr), color="tab:green", marker="s", ms=4)
        ax2.axhline(s["global_miss_rate"], color="tab:green", ls=":", lw=1)
        ax2.set_ylabel("local miss rate (green)", color="tab:green")
        ax2.tick_params(axis="y", labelcolor="tab:green")
    fig.tight_layout()
    fig.savefig(out / "profiles.png", dpi=150)
    plt.close(fig)

    dice = np.array([r["dice"] if r["dice"] is not None else np.nan for r in rows], float)
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for ax, name in zip(axs, names):
        mu_fp = np.array([r["axes"][name]["mu_r_fp"] for r in rows], float)
        mu_fn = np.array([r["axes"][name]["mu_r_fn"] for r in rows], float)
        ax.scatter(mu_fp, dice, s=24, c="tab:blue", label="mu_r FP")
        ax.scatter(mu_fn, dice, s=24, c="tab:red", marker="x", label="mu_r FN")
        ax.axvline(0, color="gray", lw=0.8)
        ax.set_xlabel(f"mu_r on {name} ({axis_labels_ascii.get(name, name)})")
        ax.set_ylabel("dice")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "overview.png", dpi=150)
    plt.close(fig)


# ---------------- CLI：配对清单模式 ----------------

def _load_mask(path: Path, label: int | None):
    if path.suffix == ".npz":
        arr = np.load(path)["mask"]
    elif path.suffix in (".nii", ".gz"):
        import nibabel as nib
        arr = np.asanyarray(nib.load(path).dataobj)
    elif path.suffix == ".nrrd":
        import nrrd
        arr, _ = nrrd.read(str(path), index_order="F")
    else:
        raise ValueError(f"不支持的掩码格式：{path}")
    return (arr == label) if label is not None else (arr > 0)


def _mask_spacing(path: Path):
    """从头文件读 spacing（数组轴序 mm）；读不出/有歧义返回 None，由清单显式给。

    nrrd：pynrrd index_order=F 的 space directions 即数组轴序，可靠；
    nii：nibabel 读出 (x,y,z) 轴序与调用方数组轴序的关系不自明（任务仓库
    常做 transpose），不做猜测，一律要求清单显式给 spacing。
    """
    try:
        if path.suffix == ".nrrd":
            import nrrd
            _, h = nrrd.read(str(path), index_order="F")
            sd = np.asarray(h["space directions"], float)
            if sd.shape == (3, 3):
                return [abs(float(v)) for v in np.diag(sd)]
            return [float(np.linalg.norm(v)) for v in sd]
    except Exception:
        return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", required=True, type=Path, help="配对清单 JSON")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--axis-names", default="axis0,axis1,axis2",
                    help="三个数组轴的名称（逗号分隔），如 z,y,x")
    ap.add_argument("--axis-labels", default="",
                    help='ASCII 方向语义，如 "z:+S/-I;y:+P/-A;x:+L/-R"（仅用于图题）')
    ap.add_argument("--min-err-ml", type=float, default=0.05,
                    help="参与逐例平均的最小误差体量门槛（ml），FN/FP 分别判定")
    ap.add_argument("--groups", default="", help="逗号分隔，仅分析清单中 group 在列的条目")
    args = ap.parse_args()

    axis_names = [s.strip() for s in args.axis_names.split(",") if s.strip()]
    assert len(axis_names) == 3
    labels_ascii = {}
    for kv in args.axis_labels.split(";"):
        if ":" in kv:
            k, v = kv.split(":", 1)
            labels_ascii[k.strip()] = v.strip()

    pairs = json.loads(args.pairs.read_text())
    if isinstance(pairs, dict):
        pairs = pairs.get("pairs", pairs.get("cases"))
    groups = {g.strip() for g in args.groups.split(",") if g.strip()}
    if groups:
        pairs = [p for p in pairs if p.get("group") in groups]

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    per_case_path = args.out / "per_case.jsonl"
    with per_case_path.open("w") as cf:
        for p in pairs:
            case = p["case"]
            gt_p = Path(p["gt"]) if p.get("gt") else None
            pred_p = Path(p["pred"])
            base = args.pairs.parent
            gt = _load_mask((base / gt_p) if not gt_p.is_absolute() else gt_p,
                            p.get("gt_label")) if gt_p else None
            pred = _load_mask((base / pred_p) if not pred_p.is_absolute() else pred_p,
                              p.get("pred_label"))
            sp = p.get("spacing")
            if sp is None:
                sp = _mask_spacing((base / pred_p) if not pred_p.is_absolute() else pred_p)
                if sp is None and gt_p is not None:
                    sp = _mask_spacing((base / gt_p) if not gt_p.is_absolute() else gt_p)
            assert sp is not None, f"{case}: 缺 spacing（清单未给且头文件读不出）"
            if gt is None:
                raise SystemExit(f"{case}: 无 gt 掩码，无法画像（gt=null 仅预留）")
            r = compute_case(gt, pred, sp, case, axis_names)
            if p.get("group"):
                r["group"] = p["group"]
            rows.append(r)
            cf.write(json.dumps(r, ensure_ascii=False) + "\n")
            cf.flush()
            print(f"[{len(rows)}/{len(pairs)}] {case} dice={r['dice']} "
                  f"FN={r['fn_ml']}ml FP={r['fp_ml']}ml", flush=True)

    summary = aggregate(rows, axis_names, args.min_err_ml)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    plot_profiles(summary, rows, args.out, labels_ascii)
    (args.out / "manifest_copy.json").write_text(json.dumps(pairs, indent=1, ensure_ascii=False))
    print(json.dumps(summary["dice"], ensure_ascii=False))


if __name__ == "__main__":
    main()
