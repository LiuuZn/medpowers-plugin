"""组装 bad case 可视化包（badcase-analysis 技能第 3 步）。

本脚本是参考实现：所有路径由调用方（任务仓库）经命令行参数提供，不内嵌绝对路径。

输入：--typical typical_cases.json（第 1 步产物；对应 fig.png 须已由
scripts/render_badcases.py --keys 渲染到 runs/error_analysis/figs/）
+ --img-dir 原始影像目录（<case>.nrrd，任务仓库数据根提供）。
输出：包目录（original.nrrd + fig.png + annotation.json + annotations.csv）。
README.md 由模型按 SKILL.md 第 4 步模板撰写，不在本脚本职责内。
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

FIG_DIR = Path("runs/error_analysis/figs")


def main():
    ap = argparse.ArgumentParser(description="参考实现：路径由调用方（任务仓库）提供")
    ap.add_argument("--typical", default="runs/error_analysis/typical_cases.json")
    ap.add_argument("--img-dir", required=True,
                    help="原始影像目录（<case>.nrrd，任务仓库数据根提供）")
    ap.add_argument("--out", default="runs/badcase_pack")
    args = ap.parse_args()

    doc = json.load(open(args.typical))
    out = Path(args.out)
    rows = []
    n_files = 0
    for cat, cases in doc["selected"].items():
        for item in cases:
            case, split = item["case"], item["split"]
            cid = f"{case}__{split}"
            cdir = out / "cases" / cid
            cdir.mkdir(parents=True, exist_ok=True)

            src_nrrd = Path(args.img_dir) / f"{case}.nrrd"
            dst_nrrd = cdir / "original.nrrd"
            if not dst_nrrd.exists():
                shutil.copyfile(src_nrrd, dst_nrrd)
            n_files += 1

            src_fig = FIG_DIR / f"{split}__{case}.png"
            dst_fig = cdir / "fig.png"
            if not src_fig.exists():
                raise FileNotFoundError(f"缺少渲染图 {src_fig}，先跑 render_badcases.py --keys {cid}")
            shutil.copyfile(src_fig, dst_fig)

            item["nrrd_size_mb"] = round(dst_nrrd.stat().st_size / 1e6, 1)
            annot = {k: item[k] for k in ("case", "split", "annots", "gaps_mm",
                                          "flags", "bucket", "err_worst", "pairs")}
            with open(cdir / "annotation.json", "w", encoding="utf-8") as f:
                json.dump(annot, f, ensure_ascii=False, indent=1)

            for t, p in item["pairs"].items():
                err = p["pred"] - p["true"] if p["pred"] > -900 else None
                rows.append({
                    "case": case, "split": split, "category": cat,
                    "target": t,
                    "true_slice": p["true"],
                    "pred_slice": p["pred"] if p["pred"] > -900 else "uncovered",
                    "err_slices": err, "err_mm": None if err is None else round(err * p["native_z_mm"], 1),
                    "anchor": p["anchor"], "anchor_err": round(p["anchor"] - p["true"], 1),
                    "preds_anchor_at_truth": ";".join(str(i) for i in p["intrinsic"]),
                    "vote_std": p["vote_std"],
                    "annots_all": json.dumps(item["annots"], ensure_ascii=False),
                    "gaps_mm": json.dumps(item["gaps_mm"], ensure_ascii=False),
                    "flags": ",".join(k for k, v in item["flags"].items() if v),
                })

    with open(out / "annotations.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    total_mb = sum(r["nrrd_size_mb"] for cs in doc["selected"].values() for r in cs)
    print(f"pack -> {out}  cases={n_files}  nrrd_total={total_mb:.0f}MB  rows={len(rows)}")
    print("README.md 需按 SKILL.md 第 4 步模板由模型撰写")


if __name__ == "__main__":
    main()
