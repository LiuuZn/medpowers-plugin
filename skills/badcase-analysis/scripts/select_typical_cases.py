"""按五类错误遴选典型案例（badcase-analysis 技能第 1 步）。

本脚本是参考实现：所有路径由调用方（任务仓库）经命令行参数提供，不内嵌绝对路径。
数据集布局（--data-root 下 processed/raw-ref/cache/meta.json）与 mapping 格式
见 badcase-analysis SKILL.md。

输入：runs/error_analysis/preds_<split>.json + badcase_summary.json（相对 cwd）
+ --mapping nrrd_slice_mapping.json（标注结构检查）；输出 --out（默认
runs/error_analysis/typical_cases.json）：{类别: [案例条目]}，每类 3-5 例按严重度降序。

类别（与 SKILL.md 一致）：标注数值错误（多目标标在同一片 / 顺序颠倒或间距异常，
两小类）、疑似标注错误待人工复核、模型错误（锚点偏差超出精化范围 / 粗定位严重
偏离，两小类）、评测工具缺陷。

case 级 flag：
  same_slice  任一相邻对 <2mm（多目标标在同一片）；T12 与 L1 跨节段 <8mm 同样计入
  disorder    三目标齐全且顺序非 T12>T13>L1（本数据集 z 序号向头侧递增）
  gap_out     任一相邻对间距 ∉[4,30]mm
  thin        n_native>120（历史固定长度概率数组的影响面）
  sane        以上皆无
归类优先级 A>B>C>D>E>F（同一 case 多 flag 时不重复收录）。
"""
import argparse
import json
from pathlib import Path

import numpy as np

TARGETS = ("T12", "T13", "L1")
PER_CAT = (3, 5)

CAT_DEFS = {
    "A_collapse": "标注数值错误：多目标标在同一片",
    "B_gross_annot": "标注数值错误：顺序颠倒或间距异常",
    "C_annot_ambiguity": "疑似标注错误：外观不支持标注，待人工复核",
    "D_refine_range": "模型错误：锚点偏差超出精化范围",
    "E_stage1_fail": "模型错误：粗定位严重偏离",
    "F_eval_infra": "评测工具缺陷",
}


def case_flags(case, zmm_meta, mapping):
    tg = mapping.get(case, {})
    t12, t13, l1 = tg.get("T12"), tg.get("T13"), tg.get("L1")
    zmm = zmm_meta.get(case, {}).get("native_z_mm", 5.0)
    f = {"same_slice": False, "disorder": False, "gap_out": False, "thin": False}
    gaps = {}
    if None not in (t12, t13, l1) and not (t12 > t13 > l1):
        f["disorder"] = True
    for a, b in (("T12", "T13"), ("T13", "L1"), ("T12", "L1")):
        if tg.get(a) is not None and tg.get(b) is not None:
            g = abs(tg[a] - tg[b]) * zmm
            gaps[f"{a}-{b}"] = round(g, 1)
            # T12-L1 同片必不可能（中间隔整个运动节段）；相邻对阈值 2mm
            if g < (8 if (a, b) == ("T12", "L1") else 2):
                f["same_slice"] = True
            elif not (4 <= g <= 30) and (a, b) != ("T12", "L1"):
                f["gap_out"] = True
    if zmm_meta.get(case, {}).get("_n_native", 0) > 120:
        f["thin"] = True
    f["sane"] = not any(f.values())
    return f, {"T12": t12, "T13": t13, "L1": l1}, gaps


def main():
    ap = argparse.ArgumentParser(description="参考实现：路径由调用方（任务仓库）提供")
    ap.add_argument("--data-root", required=True,
                    help="数据集根目录（内含 processed/raw-ref/cache/meta.json）")
    ap.add_argument("--mapping", required=True,
                    help="nrrd_slice_mapping.json：{case: {target: z 索引}}（任务仓库提供）")
    ap.add_argument("--out", default="runs/error_analysis/typical_cases.json")
    args = ap.parse_args()

    mapping = json.load(open(args.mapping))
    zmm_meta = json.load(open(Path(args.data_root) / "processed/raw-ref/cache/meta.json"))
    # _n_native 换算（cache meta 存的是 4mm 箱）
    for c, m in zmm_meta.items():
        m["_n_native"] = int(round(m["n_valid"] * m["effective_bin_mm"] / m["native_z_mm"]))

    preds = {sp: json.load(open(f"runs/error_analysis/preds_{sp}.json"))
             for sp in ("val", "test")}
    summary = json.load(open("runs/error_analysis/badcase_summary.json"))
    bucket_of = {}
    for sp in ("val", "test"):
        for b in ("UNCOV", "ANNOT?", "MODEL"):
            for r in summary[sp]["detail"][b]:
                bucket_of[r["key"]] = (b, sp)

    # case 级聚合（splits 互斥；key "case__target" → pairs[target]）
    cases = {}
    for sp in ("val", "test"):
        for k, v in preds[sp].items():
            case, t = k.rsplit("__", 1)
            info = cases.setdefault(case, {"split": sp, "pairs": {}})
            info["pairs"][t] = v

    selected = {c: [] for c in CAT_DEFS}
    for case, info in cases.items():
        flags, annots, gaps = case_flags(case, zmm_meta, mapping)
        pairs = info["pairs"]
        covered = {t: v for t, v in pairs.items() if v["pred"] > -900}
        # 误差资格：存在未覆盖 pair（阶段1灾难）或任一 covered pair |err|>=2
        has_uncovered = len(covered) < len(pairs)
        if covered:
            worst_t = max(covered, key=lambda t: abs(covered[t]["pred"] - covered[t]["true"]))
            err_worst = covered[worst_t]["pred"] - covered[worst_t]["true"]
        else:
            worst_t, err_worst = next(iter(pairs)), None
        if not (has_uncovered or (err_worst is not None and abs(err_worst) >= 2)):
            continue
        bname, _ = bucket_of.get(f"{case}__{worst_t}", (None, None))
        cats = []
        if flags["same_slice"]:
            cats.append("A_collapse")
        elif flags["disorder"] or flags["gap_out"]:
            cats.append("B_gross_annot")
        if not cats:
            if flags["thin"] and not has_uncovered and err_worst is not None \
                    and abs(err_worst) >= 1:
                cats.append("F_eval_infra")
            elif has_uncovered:
                cats.append("E_stage1_fail")
            elif bname == "ANNOT?":
                cats.append("C_annot_ambiguity")
            elif bname == "MODEL":
                cats.append("D_refine_range")
        if not cats:
            continue
        cat = cats[0]
        # 严重度：未覆盖 > covered |err|；未覆盖内部按 |anchor_err| 排
        sev = 1e4 + max(abs(p["anchor"] - p["true"]) for p in pairs.values()) \
            if has_uncovered else abs(err_worst)
        selected[cat].append({
            "case": case, "split": info["split"], "err_worst": err_worst,
            "worst_target": worst_t, "has_uncovered": has_uncovered, "sev": sev,
            "flags": flags, "annots": annots, "gaps_mm": gaps,
            "bucket": bname,
            "pairs": {t: {"true": v["true"], "pred": v["pred"],
                          "anchor": v["anchor"], "in_win": v["in_win"],
                          "intrinsic": v["intrinsic"], "vote_std": v["vote_std"],
                          "native_z_mm": v["native_z_mm"]}
                      for t, v in pairs.items()},
        })

    # 每类取 3-5 例：按严重度排序、单 split 上限 3；若两 split 都有候选但
    # 选中只含一个 split，用另一 split 最严重的至多 2 例替换尾部；
    # 同理保证「未覆盖 / covered 错误」两种形态都有代表（都存在时）
    def pick(cands, k=PER_CAT[1], per_split=3):
        order = sorted(cands, key=lambda r: -r["sev"])
        out, cnt = [], {}
        for r in order:
            if len(out) >= k:
                break
            if cnt.get(r["split"], 0) >= per_split:
                continue
            out.append(r)
            cnt[r["split"]] = cnt.get(r["split"], 0) + 1

        def _diversify(out, keyfunc, fill_n=2):
            groups = {keyfunc(r) for r in cands}
            have = {keyfunc(r) for r in out}
            if len(groups) > 1 and len(have) == 1:
                other = (groups - have).pop()
                fill = [r for r in order if keyfunc(r) == other][:fill_n]
                out = out[:max(k - len(fill), PER_CAT[0])] + fill
            return out

        out = _diversify(out, lambda r: r["split"])
        out = _diversify(out, lambda r: "uncov" if r["has_uncovered"] else "cov")
        return out

    for cat in selected:
        selected[cat] = pick(selected[cat])
        print(f"{cat} ({CAT_DEFS[cat]}): {len(selected[cat])} 例 "
              f"{[(r['case'][:16], r['split'], r['err_worst'], 'uncov' if r['has_uncovered'] else '') for r in selected[cat]]}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"categories": CAT_DEFS, "selected": selected}, f,
                  ensure_ascii=False, indent=1)
    print("->", args.out)


if __name__ == "__main__":
    main()
