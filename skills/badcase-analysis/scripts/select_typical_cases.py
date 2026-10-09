"""按五类错误遴选典型案例，并输出全量数据问题清单（badcase-analysis 技能第 1 步）。

本脚本是参考实现：所有路径由调用方（任务仓库）经命令行参数提供，不内嵌绝对路径。
数据集布局（--data-root 下 processed/raw-ref/cache/meta.json）与 mapping 格式
见 badcase-analysis SKILL.md。

输入：runs/error_analysis/preds_<split>.json + badcase_summary.json（相对 cwd）
+ --mapping nrrd_slice_mapping.json（标注结构检查）；输出两个文件：
- --out（默认 runs/error_analysis/typical_cases.json）：{类别: [案例条目]}，
  每类 3-5 例按严重度降序；
- --data-out（默认 runs/error_analysis/data_problems.json）：数据/标注问题
  **全量**案例清单（不受每类 3-5 例限制），供组装脚本写入包内 data_problems/
  文件夹与 data_problems.md。范围 = 结构违反的全部案例（同片/乱序/间距越界，
  含未构成评测错误者）+ 全部「疑似标注错误」桶案例 + --extra-data 传入的
  任务仓库外部嫌疑清单（来自与标注生成链路独立的审计/复核工具）。模型错误与
  评测工具缺陷不属于
  数据问题，不进该清单。

--extra-data 格式（JSON）：{"categories": {<键>: <中文类别名>},
"cases": {<键>: ["<case>__<split>", ...]}}；键并入数据清单类别，案例信号
从预测文件构建；类别中文名缺省用键名原文。

类别（与 SKILL.md 一致）：标注数值错误（多目标标在同一片 / 顺序颠倒或间距异常，
两小类）、疑似标注错误待人工复核、模型错误（锚点偏差超出精化范围 / 粗定位严重
偏离，两小类）、评测工具缺陷。

case 级 flag：
  same_slice   任一相邻对 <2mm（多目标标在同一片）；T12 与 L1 跨节段 <8mm 同样计入
  disorder     三目标齐全且顺序非 T12>T13>L1（本数据集 z 序号向头侧递增）
  gap_out      任一相邻对间距 ∉[4,30]mm
  thin         n_native>120（历史固定长度概率数组的影响面）
  structure_ok 以上皆无
典型案例归类优先级 same_slice > order_gap > suspect > anchor_out_of_range >
stage1_gross > eval_tool（同一 case 多 flag 时不重复收录）；
全量数据清单归类优先级 same_slice > order_gap > suspect > 外部清单键，
次级归属写入条目的 also 字段。类别键用描述性英文、不带字母代号前缀：报告与
包内文档一律写中文名称，禁止出现 A/B/C 或 §2.X 式代号。
"""
import argparse
import json
from pathlib import Path

TARGETS = ("T12", "T13", "L1")
PER_CAT = (3, 5)

CAT_DEFS = {
    "annot_same_slice": "标注数值错误：多目标标在同一片",
    "annot_order_gap": "标注数值错误：顺序颠倒或间距异常",
    "annot_suspect": "疑似标注错误：外观不支持标注，待人工复核",
    "model_anchor_out_of_range": "模型错误：锚点偏差超出精化范围",
    "model_stage1_gross": "模型错误：粗定位严重偏离",
    "eval_tool_defect": "评测工具缺陷",
}
DATA_CATS = ("annot_same_slice", "annot_order_gap", "annot_suspect")


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
    f["structure_ok"] = not any(f.values())
    return f, {"T12": t12, "T13": t13, "L1": l1}, gaps


def main():
    ap = argparse.ArgumentParser(description="参考实现：路径由调用方（任务仓库）提供")
    ap.add_argument("--data-root", required=True,
                    help="数据集根目录（内含 processed/raw-ref/cache/meta.json）")
    ap.add_argument("--mapping", required=True,
                    help="nrrd_slice_mapping.json：{case: {target: z 索引}}（任务仓库提供）")
    ap.add_argument("--out", default="runs/error_analysis/typical_cases.json")
    ap.add_argument("--data-out", default="runs/error_analysis/data_problems.json",
                    help="数据/标注问题全量清单输出路径")
    ap.add_argument("--extra-data", default=None,
                    help='任务仓库外部嫌疑清单 JSON：{"categories": {键: 中文名}, '
                         '"cases": {键: ["case__split", ...]}}')
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

    def item_of(case):
        """构建统一条目：信号 + 标注结构；err_worst 仅在存在覆盖错误时有意义。"""
        info = cases[case]
        flags, annotations, gaps = case_flags(case, zmm_meta, mapping)
        pairs = info["pairs"]
        covered = {t: v for t, v in pairs.items() if v["pred"] > -900}
        has_uncovered = len(covered) < len(pairs)
        if covered:
            worst_t = max(covered, key=lambda t: abs(covered[t]["pred"] - covered[t]["true"]))
            err_worst = covered[worst_t]["pred"] - covered[worst_t]["true"]
        else:
            worst_t, err_worst = next(iter(pairs)), None
        bname, _ = bucket_of.get(f"{case}__{worst_t}", (None, None))
        sev = 1e4 + max(abs(p["anchor"] - p["true"]) for p in pairs.values()) \
            if has_uncovered else (abs(err_worst) if err_worst is not None else 0)
        return {
            "case": case, "split": info["split"], "err_worst": err_worst,
            "worst_target": worst_t, "has_uncovered": has_uncovered, "sev": sev,
            "flags": flags, "annotations": annotations, "gaps_mm": gaps,
            "bucket": bname,
            "pairs": {t: {"true": v["true"], "pred": v["pred"],
                          "anchor": v["anchor"], "in_win": v["in_win"],
                          "intrinsic": v["intrinsic"], "vote_std": v["vote_std"],
                          "native_z_mm": v["native_z_mm"]}
                      for t, v in pairs.items()},
        }

    # ---- 典型案例遴选（每类 3-5 例，保持原逻辑）----
    selected = {c: [] for c in CAT_DEFS}
    for case, info in cases.items():
        it = item_of(case)
        flags, pairs = it["flags"], it["pairs"]
        covered = {t: v for t, v in pairs.items() if v["pred"] > -900}
        has_uncovered = it["has_uncovered"]
        err_worst = it["err_worst"]
        if not (has_uncovered or (err_worst is not None and abs(err_worst) >= 2)):
            continue
        bname = it["bucket"]
        cats = []
        if flags["same_slice"]:
            cats.append("annot_same_slice")
        elif flags["disorder"] or flags["gap_out"]:
            cats.append("annot_order_gap")
        if not cats:
            if flags["thin"] and not has_uncovered and err_worst is not None \
                    and abs(err_worst) >= 1:
                cats.append("eval_tool_defect")
            elif has_uncovered:
                cats.append("model_stage1_gross")
            elif bname == "ANNOT?":
                cats.append("annot_suspect")
            elif bname == "MODEL":
                cats.append("model_anchor_out_of_range")
        if not cats:
            continue
        selected[cats[0]].append(it)

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

    # ---- 数据/标注问题全量清单（结构违反全量 + 疑似标注全量 + 外部清单）----
    typical_ids = {r["case"] for items in selected.values() for r in items}
    data_categories = {k: v for k, v in CAT_DEFS.items() if k in DATA_CATS}
    data_lists = {k: {} for k in DATA_CATS}  # cat -> {case: item}

    for case in cases:
        it = item_of(case)
        flags, bname = it["flags"], it["bucket"]
        assign, also = None, []
        if flags["same_slice"]:
            assign = "annot_same_slice"
        elif flags["disorder"] or flags["gap_out"]:
            assign = "annot_order_gap"
        if assign and bname == "ANNOT?":
            also.append("annot_suspect")
        elif assign is None and bname == "ANNOT?":
            assign = "annot_suspect"
        if assign:
            it["typical"] = case in typical_ids
            it["also"] = also
            data_lists[assign][case] = it

    if args.extra_data:
        extra = json.load(open(args.extra_data))
        data_categories.update(extra.get("categories", {}))
        for key, ids in extra.get("cases", {}).items():
            data_lists.setdefault(key, {})
            for cid in ids:
                case, split = cid.rsplit("__", 1)
                if case not in cases:
                    print(f"[extra-data] 警告：{cid} 不在预测文件中，跳过")
                    continue
                it = item_of(case)
                it["typical"] = case in typical_ids
                it["also"] = []
                data_lists[key][case] = it

    data_selected = {}
    n_planes = {}
    for cat, items in data_lists.items():
        ordered = sorted(items.values(), key=lambda r: -r["sev"])
        data_selected[cat] = ordered
        n_planes[cat] = sum(len(r["pairs"]) for r in ordered)
        print(f"[data] {cat} ({data_categories.get(cat, cat)}): {len(ordered)} 例 "
              f"{n_planes[cat]} 平面")

    with open(args.data_out, "w", encoding="utf-8") as f:
        json.dump({"categories": data_categories, "selected": data_selected}, f,
                  ensure_ascii=False, indent=1)
    print("->", args.data_out)


if __name__ == "__main__":
    main()
