"""组装 bad case 可视化包（badcase-analysis 技能第 3 步）。

本脚本是参考实现：所有路径由调用方（任务仓库）经命令行参数提供，不内嵌绝对路径。

输入：
- --typical typical_cases.json（第 1 步产物；对应渲染图须已由
  scripts/render_badcases.py --keys 生成到 --fig-dir，命名 <split>__<case>.png）
- --img-dir 原始影像目录（<case>.nrrd，任务仓库数据根提供）
- --data data_problems.json（第 1 步 --data-out 产物，数据/标注问题全量清单；
  渲染范围须包含其中全部案例）。
输出：包目录，包名 = `bad_case_analysis_` 前缀 + 生成时刻，精确到分钟
（bad_case_analysis_<YYYYMMDDHHMM>，如 bad_case_analysis_202610091311，目录与
同名 tar.gz 构成交付物）；未显式传 --out 时自动取 runs/bad_case_analysis_<时间戳>。
tar.gz 只是默认压缩格式：调用方所在任务仓库对交付包格式/命名另有规定时，
以仓库规定为准（本脚本只组装包目录，不执行压缩，压缩命令由调用方按适用规定执行）。
- cases/：模型/评测/参照等典型案例，全部扁平存放，文件以案例 ID
  （<case>__<split>）开头命名：<案例id>.nrrd + <案例id>_fig.png +
  <案例id>_annotation.json；
- data_problems/：数据/标注问题案例的**全量**数据（同一三件套，扁平）；
  同时属于典型案例的案例只存此处一份，不重复存放；
- 包根：annotations.csv 总表（group 列标记 typical / data / typical+data）
  与 data_problems.md 骨架（全量信号表 + 文件清单 +「模型撰写」占位）。
组装成功后自动清理同目录下的旧分析包（12 位时间戳名与历史 badcase_pack* 名的
包目录及 tar.gz；--keep-old 可保留），保证同刻只保留现行一套。
两份文档的正文（主报告 badcase_analysis_report.md 与 data_problems.md 的成因
讨论/逐例解读/仲裁结论）由模型按 SKILL.md 第 4 步模板撰写。
"""
import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

FLAG_CN = {"same_slice": "同片", "disorder": "乱序", "gap_out": "间距越界",
           "thin": ">120片"}  # 参考实现的结构 flag 中文对照；未知 flag 回退键名


def _is_prefixed_ts(name: str) -> bool:
    """bad_case_analysis_<YYYYMMDDHHMM> 形态（前缀 + 12 位数字时间戳）。"""
    return (name.startswith("bad_case_analysis_")
            and name[len("bad_case_analysis_"):].isdigit()
            and len(name) == len("bad_case_analysis_") + 12)


def clean_old_packs(out: Path):
    """删除同目录下的旧分析包，保留当前包名（目录与同名 tar.gz）。

    清理范围（历史三代命名都在内）：现行 bad_case_analysis_<12位时间戳> 名的
    包目录及 .tar.gz、过渡期裸 12 位时间戳名（YYYYMMDDHHMM）、以及更早的
    badcase_pack* 名（如 badcase_pack、badcase_pack_v2）的包目录及 .tar.gz。
    返回被删除路径列表。
    """
    removed = []
    if not out.parent.exists():
        return removed
    keep = {out.name, out.name + ".tar.gz"}
    for p in sorted(out.parent.iterdir()):
        if p.name in keep:
            continue
        stem = p.name[:-7] if p.name.endswith(".tar.gz") else None
        is_ts_dir = p.is_dir() and p.name.isdigit() and len(p.name) == 12
        is_ts_tgz = p.is_file() and stem is not None and stem.isdigit() and len(stem) == 12
        is_bca_dir = p.is_dir() and _is_prefixed_ts(p.name)
        is_bca_tgz = p.is_file() and stem is not None and _is_prefixed_ts(stem)
        is_legacy = p.name.startswith("badcase_pack") and (
            p.is_dir() or (p.is_file() and p.name.endswith(".tar.gz")))
        if is_ts_dir or is_ts_tgz or is_bca_dir or is_bca_tgz or is_legacy:
            shutil.rmtree(p) if p.is_dir() else p.unlink()
            removed.append(p)
    return removed


def place_case(item, cat, group, img_dir, fig_dir, cases_dir):
    """复制单案例三件套到 cases_dir（cases/ 或 data_problems/），返回 CSV 行组。"""
    case, split = item["case"], item["split"]
    cid = f"{case}__{split}"

    src_nrrd = Path(img_dir) / f"{case}.nrrd"
    dst_nrrd = cases_dir / f"{cid}.nrrd"
    if not dst_nrrd.exists():
        shutil.copyfile(src_nrrd, dst_nrrd)

    src_fig = Path(fig_dir) / f"{split}__{case}.png"
    dst_fig = cases_dir / f"{cid}_fig.png"
    if not src_fig.exists():
        raise FileNotFoundError(f"缺少渲染图 {src_fig}，先跑 render_badcases.py --keys {cid}")
    shutil.copyfile(src_fig, dst_fig)

    item["nrrd_size_mb"] = round(dst_nrrd.stat().st_size / 1e6, 1)
    annot = {k: item[k] for k in ("case", "split", "annotations", "gaps_mm",
                                  "flags", "bucket", "err_worst", "pairs")}
    with open(cases_dir / f"{cid}_annotation.json", "w", encoding="utf-8") as f:
        json.dump(annot, f, ensure_ascii=False, indent=1)

    rows = []
    for t, p in item["pairs"].items():
        err = p["pred"] - p["true"] if p["pred"] > -900 else None
        rows.append({
            "case": case, "split": split, "category": cat, "group": group,
            "target": t,
            "true_slice": p["true"],
            "pred_slice": p["pred"] if p["pred"] > -900 else "uncovered",
            "err_slices": err, "err_mm": None if err is None else round(err * p["native_z_mm"], 1),
            "anchor": p["anchor"], "anchor_err": round(p["anchor"] - p["true"], 1),
            "pred_anchor_at_truth": ";".join(str(i) for i in p["intrinsic"]),
            "vote_std": p["vote_std"],
            "annotations_all": json.dumps(item["annotations"], ensure_ascii=False),
            "gaps_mm": json.dumps(item["gaps_mm"], ensure_ascii=False),
            "flags": ",".join(k for k, v in item["flags"].items() if v),
        })
    return rows


def signal_row(case, split, t, p, item, typical):
    """data_problems.md 全量信号表的一行（未覆盖目标不硬造毫米数）。"""
    name = f"`{case}` [{split}]" + ("（主报告典型）" if typical else "")
    flags = "+".join(FLAG_CN.get(k, k) for k, v in item["flags"].items() if v) or "—"
    intr = ", ".join(str(i) for i in p["intrinsic"])
    if p["pred"] > -900:
        err = p["pred"] - p["true"]
        err_s = f"{err:+d} 片 / {err * p['native_z_mm']:+.0f}mm"
        pred_s = str(p["pred"])
        vstd = str(p["vote_std"])
    else:
        err_s, pred_s, vstd = "——", "未覆盖", "——"
    return (f"| {name} | {t} | {p['true']} | {pred_s} | {err_s} "
            f"| {p['anchor']:.1f}（{p['anchor'] - p['true']:+.1f}） | {intr} | {vstd} | {flags} |")


def write_data_problems_md(out: Path, data_doc, pack_name):
    """生成 data_problems.md 全量骨架：总量构成 + 各类全量信号表 + 文件清单；
    成因讨论/逐例解读/仲裁结论留「模型撰写」占位。"""
    cats = data_doc["categories"]
    lines = [
        f"# 数据/标注问题全量分析（{pack_name}）",
        "",
        "- 范围：数据/标注问题的**全部**案例（不止主报告的典型案例）：结构违反"
        "（同片 / 乱序 / 间距越界）的全部案例（含未构成评测错误者）、全部"
        "「疑似标注错误」案例、任务仓库提供的外部嫌疑清单（由任务仓库的"
        "独立审计/复核工具产出，与导出管线相互独立）。",
        "- 口径与阅读约定（评测分母口径、错误判定、图上约定、各判断信号的含义）"
        "与主报告一致，见包内 badcase_analysis_report.md；本文件只陈述数据侧证据。",
        "- 本文件由组装脚本生成全量信号骨架；标「（模型撰写）」的小节由模型"
        "依据表内信号补充定稿，逐例解读必须引用具体数值；疑似类一律注明"
        "待人工复核。",
        "",
        "## 总量构成",
        "",
        "| 类别 | 案例数 | 标注平面数 |",
        "|---|---|---|",
    ]
    for cat, items in data_doc["selected"].items():
        cn = cats.get(cat, cat)
        n_planes = sum(len(r["pairs"]) for r in items)
        lines.append(f"| {cn} | {len(items)} | {n_planes} |")

    for cat, items in data_doc["selected"].items():
        cn = cats.get(cat, cat)
        lines += [
            "",
            f"## {cn}（{len(items)} 例）",
            "",
            "成因讨论：（模型撰写，3-5 句，基于下表信号与任务仓库上下文）",
            "",
            "### 全量案例信号表",
            "",
            "| 案例 [split] | 目标 | 标注切片 | 预测切片 | 误差（片/mm） "
            "| 锚点（偏差） | 锚点设在标注片时各模型的判断 | 判断分散程度 | 结构标志 |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for it in items:
            for t in it["pairs"]:
                lines.append(signal_row(it["case"], it["split"], t,
                                        it["pairs"][t], it, it.get("typical")))
        lines += [
            "",
            "### 典型案例逐例解读",
            "",
            "（模型撰写：标「主报告典型」的案例逐例 3-6 句解读，引用上表信号；"
            "非典型案例的共性要点写进成因讨论）",
            "",
            "### 本类案例文件（data_problems/ 下，全部扁平）",
            "",
        ]
        for it in items:
            cid = f"{it['case']}__{it['split']}"
            lines.append(f"- `{cid}_fig.png` ｜ `{cid}.nrrd` ｜ `{cid}_annotation.json`")

    lines += [
        "",
        "## 交叉统计与仲裁清单",
        "",
        "（程序化总量见「总量构成」；典型形态归纳、仲裁优先级与逐案结论由模型"
        "撰写；「疑似标注错误」与外部嫌疑清单两类必须整体标注为待人工复核/待仲裁，"
        "不得当作已证实。）",
        "",
    ]
    with open(out / "data_problems.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description="参考实现：路径由调用方（任务仓库）提供")
    ap.add_argument("--typical", default="runs/error_analysis/typical_cases.json")
    ap.add_argument("--img-dir", required=True,
                    help="原始影像目录（<case>.nrrd，任务仓库数据根提供）")
    ap.add_argument("--fig-dir", default="runs/error_analysis/figs",
                    help="渲染图目录（<split>__<case>.png），须覆盖典型 + 全量数据案例")
    ap.add_argument("--data", default=None,
                    help="data_problems.json：数据/标注问题全量清单（第 1 步 --data-out 产物）")
    ap.add_argument("--out", default=None,
                    help="包目录（缺省 runs/bad_case_analysis_<YYYYMMDDHHMM>，"
                         "生成时刻精确到分钟）")
    ap.add_argument("--keep-old", action="store_true",
                    help="保留同目录下的旧分析包（缺省组装成功后自动清理）")
    args = ap.parse_args()

    doc = json.load(open(args.typical))
    data_doc = json.load(open(args.data)) if args.data else None
    out = Path(args.out) if args.out else \
        Path("runs") / ("bad_case_analysis_"
                        + datetime.now().strftime("%Y%m%d%H%M"))
    cases_dir = out / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out / "data_problems"
    if data_doc:
        data_dir.mkdir(parents=True, exist_ok=True)

    # 数据问题全量案例（cid -> (类别, 条目)）；典型案例命中则不再进 cases/
    data_index = {}
    if data_doc:
        for cat, items in data_doc["selected"].items():
            for it in items:
                data_index[f"{it['case']}__{it['split']}"] = (cat, it)

    rows = []
    n_files = 0
    placed_tdata = set()
    sizes = {}
    for cat, cases in doc["selected"].items():
        for item in cases:
            cid = f"{item['case']}__{item['split']}"
            if data_doc and cid in data_index:
                rows += place_case(item, cat, "typical+data",
                                   args.img_dir, args.fig_dir, data_dir)
                placed_tdata.add(cid)
            else:
                rows += place_case(item, cat, "typical",
                                   args.img_dir, args.fig_dir, cases_dir)
            sizes[cid] = item["nrrd_size_mb"]
            n_files += 1

    n_data = 0
    if data_doc:
        for cat, items in data_doc["selected"].items():
            for item in items:
                cid = f"{item['case']}__{item['split']}"
                if cid in placed_tdata:
                    continue  # 已按 typical+data 收录，不重复落盘与写表
                rows += place_case(item, cat, "data",
                                   args.img_dir, args.fig_dir, data_dir)
                sizes[cid] = item["nrrd_size_mb"]
                n_data += 1
        write_data_problems_md(out, data_doc, out.name)

    with open(out / "annotations.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    total_mb = sum(sizes.values())
    print(f"pack -> {out}  typical={n_files}  data_full={n_data}  "
          f"nrrd_total={total_mb:.0f}MB  rows={len(rows)}")
    if args.keep_old:
        print("keep-old: 未清理旧分析包")
    else:
        removed = clean_old_packs(out)
        print(f"cleaned old packs: {len(removed)}" +
              ("  " + " ".join(str(p) for p in removed) if removed else ""))
    print(f"打包（默认 tar.gz；若任务仓库另有交付格式规定，按仓库规定压缩）："
          f"tar czf {out.parent / (out.name + '.tar.gz')} -C {out.parent} {out.name}")
    print("badcase_analysis_report.md 与 data_problems.md 的模型撰写小节"
          "需按 SKILL.md 第 4 步模板补齐")


if __name__ == "__main__":
    main()
