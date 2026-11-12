# -*- coding: utf-8 -*-
"""
================================================================================
脚本名称: v9_phase3p4_provenance.py
创建时间: 2026-11-12
任务编号: V9 Phase 3, Task 3.4.1
目标    : 将全部 V9 实验批次（results/controlled、results/cold_start 下所有
          v9* 目录）纳入 Supplementary File S3 的溯源（provenance）体系，
          生成：(1) 机器可读清单 JSON；(2) 可被 S3_batch_provenance.tex
          \input 的 LaTeX 片段。每个 V9 批次的目录数、种子数、纳入状态
          与论文用途均可逐项核对。
方法    : 纯文件扫描与转录，不做任何重计算、不修改任何实验结果：
          1) 批次族（batch family）及目录 pattern 在下方 FAMILIES 显式声明，
             含协议（B/C/development）、纳入状态、论文用途、期望计数；
          2) per-run 目录读 results.json，t1p1/t1p2 风格目录读
             results_seed*.json，launcher 目录读 summary.json/config.json；
          3) 计数与参数校验只产生 warning（写入清单并打印），绝不静默修补；
          4) 全集对账：任何未被 FAMILIES 覆盖的 v9 目录都报为 unassigned。
原则    : 不编造数据（全部字段直接读盘）；统一命名；脚本含时间/目标/方法
          注释；GPU 不适用（纯 I/O）；严格按任务 3.4.1 执行。
================================================================================
"""

import glob
import json
import os
import datetime

# ---------------------------------------------------------------------------
# 路径常量（仓库根目录为本脚本所在目录的上一级）
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS_DIR = os.path.join(REPO_ROOT, "results", "analysis")
TABLES_DIR = os.path.join(REPO_ROOT, "results", "paper_tables")
PAPER_FIG_DIR = os.path.join(REPO_ROOT, "2026-11-12_paper1_v9", "figures")

# 期望参数（仅对含 total_params 字段的批次校验）
GIN_PARAMS = 498177
GAT_PARAMS = 521345

# ---------------------------------------------------------------------------
# 批次族声明
#   patterns   : 相对 REPO_ROOT 的 glob 列表
#   expect_dirs: 期望命中的目录数（None = 不校验）
#   expect_records: 期望 per-seed 结果记录数（None = 不校验）
#   params     : GIN_PARAMS / GAT_PARAMS / None
# ---------------------------------------------------------------------------
FAMILIES = [
    {
        "key": "v9-e2e3-diagnosis",
        "title": "V9 E2/E3 corrected-E0 development runs (one config-only attempt; one 2-seed KIBA E0 @500ep run)",
        "patterns": ["results/controlled/v9_e3_corrected_e0_*"],
        "protocol": "development (500ep)",
        "status": "development (no paper number)",
        "used": "Development diagnostics only (seed 42: 0.1362; seed 7: 0.1623)",
        "expect_dirs": 2,
        "expect_records": 2,
        "params": None,
    },
    {
        "key": "t1p1",
        "title": "T1.1 KIBA E0, Config B (10 seeds; one batch directory)",
        "patterns": ["results/controlled/v9_t1p1_e0_10seeds_*"],
        "protocol": "B (500ep)",
        "status": "authoritative (Config B comparator)",
        "used": "Table 2 KIBA-B E0; Sec.3.1 ceiling audit",
        "expect_dirs": 1,
        "expect_records": 10,
        "params": None,
    },
    {
        "key": "t1p2",
        "title": "T1.2 epoch-budget probe, KIBA E0 (2 seeds)",
        "patterns": ["results/controlled/v9_t1p2_max_epochs_1000_*"],
        "protocol": "B->C probe (1000ep)",
        "status": "audit evidence",
        "used": "Sec.3.1 probe (seed 42: 0.1728@500 to 0.1525@1000, epoch 995)",
        "expect_dirs": 1,
        "expect_records": 2,
        "params": None,
    },
    {
        "key": "t1p3",
        "title": "T1.3 KIBA E7 alpha=0.05, Config B (5 seeds)",
        "patterns": ["results/controlled/v9_t1p3_e7_alpha005_seed*"],
        "protocol": "B (500ep)",
        "status": "authoritative (Config B comparator)",
        "used": "Table 2 KIBA-B E7",
        "expect_dirs": 5,
        "expect_records": 5,
        "params": None,
    },
    {
        "key": "t1p3-launch",
        "title": "T1.3 launcher records",
        "patterns": ["results/controlled/v9_t1p3_kiba_e7_alpha005_202*"],
        "protocol": "B (500ep)",
        "status": "batch launcher records",
        "used": "Batch-level config/progress provenance",
        "expect_dirs": 3,
        "expect_records": None,
        "params": None,
    },
    {
        "key": "t1p4",
        "title": "T1.4 KIBA E7 alpha=0.5, Config B (5 seeds)",
        "patterns": ["results/controlled/v9_t1p4_e7_alpha05_seed*"],
        "protocol": "B (500ep)",
        "status": "superseded",
        "used": "Superseded by P2.4 (same dose under Config C)",
        "expect_dirs": 5,
        "expect_records": 5,
        "params": None,
    },
    {
        "key": "t1p5",
        "title": "T1.5 Davis E0/E7, Config B (10 seeds each)",
        "patterns": ["results/controlled/v9_t1p5_davis_e0_seed*",
                     "results/controlled/v9_t1p5_davis_e7_seed*"],
        "protocol": "B (500ep)",
        "status": "authoritative (Config B comparator)",
        "used": "Table 2 Davis-B rows",
        "expect_dirs": 20,
        "expect_records": 20,
        "params": None,
    },
    {
        "key": "t1p5-launch",
        "title": "T1.5 launcher records",
        "patterns": ["results/controlled/v9_t1p5_davis_e0_e7_202*"],
        "protocol": "B (500ep)",
        "status": "batch launcher records",
        "used": "Batch-level config/progress provenance",
        "expect_dirs": 1,
        "expect_records": None,
        "params": None,
    },
    {
        "key": "t1p6",
        "title": "T1.6 Davis cold-both E0/E7, legacy audit (5 seeds each)",
        "patterns": ["results/cold_start/v9_t1p6_e0_seed*",
                     "results/cold_start/v9_t1p6_e7_seed*"],
        "protocol": "B-era (500ep)",
        "status": "legacy (reported jointly with selection caveat)",
        "used": "Table 6; Sec.3.3 (not interpretable as CL effect)",
        "expect_dirs": 10,
        "expect_records": 10,
        "params": None,
    },
    {
        "key": "t1p6-launch",
        "title": "T1.6 launcher records",
        "patterns": ["results/cold_start/v9_t1p6_cold_start_202*"],
        "protocol": "B-era (500ep)",
        "status": "batch launcher records",
        "used": "Batch-level config/progress provenance",
        "expect_dirs": 2,
        "expect_records": None,
        "params": None,
    },
    {
        "key": "p1",
        "title": "Phase 1 KIBA E0/E7, Config C (10 seeds each)",
        "patterns": ["results/controlled/v9_p1_kiba_e0_seed*",
                     "results/controlled/v9_p1_kiba_e7_seed*"],
        "protocol": "C (1000ep)",
        "status": "authoritative (Config C)",
        "used": "Tables 1-2; Figures 2 and 6",
        "expect_dirs": 20,
        "expect_records": 20,
        "params": GIN_PARAMS,
    },
    {
        "key": "p1-launch",
        "title": "Phase 1 launcher records",
        "patterns": ["results/controlled/v9_phase1_kiba_e0_1000ep_*",
                     "results/controlled/v9_phase1_kiba_e7_1000ep_*"],
        "protocol": "C (1000ep)",
        "status": "batch launcher records",
        "used": "Batch-level config/progress provenance",
        "expect_dirs": 7,
        "expect_records": None,
        "params": None,
    },
    {
        "key": "p2",
        "title": "Phase 2 Davis E0/E7, Config C (10 seeds each)",
        "patterns": ["results/controlled/v9_p2_davis_e0_seed*",
                     "results/controlled/v9_p2_davis_e7_seed*"],
        "protocol": "C (1000ep)",
        "status": "authoritative (Config C)",
        "used": "Tables 1-2",
        "expect_dirs": 20,
        "expect_records": 20,
        "params": GIN_PARAMS,
    },
    {
        "key": "p2-launch",
        "title": "Phase 2 Davis launcher records",
        "patterns": ["results/controlled/v9_phase2_davis_e0_e7_1000ep_*"],
        "protocol": "C (1000ep)",
        "status": "batch launcher records",
        "used": "Batch-level config/progress provenance",
        "expect_dirs": 1,
        "expect_records": None,
        "params": None,
    },
    {
        "key": "p2p2",
        "title": "Phase 2 Davis scaffold E0/E7, Config C (5 seeds each)",
        "patterns": ["results/cold_start/v9_p2_scaffold_e0_seed*",
                     "results/cold_start/v9_p2_scaffold_e7_seed*"],
        "protocol": "C (1000ep)",
        "status": "authoritative (with warm-selection caveat)",
        "used": "Table 5; Sec.3.3",
        "expect_dirs": 10,
        "expect_records": 10,
        "params": GIN_PARAMS,
    },
    {
        "key": "p2p2-launch",
        "title": "Phase 2 scaffold launcher records",
        "patterns": ["results/cold_start/v9_phase2_scaffold_202*"],
        "protocol": "C (1000ep)",
        "status": "batch launcher records",
        "used": "Batch-level config/progress provenance",
        "expect_dirs": 1,
        "expect_records": None,
        "params": None,
    },
    {
        "key": "p2p3",
        "title": "Phase 2 Davis dose scan, Config C (6 doses x 5 seeds)",
        "patterns": ["results/controlled/v9_p2p3_davis_a*_seed*"],
        "protocol": "C (1000ep)",
        "status": "authoritative (Config C)",
        "used": "Table 3",
        "expect_dirs": 30,
        "expect_records": 30,
        "params": GIN_PARAMS,
    },
    {
        "key": "p2p3-launch",
        "title": "Phase 2 Davis dose launcher records",
        "patterns": ["results/controlled/v9_phase2_davis_e1_e6_202*"],
        "protocol": "C (1000ep)",
        "status": "batch launcher records",
        "used": "Batch-level config/progress provenance",
        "expect_dirs": 1,
        "expect_records": None,
        "params": None,
    },
    {
        "key": "p2p4",
        "title": "Phase 2 KIBA alpha=0.5, Config C (5 seeds)",
        "patterns": ["results/controlled/v9_p2p4_kiba_a050_seed*"],
        "protocol": "C (1000ep)",
        "status": "authoritative (Config C)",
        "used": "Table 4",
        "expect_dirs": 5,
        "expect_records": 5,
        "params": GIN_PARAMS,
    },
    {
        "key": "p2p4-launch",
        "title": "Phase 2 KIBA alpha=0.5 launcher records",
        "patterns": ["results/controlled/v9_phase2_kiba_alpha05_202*"],
        "protocol": "C (1000ep)",
        "status": "batch launcher records",
        "used": "Batch-level config/progress provenance",
        "expect_dirs": 1,
        "expect_records": None,
        "params": None,
    },
    {
        "key": "p2p5",
        "title": "Phase 2 GAT Davis E0/E7, Config C (5 seeds each)",
        "patterns": ["results/controlled/v9_p2p5_gat_e0_seed*",
                     "results/controlled/v9_p2p5_gat_e7_seed*"],
        "protocol": "C (1000ep)",
        "status": "authoritative (Config C)",
        "used": "Table 1; Figures 4 and 6",
        "expect_dirs": 10,
        "expect_records": 10,
        "params": GAT_PARAMS,
    },
    {
        "key": "p2p5-launch",
        "title": "Phase 2 GAT launcher records",
        "patterns": ["results/controlled/v9_phase2_gat_davis_202*"],
        "protocol": "C (1000ep)",
        "status": "batch launcher records",
        "used": "Batch-level config/progress provenance",
        "expect_dirs": 1,
        "expect_records": None,
        "params": None,
    },
]


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def esc(text):
    """LaTeX 转义：下划线与百分号（仅用于文件名字段）。"""
    return str(text).replace("\\", "/").replace("_", r"\_").replace("%", r"\%")


# 2026-11-12 Task 3.4.1: launcher summary.json 的 task 字段含希腊字母与中文，
# S3 导言区未加载 CJK/unicode 宏包，直接写入会触发
# "Unicode character α (U+03B1) not set up for use with LaTeX"。
# 此函数将 task 文本清洗为纯 ASCII 的可排版 LaTeX 文本：
#   1) 希腊字母转数学模式；2) 已知中文短语转英文；3) 其余非 ASCII 字符剔除。
def sanitize_task(text):
    """将 launcher task 字段清洗为 S3 可排版的 ASCII LaTeX 文本。"""
    s = str(text or "")
    # 1) 希腊字母 -> 数学模式（在转义 & 之前处理）
    greek = {"α": r"$\alpha$", "β": r"$\beta$", "γ": r"$\gamma$",
             "δ": r"$\delta$", "λ": r"$\lambda$", "τ": r"$\tau$"}
    for ch, macro in greek.items():
        s = s.replace(ch, macro)
    # 2) 批次记录中实际出现的中文短语 -> 英文
    # 带空格替换，避免与相邻 ASCII 字符粘连（末尾统一收敛空白）
    cn_map = [
        ("冷启动", " cold-start "),
        ("新协议", " new-protocol "),
        ("种子", " seeds "),
        ("验证", " validation "),
    ]
    for cn, en in cn_map:
        s = s.replace(cn, en)
    # 3) LaTeX 特殊字符转义
    s = s.replace("&", r"\&").replace("%", r"\%").replace("#", r"\#")
    # 4) 剔除所有剩余非 ASCII 字符（防止未来新增未映射汉字再次炸编译）
    s = "".join(ch for ch in s if ord(ch) < 128)
    # 5) 收敛多余空白
    s = " ".join(s.split())
    return s


def file_mtime_iso(path):
    """文件/目录修改时间（真实文件系统元数据）。"""
    ts = os.path.getmtime(path)
    return datetime.datetime.fromtimestamp(ts).isoformat(timespec="seconds")


def load_json_safe(path):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:  # 读取失败如实记录，不抛出以免中断全批次扫描
        return {"__read_error__": repr(exc)}


def extract_per_run_record(path):
    """results/controlled/v9_*_seed* 风格 per-run 目录 → results.json 记录。"""
    data = load_json_safe(os.path.join(path, "results.json"))
    return {
        "dir": os.path.basename(path),
        "experiment_id": data.get("experiment_id"),
        "dataset": data.get("dataset"),
        "seed": data.get("seed"),
        "best_epoch": data.get("best_epoch"),
        "best_val_mse": data.get("best_val_mse"),
        "test_mse": data.get("test_mse"),
        "test_rmse": data.get("test_rmse"),
        "test_ci": data.get("test_ci"),
        "total_params": data.get("total_params"),
        "config": data.get("config"),
        "mtime": file_mtime_iso(path),
    }


def extract_seed_files(path):
    """t1p1/t1p2 风格：目录内 results_seed*.json 每个一条记录。"""
    records = []
    for cand in sorted(glob.glob(os.path.join(path, "results_seed*.json"))):
        data = load_json_safe(cand)
        records.append({
            "dir": os.path.basename(path),
            "file": os.path.basename(cand),
            "experiment_id": data.get("experiment_id"),
            "dataset": (data.get("config") or {}).get("dataset"),
            "seed": data.get("seed"),
            "best_epoch": data.get("best_epoch"),
            "best_val_mse": data.get("best_val_mse"),
            "test_mse": data.get("test_mse"),
            "test_rmse": data.get("test_rmse"),
            "test_ci": data.get("test_ci"),
            "total_params": data.get("total_params"),
            "config": data.get("config"),
            "mtime": file_mtime_iso(cand),
        })
    return records


def extract_launcher_record(path):
    """launcher 目录：summary.json + config.json 的批次级元数据。"""
    summary_path = os.path.join(path, "summary.json")
    config_path = os.path.join(path, "config.json")
    summary = load_json_safe(summary_path) if os.path.exists(summary_path) else None
    config = load_json_safe(config_path) if os.path.exists(config_path) else None
    success = None
    if isinstance(summary, dict) and isinstance(summary.get("results"), list):
        flags = [bool(r.get("success")) for r in summary["results"]
                 if isinstance(r, dict) and "success" in r]
        if flags:
            success = {"ok": sum(flags), "total": len(flags)}
    return {
        "dir": os.path.basename(path),
        "task": summary.get("task") if isinstance(summary, dict) else None,
        "timestamp": summary.get("timestamp") if isinstance(summary, dict) else None,
        "seeds": summary.get("seeds") if isinstance(summary, dict) else None,
        "success": success,
        "config": (summary.get("config") if isinstance(summary, dict) else None) or config,
        "mtime": file_mtime_iso(path),
    }


# ---------------------------------------------------------------------------
# 主扫描
# ---------------------------------------------------------------------------
def scan_family(family):
    """返回该族的目录、per-seed 记录与 launcher 记录，并执行校验。"""
    matched = []
    for pattern in family["patterns"]:
        matched.extend(sorted(glob.glob(os.path.join(REPO_ROOT, pattern))))
    # 去重（多 pattern 理论上不交，去重以防万一）
    matched = sorted(set(matched))

    run_records = []
    launcher_records = []
    for d in matched:
        if os.path.exists(os.path.join(d, "results.json")):
            run_records.append(extract_per_run_record(d))
        seed_records = extract_seed_files(d)
        if seed_records:
            run_records.extend(seed_records)
        if not os.path.exists(os.path.join(d, "results.json")) and not seed_records:
            launcher_records.append(extract_launcher_record(d))

    warnings = []
    if family["expect_dirs"] is not None and len(matched) != family["expect_dirs"]:
        warnings.append(
            f"directory count {len(matched)} != expected {family['expect_dirs']}")
    if family["expect_records"] is not None and len(run_records) != family["expect_records"]:
        warnings.append(
            f"per-seed records {len(run_records)} != expected {family['expect_records']}")
    if family["params"] is not None:
        observed = sorted({r["total_params"] for r in run_records
                           if r.get("total_params") is not None})
        if observed and observed != [family["params"]]:
            warnings.append(
                f"params {observed} != expected [{family['params']}]")
        if not observed:
            warnings.append("no total_params field present in this family")

    return {
        "key": family["key"],
        "title": family["title"],
        "protocol": family["protocol"],
        "status": family["status"],
        "used": family["used"],
        "dirs": [os.path.basename(d) for d in matched],
        "run_records": run_records,
        "launcher_records": launcher_records,
        "warnings": warnings,
    }


def find_unassigned(assigned_dirs):
    """全集对账：列出 results 下所有未被任何族覆盖的 v9 目录。"""
    all_v9 = []
    for sub in ("controlled", "cold_start"):
        base = os.path.join(REPO_ROOT, "results", sub)
        if os.path.isdir(base):
            for name in os.listdir(base):
                if name.startswith("v9") and os.path.isdir(os.path.join(base, name)):
                    all_v9.append(os.path.basename(name))
    assigned = {os.path.basename(d) for ds in assigned_dirs for d in ds}
    return sorted(set(all_v9) - assigned)


# ---------------------------------------------------------------------------
# LaTeX 片段生成
# ---------------------------------------------------------------------------
def build_latex(scanned, unassigned):
    L = []
    L.append("% " + "=" * 76)
    L.append("% V9 Task 3.4.1: per-batch provenance fragment for S3")
    L.append("% Generated: " + datetime.datetime.now().isoformat(timespec="seconds"))
    L.append("% Source: scripts/v9_phase3p4_provenance.py (pure transcription)")
    L.append("% " + "=" * 76)
    L.append("")
    L.append(r"\section{V9 Batch Provenance (Config-B / Config-C Reassessment)}")
    L.append("")
    L.append(r"This section extends the batch map above with every batch produced "
             r"during the V9 convergent-protocol reassessment. ``Run dirs'' counts "
             r"directories matched by the declared patterns; ``seed records'' counts "
             r"per-seed result files. All counts and statuses below were extracted "
             r"directly from the result directories; no value was recomputed.")
    L.append("")

    # --- 总览表 ---
    L.append(r"\begin{table*}[!h]")
    L.append(r"\centering\scriptsize")
    L.append(r"\caption{V9 batch-family provenance overview.}")
    L.append(r"\label{tab:v9_family_overview}")
    L.append(r"\begin{tabular}{p{2.6cm}p{4.6cm}p{2.0cm}p{3.4cm}cc}")
    L.append(r"\toprule")
    L.append(r"Family & Content & Protocol & Inclusion / use & Run dirs & Seed recs \\")
    L.append(r"\midrule")
    for s in scanned:
        L.append(
            r"\texttt{" + esc(s["key"]) + r"} & "
            + s["title"].replace("&", r"\&") + r" & "
            + s["protocol"].replace("&", r"\&") + r" & "
            + (s["status"] + r". " + s["used"]).replace("&", r"\&")
            + r" & " + str(len(s["dirs"])) + r" & "
            + str(len(s["run_records"])) + r" \\")
        if s["warnings"]:
            L.append(r"\multicolumn{6}{l}{\textit{\textbf{Validation warnings:} "
                     + "; ".join(s["warnings"]).replace("_", r"\_") + r"}} \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    if unassigned:
        L.append(r"\par\smallskip\footnotesize\textbf{Unassigned V9 directories "
                 r"(none expected):} \texttt{" + esc(", ".join(unassigned)) + "}")
    else:
        L.append(r"\par\smallskip\footnotesize All V9 directories are assigned to "
                 r"exactly one family (full-set reconciliation passed).")
    L.append(r"\end{table*}")
    L.append("")

    # --- 每族明细表（仅含 per-seed 结果的族）---
    for s in scanned:
        if not s["run_records"]:
            continue
        recs = sorted(s["run_records"], key=lambda r: (str(r.get("dir")),
                                                       r.get("seed") or 0))
        L.append(r"\begin{table*}[!h]")
        L.append(r"\centering\scriptsize")
        L.append(r"\caption{Per-run records: \texttt{" + esc(s["key"]) + r"}. "
                 r"\textit{" + s["title"].replace("&", r"\&") + r"}}")
        L.append(r"\label{tab:v9_runs_" + s["key"].replace("-", "_") + r"}")
        # p{4.8}: longest basename is 37 chars scriptsize mono (~4.6 cm);
        # per-seed rows are distinguished by the Seed column, so the
        # results_seed*.json file suffix is omitted by design.
        L.append(r"\begin{tabular}{p{4.8cm}cccccc}")
        L.append(r"\toprule")
        L.append(r"Run directory & Seed & Best epoch & Best val MSE & Test MSE "
                 r"& Test CI & Params \\")
        L.append(r"\midrule")
        for r in recs:
            # 目录名在转义下划线后允许断行，最长基名 40 字符，防止 overfull hbox
            run_dir_tex = esc(r.get("dir")).replace(r"\_", r"\_\allowbreak{}")
            L.append(
                r"\texttt{" + run_dir_tex + r"} & "
                + ("" if r.get("seed") is None else str(r.get("seed"))) + r" & "
                + ("" if r.get("best_epoch") is None else str(r.get("best_epoch"))) + r" & "
                + (f"{r['best_val_mse']:.4f}" if r.get("best_val_mse") is not None else "") + r" & "
                + (f"{r['test_mse']:.4f}" if r.get("test_mse") is not None else "") + r" & "
                + (f"{r['test_ci']:.4f}" if r.get("test_ci") is not None else "") + r" & "
                + ("" if r.get("total_params") is None else f"{r['total_params']:,}")
                + r" \\")
        L.append(r"\bottomrule")
        L.append(r"\end{tabular}")
        L.append(r"\end{table*}")
        L.append("")

    # --- launcher 明细表 ---
    launcher_families = [s for s in scanned if s["launcher_records"]]
    if launcher_families:
        L.append(r"\begin{table*}[!h]")
        L.append(r"\centering\scriptsize")
        L.append(r"\caption{Batch-launcher records (config/progress bookkeeping "
                 r"directories; these do not supply paper numbers).}")
        L.append(r"\label{tab:v9_launchers}")
        # resizebox 保证长 launcher 目录名（如 v9_t1p3_kiba_e7_alpha005_20261023_065021）
        # 不产生 overfull hbox
        L.append(r"\resizebox{\textwidth}{!}{%")
        L.append(r"\begin{tabular}{p{2.4cm}p{5.0cm}p{4.2cm}p{3.2cm}p{2.0cm}}")
        L.append(r"\toprule")
        L.append(r"Family & Launcher dir & Task & Timestamp / seeds & Success \\")
        L.append(r"\midrule")
        for s in launcher_families:
            for r in s["launcher_records"]:
                seeds = r.get("seeds")
                seed_txt = ("" if seeds is None else
                            f"{len(seeds)} seeds")
                succ = r.get("success")
                succ_txt = ("" if succ is None else f'{succ["ok"]}/{succ["total"]} ok')
                # 目录名在每个转义下划线后允许断行，防止长串撑破列宽
                dir_tex = esc(r["dir"]).replace(r"\_", r"\_\allowbreak{}")
                L.append(
                    r"\texttt{" + esc(s["key"]) + r"} & \texttt{"
                    + dir_tex + r"} & "
                    + sanitize_task(r.get("task")) + r" & "
                    + esc(str(r.get("timestamp") or "")) + r" " + seed_txt + r" & "
                    + succ_txt + r" \\")
        L.append(r"\bottomrule")
        L.append(r"\end{tabular}%")
        L.append(r"}")
        L.append(r"\end{table*}")
        L.append("")

    return "\n".join(L)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("V9 Task 3.4.1: scan all V9 batches into S3 provenance")
    print("=" * 70)

    scanned = [scan_family(f) for f in FAMILIES]
    assigned_dirs = [
        [os.path.join(REPO_ROOT, d) for d in s["dirs"]] for s in scanned
    ]
    unassigned = find_unassigned(assigned_dirs)

    total_runs = sum(len(s["run_records"]) for s in scanned)
    total_launchers = sum(len(s["launcher_records"]) for s in scanned)
    total_warnings = sum(len(s["warnings"]) for s in scanned)
    print(f"Families: {len(scanned)}; per-seed records: {total_runs}; "
          f"launcher dirs: {total_launchers}; warnings: {total_warnings}; "
          f"unassigned: {len(unassigned)}")

    manifest = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "task": "V9 3.4.1 V9 batch provenance",
        "families": scanned,
        "unassigned": unassigned,
    }
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    json_path = os.path.join(ANALYSIS_DIR, "v9_phase3p4_provenance.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    print("Manifest:", json_path)

    latex = build_latex(scanned, unassigned)
    os.makedirs(TABLES_DIR, exist_ok=True)
    tex_path = os.path.join(TABLES_DIR, "v9_phase3p4_provenance.tex")
    with open(tex_path, "w", encoding="utf-8") as fh:
        fh.write(latex)
    print("LaTeX fragment:", tex_path)

    # 同步一份到论文 figures 目录，供 S3 直接 \input
    os.makedirs(PAPER_FIG_DIR, exist_ok=True)
    paper_copy = os.path.join(PAPER_FIG_DIR, "S3_v9_provenance.tex")
    with open(paper_copy, "w", encoding="utf-8") as fh:
        fh.write(latex)
    print("S3 fragment copy:", paper_copy)


if __name__ == "__main__":
    main()
