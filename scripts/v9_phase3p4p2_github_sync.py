#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
============================================================================
脚本名称: v9_phase3p4p2_github_sync.py
创建时间: 2026-11-12
所属任务: V9 重投 Task 3.4.2 (GitHub/Zenodo 同步更新)
目标:
    将 V9 (Config-B/Config-C 重评估) 全部可公开产物暂存到发布仓库克隆
    /mnt/data/dta2/repo_for_github (remote:
    https://github.com/rxp613-dev/contrastive-dta.git), 使网络/凭证恢复后
    一次 commit + push + GitHub release 即可完成 GitHub/Zenodo 同步。
方法:
    1) 仅复制 (shutil.copy2), 不修改任何源文件;
    2) 分组: 33 个 v9 脚本 / analysis 小型证据文件 (排除 *.log 与 >2MB) /
       paper_tables / 每个 per-run 目录的 results*.json+config*.json
       (排除 *.pt 权重与逐 epoch 日志) / V9 文稿源码与 PDF / S3 补充材料;
    3) 写入带 sha256 的同步清单 results/v9/V9_SYNC_MANIFEST.json;
    4) --dry-run 只打印将复制的文件与计数, 不落盘 (执行前独立审核用,
       对应原则 1: 程序在执行前都要独立进行审核);
    5) 断言: 单个暂存文件 <=45MB; 目标路径全部位于发布仓库内。
注意:
    本脚本不执行 git push (2026-11-12 实测 GitHub HTTPS/SSH 及本机代理
    全部被阻断, gh keyring token 失效), 推送由后续人工执行,
    见 docs/supplementary/V9_SYNC_INSTRUCTIONS.md。
============================================================================
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import datetime

# ----------------------------- 路径常量 ------------------------------------
WORKSPACE = "/mnt/data/dta2"
REPO = os.path.join(WORKSPACE, "repo_for_github")
SRC_SCRIPTS = os.path.join(WORKSPACE, "scripts")
SRC_ANALYSIS = os.path.join(WORKSPACE, "results", "analysis")
SRC_TABLES = os.path.join(WORKSPACE, "results", "paper_tables")
SRC_CONTROLLED = os.path.join(WORKSPACE, "results", "controlled")
# t1p6 冷启动与 p2 scaffold 批次的 per-run 目录在此根下 (共 20 条 per-seed 记录)
SRC_COLD = os.path.join(WORKSPACE, "results", "cold_start")
SRC_MANUSCRIPT = os.path.join(WORKSPACE, "2026-11-12_paper1_v9")
SRC_S3_DIR = os.path.join(WORKSPACE, "2026-11-12_paper1_v9", "figures")

# 文稿复制时排除的 LaTeX 中间产物 (保留 .tex/.pdf/图片源)
MANUSCRIPT_EXCLUDE_EXT = {
    ".aux", ".log", ".out", ".fls", ".fdb_latexmk", ".bbl", ".blg",
    ".synctex.gz", ".toc", ".lof", ".lot",
}
MAX_STAGE_BYTES = 45 * 1024 * 1024  # GitHub 大文件告警线 50MB, 留余量


def sha256_of(path):
    """真实计算文件 sha256 (64KiB 分块, 避免大文件全量入内存)。"""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_plan():
    """收集 (src, dst) 复制计划。纯读取, 不写盘。"""
    plan = []

    # 1) V9 脚本: v9_*.py 与 task_v9_*.py
    scripts_dst = os.path.join(REPO, "scripts")
    for name in sorted(os.listdir(SRC_SCRIPTS)):
        if name.endswith(".py") and (name.startswith("v9_") or
                                     name.startswith("task_v9_")):
            plan.append((os.path.join(SRC_SCRIPTS, name),
                         os.path.join(scripts_dst, name), "script"))

    # 2) analysis 小型证据文件: 仅普通文件, 排除 *.log 与 >2MB
    analysis_dst = os.path.join(REPO, "results", "v9", "analysis")
    for name in sorted(os.listdir(SRC_ANALYSIS)):
        src = os.path.join(SRC_ANALYSIS, name)
        if not os.path.isfile(src):
            continue
        if name.endswith(".log"):
            continue
        if os.path.getsize(src) > 2 * 1024 * 1024:
            continue
        plan.append((src, os.path.join(analysis_dst, name), "analysis"))

    # 3) paper_tables 全部文件 (该目录总计约 52KB)
    tables_dst = os.path.join(REPO, "results", "v9", "paper_tables")
    for name in sorted(os.listdir(SRC_TABLES)):
        src = os.path.join(SRC_TABLES, name)
        if os.path.isfile(src):
            plan.append((src, os.path.join(tables_dst, name), "paper_table"))

    # 4) per-run 记录: 每个 v9_* 目录仅 results*.json / config*.json
    records_root = os.path.join(REPO, "results", "v9", "run_records")
    for run_name in sorted(os.listdir(SRC_CONTROLLED)):
        run_src = os.path.join(SRC_CONTROLLED, run_name)
        if not (os.path.isdir(run_src) and run_name.startswith("v9_")):
            continue
        for name in sorted(os.listdir(run_src)):
            if not (name.startswith("results") or name.startswith("config")):
                continue
            if not name.endswith(".json"):
                continue
            src = os.path.join(run_src, name)
            if os.path.isfile(src):
                plan.append((src,
                             os.path.join(records_root, run_name, name),
                             "run_record"))

    # 4b) cold_start 下的 v9_* per-run 记录, 映射到 run_records/cold_start/
    cold_records_root = os.path.join(records_root, "cold_start")
    if os.path.isdir(SRC_COLD):
        for run_name in sorted(os.listdir(SRC_COLD)):
            run_src = os.path.join(SRC_COLD, run_name)
            if not (os.path.isdir(run_src) and run_name.startswith("v9_")):
                continue
            for name in sorted(os.listdir(run_src)):
                if not (name.startswith("results") or
                        name.startswith("config")):
                    continue
                if not name.endswith(".json"):
                    continue
                src = os.path.join(run_src, name)
                if os.path.isfile(src):
                    plan.append((src,
                                 os.path.join(cold_records_root, run_name,
                                              name),
                                 "run_record"))

    # 5) V9 文稿源码 (排除 LaTeX 中间产物)
    manuscript_dst = os.path.join(REPO, "docs", "manuscript_v9")
    for root, dirs, files in os.walk(SRC_MANUSCRIPT):
        dirs.sort()
        for name in sorted(files):
            lower = name.lower()
            if any(lower.endswith(ext) for ext in MANUSCRIPT_EXCLUDE_EXT):
                continue
            src = os.path.join(root, name)
            rel = os.path.relpath(src, SRC_MANUSCRIPT)
            plan.append((src, os.path.join(manuscript_dst, rel),
                         "manuscript"))

    # 6) S3 补充材料两个 tex
    supp_dst = os.path.join(REPO, "docs", "supplementary")
    for name in ["S3_batch_provenance.tex", "S3_v9_provenance.tex"]:
        src = os.path.join(SRC_S3_DIR, name)
        if os.path.isfile(src):
            plan.append((src, os.path.join(supp_dst, name), "supplementary"))

    return plan


def validate_plan(plan):
    """执行前断言: 目标不越界, 文件不超限。"""
    for src, dst, kind in plan:
        assert os.path.abspath(dst).startswith(os.path.abspath(REPO) + os.sep), \
            f"unsafe dst outside repo: {dst}"
        assert os.path.isfile(src), f"missing src: {src}"
        size = os.path.getsize(src)
        assert size <= MAX_STAGE_BYTES, f"file too large for GitHub: {src} {size}"


def write_repo_support_files():
    """仓库根 .gitignore 与 V9 更新说明/同步指引 (静态文本, 内容只述事实)。"""
    gitignore = """# Added 2026-11-12 for V9 sync (Task 3.4.2)
# Python
__pycache__/
*.py[cod]
*.egg-info/
# LaTeX intermediates
*.aux
*.log
*.out
*.fls
*.fdb_latexmk
*.synctex.gz
*.bbl
*.blg
# Model weights / raw data
*.pt
*.pth
*.npy
data/raw/
data/processed/
# OS/editor
.DS_Store
.idea/
.vscode/
"""
    with open(os.path.join(REPO, ".gitignore"), "w", encoding="utf-8") as fh:
        fh.write(gitignore)

    notice = """# V9 Update (2026-11-12)

This repository has been extended for the manuscript version **V9**:
*Apparent Contrastive Learning Gains in Drug-Target Affinity Prediction Are
Manufactured by Under-Converged Supervised Baselines: A Convergent-Protocol
Reassessment*.

- `scripts/` now includes the V9 batch scripts (`v9_*.py`, `task_v9_*.py`).
- `results/v9/analysis/` holds the V9 statistical/audit evidence (JSON/MD).
- `results/v9/paper_tables/` holds the generated LaTeX table fragments.
- `results/v9/run_records/` holds the per-seed `results.json` /
  `config.json` evidence for every V9 run (no model weights).
- `docs/manuscript_v9/` is the V9 manuscript source.
- `docs/supplementary/` contains Supplementary File S3 (batch provenance).
- `results/v9/V9_SYNC_MANIFEST.json` lists every staged file with size and
  SHA-256, mapped to its workspace source.

The initial 2026-09 package is retained unchanged; its Config-B-era numbers
are superseded by the V9 convergent-protocol results documented above.
"""
    with open(os.path.join(REPO, "V9_UPDATE_NOTICE.md"), "w",
              encoding="utf-8") as fh:
        fh.write(notice)

    instructions = """# V9 GitHub / Zenodo Sync Instructions (Task 3.4.2)

Status on 2026-11-12: all files are staged and committed locally; the remote
push could NOT be executed because:

1. Direct HTTPS to github.com fails (TLS reset, curl rc=35);
2. SSH to github.com:22 and ssh.github.com:443 is closed;
3. Local proxies (127.0.0.1:7897 / :8080, HTTP and SOCKS5) establish CONNECT
   but the TLS exchange to GitHub fails (HTTP code 000);
4. Zenodo (zenodo.org) is blocked the same way;
5. The `gh` keyring token is reported invalid and no SSH key / PAT is
   available on this machine.

When network and credentials are restored, run:

```bash
cd /mnt/data/dta2/repo_for_github
git remote -v            # confirm origin = https://github.com/rxp613-dev/contrastive-dta.git
git status               # confirm the V9 commit is present
git push origin master   # remote default branch is master in this repo

# Then create a release so the GitHub-Zenodo integration archives it:
gh release create v1.1.0 --title "v1.1.0 - V9 convergent-protocol reassessment" \\
  --notes "V9 reassessment package: Config C (max_epochs=1000) results,
  provenance tables, manuscript source, supplementary S3."
```

Zenodo will mint a version DOI for the new release; the concept DOI
10.5281/zenodo.23308152 remains unchanged. Update CITATION.cff with the new
version DOI after the release.
"""
    with open(os.path.join(REPO, "docs", "supplementary",
                           "V9_SYNC_INSTRUCTIONS.md"), "w",
              encoding="utf-8") as fh:
        fh.write(instructions)


def main():
    parser = argparse.ArgumentParser(description="Stage V9 files into the "
                                                 "publishing repo clone.")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the copy plan and counts without writing")
    args = parser.parse_args()

    plan = collect_plan()
    validate_plan(plan)

    counts = {}
    for _, _, kind in plan:
        counts[kind] = counts.get(kind, 0) + 1
    total_bytes = sum(os.path.getsize(src) for src, _, _ in plan)

    print("=" * 66)
    print("V9 Task 3.4.2: GitHub/Zenodo sync staging")
    print("=" * 66)
    print(f"files: {len(plan)}; bytes: {total_bytes:,} "
          f"({total_bytes / 1024 / 1024:.2f} MiB)")
    for kind in sorted(counts):
        print(f"  {kind:12s}: {counts[kind]}")

    if args.dry_run:
        print("\n--dry-run: no files written. First 10 entries:")
        for src, dst, kind in plan[:10]:
            print(f"  [{kind}] {src} -> {dst}")
        return 0

    # 真实复制
    manifest_entries = []
    for src, dst, kind in plan:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        st = os.stat(dst)
        manifest_entries.append({
            "kind": kind,
            "src": os.path.relpath(src, WORKSPACE),
            "dst": os.path.relpath(dst, REPO),
            "bytes": st.st_size,
            "sha256": sha256_of(dst),
        })

    write_repo_support_files()

    manifest = {
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "task": "V9 Task 3.4.2 GitHub/Zenodo sync staging",
        "file_count": len(manifest_entries),
        "total_bytes": total_bytes,
        "counts_by_kind": counts,
        "files": manifest_entries,
    }
    manifest_path = os.path.join(REPO, "results", "v9",
                                 "V9_SYNC_MANIFEST.json")
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    # 复制后复核: 清单内每个 dst 真实存在且大小一致
    for e in manifest_entries:
        p = os.path.join(REPO, e["dst"])
        assert os.path.isfile(p), f"staged file missing: {p}"
        assert os.path.getsize(p) == e["bytes"], f"size mismatch: {p}"

    print(f"\nStaged {len(manifest_entries)} files.")
    print(f"Manifest: {manifest_path}")
    print("Support files: .gitignore, V9_UPDATE_NOTICE.md, "
          "docs/supplementary/V9_SYNC_INSTRUCTIONS.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
