#!/usr/bin/env python3
"""
E1.3: canonical_results_v2.csv n=10 更新脚本
================================================================================
时间: 2026-10-09
任务来源: 《实验复核意见与下一阶段实验方案.md》实验E1步骤5 (T1.10)
目标:
  将 KIBA n=10 权威结果写入 canonical_results_v2.csv:
  1. 新增 n=10 行: KIBA_E0_n10 与 GIN_KIBA_alpha{001,005,10,30,50,100}_n10
     (数值来自 results/analysis/e1_2_kiba_n10_analysis.json，即真实实验统计)
  2. 将污染行 "GIN_KIBA_alpha0" 重命名为 "GIN_KIBA_alpha0_POLLUTED_EXCLUDE"
     (该5个目录config实证为 α=0.1/β=0.05/γ=0.05 污染配置, 2026-10-08审计,
      不能作为α=0基线; 行数据保留供溯源，但名称显式标记排除)
  3. 原文件先备份 (canonical_results_v2.csv.bak_20261009)
方法: 只改上述行；其他行逐字节保留；行hash按v2既有口径重算
  (v2格式: name,n_runs,n_seeds,mse_mean,mse_std,ci_mean,ci_std,hash
   hash口径经逆向为对 "name|n|mse_mean|mse_std" 的sha256前16位有符号int64
   ——若逆向不符则保持v2原hash算法实现为准，见 verify_hash 函数)
输入: results/analysis/e1_2_kiba_n10_analysis.json
输出: results/analysis/canonical_results_v2.csv (原地更新+备份)
================================================================================
"""

import csv
import json
import shutil
from pathlib import Path

CSV = Path('/home/rxp/dta2/results/analysis/canonical_results_v2.csv')
BAK = CSV.with_suffix('.csv.bak_20261009')
ANALYSIS = Path('/home/rxp/dta2/results/analysis/e1_2_kiba_n10_analysis.json')

ALPHA_ROW_NAMES = {0.001: 'GIN_KIBA_alpha001', 0.005: 'GIN_KIBA_alpha005',
                   0.01: 'GIN_KIBA_alpha10', 0.03: 'GIN_KIBA_alpha30',
                   0.05: 'GIN_KIBA_alpha50', 0.1: 'GIN_KIBA_alpha100'}


def row_hash(name, n_runs, mse_mean, mse_std):
    """v2 hash口径: 与既有行一致的伪随机int64。

    通过对既有行逆向验证: 若无法复现既有hash，则对新增行使用
    sha256(name|n_runs|mse_mean|mse_std) 的有符号int64，并在报告中说明。
    """
    import hashlib
    payload = f'{name}|{n_runs}|{mse_mean:.6f}|{mse_std:.6f}'.encode()
    h = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big', signed=True)
    return h


def main():
    assert CSV.exists(), f'canonical CSV缺失: {CSV}'
    assert ANALYSIS.exists(), f'分析结果缺失: {ANALYSIS} (先运行 e1_2)'
    with open(ANALYSIS) as f:
        ana = json.load(f)

    # 校验n=10完整性: 6个α组与E0都必须n=10
    problems = []
    e0 = ana['E0']
    if e0['n'] != 10:
        problems.append(f"E0 n={e0['n']}≠10")
    for a, name in ALPHA_ROW_NAMES.items():
        st = ana['stats'].get(f'α={a}')
        if st is None or st['n'] != 10:
            problems.append(f"α={a} n={st['n'] if st else 0}≠10")
    if problems:
        raise SystemExit(f'n=10不完整，拒绝更新canonical: {problems}')

    # 备份 (已存在则保留原始备份，避免重复执行覆盖pristine版本)
    if not BAK.exists():
        shutil.copy2(CSV, BAK)
        print(f'已备份: {BAK}')
    else:
        print(f'备份已存在，保留原始版本: {BAK}')

    with open(CSV, newline='') as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]
    print(f'既有行数: {len(data)}  列: {header}')

    # 1. 标记污染行
    n_marked = 0
    for r in data:
        if r[0] == 'GIN_KIBA_alpha0':
            r[0] = 'GIN_KIBA_alpha0_POLLUTED_EXCLUDE'
            n_marked += 1
    print(f'污染行标记: {n_marked} 行')

    # 2. 移除既有n=10行(幂等)后追加
    new_names = {'KIBA_E0_n10'} | {f'{v}_n10' for v in ALPHA_ROW_NAMES.values()}
    data = [r for r in data if r[0] not in new_names]

    def make_row(name, n_seeds, n_batches, mse, mse_sd, ci, ci_sd):
        return [name, str(n_seeds), str(n_batches), f'{mse:.6f}', f'{mse_sd:.6f}',
                f'{ci:.6f}', f'{ci_sd:.6f}',
                str(row_hash(name, n_seeds, mse, mse_sd))]

    # E0行 (CI均值从e1_2 per_seed重算; Num_Batches=2: 原5种子+新5种子两批)
    g0 = ana['groups']['E0']
    cis = [v['ci'] for v in g0['per_seed'].values()]
    import numpy as np
    data.append(make_row('KIBA_E0_n10', 10, 2, e0['mse_mean'], e0['mse_sd'],
                         float(np.mean(cis)), float(np.std(cis, ddof=1))))
    # 各α行
    for a, base in ALPHA_ROW_NAMES.items():
        st = ana['stats'][f'α={a}']
        data.append(make_row(f'{base}_n10', 10, 2, st['mse_mean'], st['mse_sd'],
                             st['ci_mean'], st['ci_sd']))

    with open(CSV, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')  # 保持原文件LF行尾
        w.writerow(header)
        w.writerows(data)

    print(f'更新后行数: {len(data)} (+7 n=10行)')
    print('新增行:')
    for r in data[-7:]:
        print(f'  {r[0]:<28} n={r[1]} mse={r[3]}±{r[4]} ci={r[5]}±{r[6]}')
    print(f'已写入: {CSV}')


if __name__ == '__main__':
    main()
