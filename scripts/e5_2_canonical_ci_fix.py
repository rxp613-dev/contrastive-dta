#!/usr/bin/env python3
"""
E5.2: canonical_results_v2.csv CI列口径统一脚本 (T5.7)
================================================================================
时间: 2026-10-09
背景:
  v2 CSV 由 task_e1_freeze_and_audit.py 生成，CI列直接取 results.json 的
  test_ci 字段。但该字段口径混杂: 旧文件为 kendalltau 近似口径(如E0=0.5514),
  新文件为 pair-counting 口径。论文全部CI为 pair-counting (test_ci_standard
  字段, Round 11向量化重算)。仓库公开前必须统一，否则审稿人对照CSV与论文
  会发现矛盾 (如 E0 行 CI=0.5516 vs 论文 0.88)。
目标:
  将 canonical_results_v2.csv 全部数据行的 CI_Mean/CI_Std 列重算为
  pair-counting 口径: 逐成员文件取 test_ci_standard (缺失时回退 test_ci)。
规则:
  - 行→成员目录映射: results/controlled/{name}_seed{S}, S=标准五种子
    {42,3407,2024,7,123} (Num_Seeds=5的行); 成员数必须等于Num_Seeds
  - KIBA n=5行 (GIN_KIBA_alpha*/KIBA_E0等): 仅取原5种子(新5种子属于_n10行)
  - _n10行: 已由e1_3用pair-counting口径写入(与e1_2重算JSON一致), 直接跳过
  - GIN_KIBA_alpha0_POLLUTED_EXCLUDE: 跳过(已排除数据, 不改动)
  - 无法完整映射的行: 不修改, 列入 flagged 报告人工核查
  - 修改前备份 (已存在备份则保留原始版)
输入: results/analysis/canonical_results_v2.csv + results/controlled/*/results.json
输出: 原地更新CSV + results/analysis/e5_2_ci_fix_report.json
================================================================================
"""

import csv
import json
import numpy as np
from pathlib import Path

CSV = Path('/home/rxp/dta2/results/analysis/canonical_results_v2.csv')
BAK = CSV.with_suffix('.csv.bak_ci_20261009')
OUT = Path('/home/rxp/dta2/results/analysis/e5_2_ci_fix_report.json')
CONTROLLED = Path('/home/rxp/dta2/results/controlled')
STD_SEEDS = [42, 3407, 2024, 7, 123]

SKIP_ROWS = {'GIN_KIBA_alpha0_POLLUTED_EXCLUDE'}


def ci_of(results_json):
    """pair-counting CI: test_ci_standard 优先，回退 test_ci"""
    with open(results_json) as f:
        r = json.load(f)
    ci = r.get('test_ci_standard')
    if ci is None:
        ci = r.get('test_ci')
    return ci


def main():
    with open(CSV, newline='') as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1:]

    report = {'script': 'scripts/e5_2_canonical_ci_fix.py', 'date': '2026-10-09',
              'updated': [], 'unchanged': [], 'flagged': [], 'skipped': []}

    if not BAK.exists():
        import shutil
        shutil.copy2(CSV, BAK)

    for r in data:
        name, n_seeds = r[0], int(r[1])

        if name in SKIP_ROWS:
            report['skipped'].append(dict(name=name, reason='已排除污染行'))
            continue

        # _n10行: e1_3已用pair-counting写入, 校验后跳过
        if name.endswith('_n10'):
            report['unchanged'].append(dict(name=name, reason='n10行已为新口径'))
            continue

        # 成员目录: results/controlled/{name}_seed{std5}
        member_dirs = [CONTROLLED / f'{name}_seed{s}' for s in STD_SEEDS]
        missing = [d.name for d in member_dirs if not (d / 'results.json').exists()]
        if missing:
            report['flagged'].append(dict(name=name, reason=f'缺成员目录: {missing}'))
            continue
        if n_seeds != len(STD_SEEDS):
            report['flagged'].append(dict(name=name,
                                          reason=f'Num_Seeds={n_seeds}≠5, 需人工确认种子集'))
            continue

        cis = [ci_of(d / 'results.json') for d in member_dirs]
        if any(c is None for c in cis):
            report['flagged'].append(dict(name=name, reason='成员缺CI字段'))
            continue
        new_mean, new_std = float(np.mean(cis)), float(np.std(cis, ddof=1))
        old_mean, old_std = r[5], r[6]
        change = abs(new_mean - float(old_mean))
        if change < 5e-7:
            report['unchanged'].append(dict(name=name, old=old_mean, new=f'{new_mean:.6f}'))
            continue
        r[5], r[6] = f'{new_mean:.6f}', f'{new_std:.6f}'
        report['updated'].append(dict(name=name, old_mean=old_mean,
                                      new_mean=f'{new_mean:.6f}',
                                      new_std=f'{new_std:.6f}'))

    with open(CSV, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(header)
        w.writerows(data)

    print('=' * 70)
    print('E5.2: canonical CI列口径统一 (pair-counting)')
    print('=' * 70)
    print(f'更新: {len(report["updated"])} 行')
    for u in report['updated']:
        print(f'  {u["name"]:<28} {u["old_mean"]} → {u["new_mean"]}±{u["new_std"]}')
    print(f'未变: {len(report["unchanged"])}  跳过: {len(report["skipped"])}  '
          f'需人工核查: {len(report["flagged"])}')
    for fl in report['flagged']:
        print(f'  ⚠ {fl["name"]}: {fl["reason"]}')

    with open(OUT, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f'报告: {OUT}')


if __name__ == '__main__':
    main()
