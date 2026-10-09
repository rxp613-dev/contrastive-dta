#!/usr/bin/env python3
"""
E1.4: KIBA n=10 验证集α选择分析脚本
================================================================================
时间: 2026-10-09
任务来源: 《实验复核意见与下一阶段实验方案.md》E1后续一致性修订
  (P0.2.4分析的n=10重算: 论文 sec:validation_selection_appendix 的KIBA行
   当前引用n=5旧值0.3421±0.1662，与n=10权威值不一致)
目标:
  在 KIBA n=10 权威扫描(β=0.0, γ=0.0, lr=5e-4, noise)上重做验证集α选择:
  - 每个种子: α_val = 使 best_val_mse 最小的 α (α∈{0.001,0.005,0.01,0.03,0.05,0.1})
  - 记录该种子在 α_val 的 test MSE
  - 对照: 该种子在测试集选择 α=0.1 (全局最优) 的 test MSE
  - 膨胀率(与论文既有口径一致): per-seed
      inflation_s = (test_s@α=0.1 - test_s@α_val) / test_s@α_val × 100%
    报告均值±标准差 (ddof=1)
数据装载原则: 以 results.json 的 config 为准 (同 e1_2 协议断言)
输入: results/controlled/GIN_KIBA_alpha*_seed*/results.json
输出: results/analysis/e1_4_kiba_val_selection_n10.json + 终端报告
================================================================================
"""

import json
import glob
import numpy as np
from pathlib import Path

RESULTS = Path('/home/rxp/dta2/results/controlled')
OUT = Path('/home/rxp/dta2/results/analysis/e1_4_kiba_val_selection_n10.json')
ALPHA_ORDER = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]
OLD_SEEDS = {42, 3407, 2024, 7, 123}
NEW_SEEDS = {251, 1496, 5934, 7350, 9889}
TEST_SELECTED_ALPHA = 0.1   # n=10测试集最优α (e1_2结果)


def protocol_ok(c):
    return (bool(c.get('use_contrastive')) is True
            and float(c.get('beta', -1)) == 0.0
            and float(c.get('gamma', -1)) == 0.0
            and abs(float(c.get('learning_rate', -1)) - 5e-4) < 1e-9
            and str(c.get('augmentation')) == 'noise')


def load():
    """{(alpha, seed): (best_val_mse, test_mse)}"""
    data = {}
    for f in sorted(glob.glob(str(RESULTS / 'GIN_KIBA_alpha*_seed*' / 'results.json'))):
        d = Path(f).parent.name
        with open(f) as fh:
            r = json.load(fh)
        c = r.get('config', {})
        if not protocol_ok(c):
            continue
        if r.get('best_val_mse') is None or r.get('test_mse') is None:
            continue
        data[(float(c['alpha']), c['seed'])] = (r['best_val_mse'], r['test_mse'])
    return data


def main():
    print('=' * 70)
    print('E1.4: KIBA n=10 验证集α选择分析')
    print('=' * 70)
    data = load()

    seeds = sorted({s for _, s in data})
    assert set(seeds) == OLD_SEEDS | NEW_SEEDS, f'种子集合异常: {seeds}'

    rows = []
    for seed in seeds:
        # 该种子全部α齐全才纳入
        if not all((a, seed) in data for a in ALPHA_ORDER):
            missing = [a for a in ALPHA_ORDER if (a, seed) not in data]
            print(f'⚠️  seed={seed} 缺α={missing}，排除')
            continue
        vals = {a: data[(a, seed)] for a in ALPHA_ORDER}
        a_val = min(vals, key=lambda a: vals[a][0])
        test_at_val = vals[a_val][1]
        test_at_test = vals[TEST_SELECTED_ALPHA][1]
        inflation = (test_at_test - test_at_val) / test_at_val * 100
        batch = '原' if seed in OLD_SEEDS else '新'
        rows.append(dict(seed=seed, batch=batch, alpha_val=a_val,
                         val_mse_at_val=vals[a_val][0],
                         test_at_val=test_at_val, test_at_test=test_at_test,
                         inflation_pct=inflation))
        print(f'  [{batch}] seed={seed:<6} α_val={a_val:<6} '
              f'test@α_val={test_at_val:.4f}  test@α=0.1={test_at_test:.4f}  '
              f'inflation={inflation:+.1f}%')

    assert len(rows) == 10, f'纳入种子数={len(rows)}≠10'

    alphas = np.array([r['alpha_val'] for r in rows])
    t_val = np.array([r['test_at_val'] for r in rows])
    t_test = np.array([r['test_at_test'] for r in rows])
    infl = np.array([r['inflation_pct'] for r in rows])

    from collections import Counter
    alpha_counts = Counter(alphas.tolist())
    report = dict(
        script='scripts/e1_4_kiba_val_selection_n10.py', date='2026-10-09',
        n_seeds=len(rows),
        alpha_val_mean=float(alphas.mean()),
        alpha_val_std=float(alphas.std(ddof=1)),
        alpha_val_distribution={str(k): v for k, v in sorted(alpha_counts.items())},
        test_mse_at_val_alpha_mean=float(t_val.mean()),
        test_mse_at_val_alpha_std=float(t_val.std(ddof=1)),
        test_mse_at_test_alpha_mean=float(t_test.mean()),
        test_mse_at_test_alpha_std=float(t_test.std(ddof=1)),
        inflation_pct_mean=float(infl.mean()),
        inflation_pct_std=float(infl.std(ddof=1)),
        per_seed=rows,
    )

    print('\n' + '=' * 70)
    print(f'验证集选择α: 均值={alphas.mean():.3f}±{alphas.std(ddof=1):.3f}  '
          f'分布={dict(alpha_counts)}')
    print(f'test@α_val:  {t_val.mean():.4f}±{t_val.std(ddof=1):.4f}')
    print(f'test@α=0.1:  {t_test.mean():.4f}±{t_test.std(ddof=1):.4f}')
    print(f'膨胀率: {infl.mean():+.1f}±{infl.std(ddof=1):.1f}% '
          f'(正=测试集选择优于验证集选择)')
    print('=' * 70)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f'报告已保存: {OUT}')


if __name__ == '__main__':
    main()
