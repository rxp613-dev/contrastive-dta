#!/usr/bin/env python3
"""
E3.3: scaffold 冷启动划分 E0 vs E7(α=0.05) 统计分析脚本
================================================================================
时间: 2026-10-09
任务来源: 《实验复核意见与下一阶段实验方案.md》实验E3 (T3.4统计)
目标:
  对 e3_scaffold_E0_seed* 与 e3_scaffold_E7a05_seed* (各5种子,
  同种子配对) 做配对t检验 + 效应量(Hedges' g)，判定对比学习在
  真scaffold cold-drug 划分下是否保持标准划分上的收益。
  统计口径(与论文一致): 改善%=(E0-E7)/E0均值 (Convention B 正=改善),
  配对t (ddof=1), Hedges' g (配对差值口径, 正=CL有益)。
数据原则: 以 results.json config 为准做协议断言(同e3_2 verify逻辑);
  5种子必须齐全 {42,3407,2024,7,123}。
输入: results/cold_start/e3_scaffold_{E0,E7a05}_seed*/results.json
输出: results/analysis/e3_3_scaffold_stats.json + 终端报告
================================================================================
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

BASE = Path('/home/rxp/dta2/results/cold_start')
OUT = Path('/home/rxp/dta2/results/analysis/e3_3_scaffold_stats.json')
SEEDS = [42, 3407, 2024, 7, 123]


def load_group(tpl, expect_uc):
    rows = {}
    for seed in SEEDS:
        rj = BASE / tpl.format(seed) / 'results.json'
        assert rj.exists(), f'缺失: {rj}'
        with open(rj) as f:
            r = json.load(f)
        c = r['config']
        # 协议断言
        assert bool(c.get('use_contrastive')) is expect_uc, f'{rj}: uc异常'
        assert str(c.get('split_type')) == 'scaffold', f'{rj}: split_type异常'
        assert c.get('seed') == seed
        assert abs(float(c.get('learning_rate')) - 5e-4) < 1e-9
        if expect_uc:
            assert abs(float(c.get('alpha')) - 0.05) < 1e-9
            assert float(c.get('beta')) == 0.0 and float(c.get('gamma')) == 0.0
            assert str(c.get('augmentation')) == 'noise'
        else:
            assert str(c.get('augmentation')) == 'none'
        rows[seed] = dict(mse=r['test_mse'], ci=r['test_ci'],
                          epoch=r['best_epoch'])
    return rows


def hedges_g_paired(diff):
    """配对差值的Hedges' g (正=CL有益, 即E7<E0)"""
    n = len(diff)
    d = diff.mean() / diff.std(ddof=1)
    j = 1 - 3 / (4 * n - 5)   # 小样本校正
    return j * d


def main():
    print('=' * 70)
    print('E3.3: scaffold划分 E0 vs E7(α=0.05) 统计分析')
    print('=' * 70)
    e0 = load_group('e3_scaffold_E0_seed{}', False)
    e7 = load_group('e3_scaffold_E7a05_seed{}', True)

    m0 = np.array([e0[s]['mse'] for s in SEEDS])
    m7 = np.array([e7[s]['mse'] for s in SEEDS])
    print(f'{"seed":<8} {"E0 MSE":>9} {"E7 MSE":>9} {"E0 ep":>6} {"E7 ep":>6}')
    for s in SEEDS:
        print(f'{s:<8} {e0[s]["mse"]:>9.4f} {e7[s]["mse"]:>9.4f} '
              f'{e0[s]["epoch"]:>6} {e7[s]["epoch"]:>6}')

    diff = m0 - m7   # 正=E7优于E0 (CL有益)
    t, p = stats.ttest_rel(m7, m0)
    imp = diff.mean() / m0.mean() * 100
    g = hedges_g_paired(diff)
    ci0 = np.array([e0[s]['ci'] for s in SEEDS])
    ci7 = np.array([e7[s]['ci'] for s in SEEDS])

    print('\n' + '=' * 70)
    print(f'E0: MSE={m0.mean():.4f}±{m0.std(ddof=1):.4f}  CI={ci0.mean():.4f}')
    print(f'E7: MSE={m7.mean():.4f}±{m7.std(ddof=1):.4f}  CI={ci7.mean():.4f}')
    print(f'改善(Convention B): {imp:+.2f}%  (5/5种子方向: '
          f'{"全部CL有益" if (diff > 0).all() else "全部CL有害" if (diff < 0).all() else "不一致"})')
    print(f'配对t检验: t={t:.3f}, p={p:.4f}')
    print(f"Hedges' g (配对): {g:+.3f}")
    print('=' * 70)

    report = dict(
        script='scripts/e3_3_scaffold_stats.py', date='2026-10-09',
        n_seeds=len(SEEDS),
        e0_mse_mean=float(m0.mean()), e0_mse_std=float(m0.std(ddof=1)),
        e7_mse_mean=float(m7.mean()), e7_mse_std=float(m7.std(ddof=1)),
        e0_ci_mean=float(ci0.mean()), e7_ci_mean=float(ci7.mean()),
        improvement_pct=float(imp), paired_t=float(t), paired_p=float(p),
        hedges_g=float(g),
        direction_consistency=int((diff > 0).sum()),
        per_seed={str(s): dict(e0_mse=float(e0[s]['mse']),
                               e7_mse=float(e7[s]['mse']),
                               e0_epoch=int(e0[s]['epoch']),
                               e7_epoch=int(e7[s]['epoch']))
                  for s in SEEDS},
    )
    with open(OUT, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f'报告已保存: {OUT}')


if __name__ == '__main__':
    main()
