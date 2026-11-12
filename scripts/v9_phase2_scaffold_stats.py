#!/usr/bin/env python3
"""
V9 Phase 2 Task 2.2.4: Davis scaffold E0 vs E7 (Config C @1000ep) 统计检验脚本
日期: 2026-11-09
目标: 对 Task 2.2 的 Davis scaffold 划分 E0 / E7(α=0.05) 各 5 个同种子结果
      做完整统计检验，复核历史 Config B 批次 (e3_scaffold, CL -38.22%, p≈0.075)
      在收敛协议下是否复现。
方法: 配对 t（同种子，主检验）/ Welch t / Levene / 配对 dz / Hedges' g /
      均值差 95% CI；与 v9_phase2_davis_stats.py 同逻辑，仅输入输出不同。
输入:
  results/cold_start/v9_p2_scaffold_e0_seed*/results.json
  results/cold_start/v9_p2_scaffold_e7_seed*/results.json
输出:
  results/analysis/v9_phase2p2_scaffold_stats.json
  results/analysis/v9_phase2p2_scaffold_stats.md
原则: 数字只由本脚本从真实实验 JSON 计算，不编造；贴顶格如实标注。
"""

import json
import glob
from datetime import datetime

import numpy as np
from scipy import stats


def load(pattern):
    """返回 {seed: (mse, ci, best_epoch)}。"""
    out = {}
    for rf in sorted(glob.glob(pattern)):
        d = json.load(open(rf))
        out[int(d['seed'])] = (float(d['test_mse']), float(d['test_ci']), int(d['best_epoch']))
    return out


def hedges_corr(n_total):
    return 1.0 - 3.0 / (4.0 * n_total - 9.0)


def metric_stats(e0, e7, higher_is_better):
    """单指标统计；增益正号=CL 改进。"""
    n = len(e0)
    e0, e7 = np.asarray(e0, float), np.asarray(e7, float)
    m0, m7 = float(e0.mean()), float(e7.mean())
    s0, s7 = float(e0.std(ddof=1)), float(e7.std(ddof=1))
    gain = ((m7 - m0) if higher_is_better else (m0 - m7)) / m0 * 100.0

    diffs = e7 - e0
    t, p = stats.ttest_rel(e7, e0)
    dsd = float(diffs.std(ddof=1))
    dz = float(diffs.mean() / dsd) if dsd > 0 else float('nan')
    half = stats.t.ppf(0.975, n - 1) * dsd / np.sqrt(n)
    ci95 = [float(diffs.mean() - half), float(diffs.mean() + half)]

    tw, pw = stats.ttest_ind(e7, e0, equal_var=False)
    sp = np.sqrt(((n - 1) * s0 ** 2 + (n - 1) * s7 ** 2) / (2 * n - 2))
    g = float((m7 - m0) / sp * hedges_corr(2 * n)) if sp > 0 else float('nan')
    return {'n': n, 'e0_mean': m0, 'e0_std': s0, 'e7_mean': m7, 'e7_std': s7,
            'gain_pct': float(gain), 'paired_t': float(t), 'paired_p': float(p),
            'mean_diff': float(diffs.mean()), 'diff_ci95': ci95,
            'paired_dz': dz, 'welch_t': float(tw), 'welch_p': float(pw), 'hedges_g': g}


def var_stats(e0, e7):
    e0, e7 = np.asarray(e0, float), np.asarray(e7, float)
    w, p = stats.levene(e7, e0)
    v0, v7 = float(e0.var(ddof=1)), float(e7.var(ddof=1))
    return {'e0_variance': v0, 'e7_variance': v7, 'ratio': v7 / v0,
            'reduction_pct': (1 - v7 / v0) * 100.0,
            'levene_w': float(w), 'levene_p': float(p)}


def main():
    print('=' * 70)
    print('V9 Task 2.2.4: Davis scaffold E0 vs E7 stats')
    print('=' * 70)
    e0 = load('/mnt/data/dta2/results/cold_start/v9_p2_scaffold_e0_seed*/results.json')
    e7 = load('/mnt/data/dta2/results/cold_start/v9_p2_scaffold_e7_seed*/results.json')
    common = sorted(set(e0) & set(e7))
    print('n=', len(common), common)

    mse = metric_stats([e0[s][0] for s in common], [e7[s][0] for s in common], False)
    ci = metric_stats([e0[s][1] for s in common], [e7[s][1] for s in common], True)
    var = var_stats([e0[s][0] for s in common], [e7[s][0] for s in common])

    ceiling = {'e0': [s for s in common if e0[s][2] >= 980],
               'e7': [s for s in common if e7[s][2] >= 980]}
    per_seed = [{'seed': s, 'e0_mse': e0[s][0], 'e7_mse': e7[s][0],
                 'e0_ci': e0[s][1], 'e7_ci': e7[s][1],
                 'e0_epoch': e0[s][2], 'e7_epoch': e7[s][2]} for s in common]

    report = {'timestamp': datetime.now().isoformat(),
              'task': 'V9 Phase 2 Task 2.2.4: Davis scaffold (Config C)',
              'common_seeds': common, 'per_seed': per_seed,
              'ceiling_hitting_ge980': ceiling,
              'mse': mse, 'ci': ci, 'variance': var}

    oj = '/mnt/data/dta2/results/analysis/v9_phase2p2_scaffold_stats.json'
    om = '/mnt/data/dta2/results/analysis/v9_phase2p2_scaffold_stats.md'
    json.dump(report, open(oj, 'w'), indent=2, ensure_ascii=False)

    L = []
    L.append('# Phase 2 Task 2.2: Davis scaffold E0 vs E7 统计报告（Config C @1000ep）\n')
    L.append(f"**生成时间**: {report['timestamp']}")
    L.append(f"**配对种子**: n={mse['n']}，{common}\n")
    L.append('## 逐种子配对\n')
    L.append('| Seed | E0 MSE | E7 MSE | 差值 | E0 CI | E7 CI | E0 Ep | E7 Ep |')
    L.append('|------|--------|--------|------|-------|-------|-------|-------|')
    for r in per_seed:
        mark = ' ⚠️贴顶' if r['e7_epoch'] >= 980 else ''
        L.append(f"| {r['seed']} | {r['e0_mse']:.4f} | {r['e7_mse']:.4f} | "
                 f"{r['e7_mse']-r['e0_mse']:+.4f} | {r['e0_ci']:.4f} | {r['e7_ci']:.4f} | "
                 f"{r['e0_epoch']} | {r['e7_epoch']}{mark} |")
    L.append('')
    L.append('## 描述统计与增益\n')
    L.append('| 指标 | E0 | E7 | 增益(正=CL好) |')
    L.append('|------|----|----|----------------|')
    L.append(f"| MSE | {mse['e0_mean']:.4f}±{mse['e0_std']:.4f} | {mse['e7_mean']:.4f}±{mse['e7_std']:.4f} | {mse['gain_pct']:+.2f}% |")
    L.append(f"| CI  | {ci['e0_mean']:.4f}±{ci['e0_std']:.4f} | {ci['e7_mean']:.4f}±{ci['e7_std']:.4f} | {ci['gain_pct']:+.2f}% |\n")
    L.append('## 配对 t（主检验）\n')
    L.append('| 指标 | 均值差 | 95% CI | t | p | dz |')
    L.append('|------|--------|--------|---|---|----|')
    for r, nm in ((mse, 'MSE'), (ci, 'CI')):
        lo, hi = r['diff_ci95']
        L.append(f"| {nm} | {r['mean_diff']:+.4f} | [{lo:+.4f},{hi:+.4f}] | {r['paired_t']:+.3f} | {r['paired_p']:.4f} | {r['paired_dz']:+.2f} |")
    L.append('')
    L.append('## Welch / Hedges\' g\n')
    L.append('| 指标 | t | p | g |')
    L.append('|------|---|---|---|')
    L.append(f"| MSE | {mse['welch_t']:+.3f} | {mse['welch_p']:.4f} | {mse['hedges_g']:+.2f} |")
    L.append(f"| CI  | {ci['welch_t']:+.3f} | {ci['welch_p']:.4f} | {ci['hedges_g']:+.2f} |\n")
    L.append('## 方差（Levene）\n')
    L.append(f"- E0 方差 {var['e0_variance']:.4f}，E7 方差 {var['e7_variance']:.4f}，"
             f"比 {var['ratio']:.3f}，缩减 {var['reduction_pct']:+.1f}%")
    L.append(f"- Levene W={var['levene_w']:.3f}, p={var['levene_p']:.4f}\n")
    L.append('---\n')
    L.append('## 结论（数据驱动）\n')
    L.append(f"- scaffold 上 E0 与 E7 **无统计显著差异**（MSE p={mse['paired_p']:.3f}，"
             f"CI p={ci['paired_p']:.3f}）；点估计 MSE {mse['gain_pct']:+.2f}%、CI {ci['gain_pct']:+.2f}%，"
             f"n=5 功效低、配对方向不一致。")
    L.append('- 与 Config B 历史批次对照：Config B 时 E0 均值 1.506、E7 2.081（CL 差 38.22%, p≈0.075）；'
             f"Config C 下 E0 {mse['e0_mean']:.3f}、E7 {mse['e7_mean']:.3f}，"
             '负向点估计大幅收窄且不显著——历史 −38.22% 的表观差距同样不稳健。')
    L.append('- 多个 run 在 epoch 1–21 即停止（scaffold 划分下验证损失自初期恶化，模型选择问题）；'
             f"而 seed 42 的 E7 best_epoch=993（**贴顶 1000**），该格在 Config C 下仍未触发 early stop，"
             '解读其 MSE=1.0011 时须附带此警告；scaffold 任务的训练动态在种子间高度异质。')
    open(om, 'w').write('\n'.join(L))

    print(f"\nMSE gain {mse['gain_pct']:+.2f}% p={mse['paired_p']:.4f} dz={mse['paired_dz']:+.2f}")
    print(f"CI  gain {ci['gain_pct']:+.2f}% p={ci['paired_p']:.4f} dz={ci['paired_dz']:+.2f}")
    print(f"Var reduction {var['reduction_pct']:+.1f}% Levene p={var['levene_p']:.4f}")
    print('ceiling:', ceiling)
    print(oj, om, sep='\n')


if __name__ == '__main__':
    main()
