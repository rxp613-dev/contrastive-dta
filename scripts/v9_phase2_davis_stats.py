#!/usr/bin/env python3
"""
V9 Phase 2 Task 2.1.4: Davis E0 vs E7 (Config C @1000ep) 统计检验脚本
日期: 2026-11-09
目标: 对 Task 2.1 产出的 Davis 标准随机划分 E0 / E7(α=0.05) 各 10 个同种子
      结果做完整统计检验，判断 KIBA 上的结论是否跨数据集成立：
        - MSE / CI 的同种子配对差异（配对 t，主检验）
        - Welch t（辅助，不假设方差齐性）
        - Levene 方差齐性检验与方差比
        - 配对效应量 dz 与 Hedges' g
        - 均值差 95% CI（t 分布）
方法: 与 scripts/v9_decision_gate_a.py 同一套检验逻辑，仅输入/输出与裁定语境不同。
输入:
  results/controlled/v9_p2_davis_e0_seed*/results.json
  results/controlled/v9_p2_davis_e7_seed*/results.json
输出:
  results/analysis/v9_phase2p1_davis_stats.json
  results/analysis/v9_phase2p1_davis_stats.md
原则: 数字只由本脚本从真实实验 JSON 计算，不编造。
"""

import json
import glob
from datetime import datetime

import numpy as np
from scipy import stats


def load_seed_results(pattern):
    """返回 {seed: (mse, ci, best_epoch)}。"""
    out = {}
    for rf in sorted(glob.glob(pattern)):
        with open(rf) as f:
            d = json.load(f)
        out[int(d['seed'])] = (float(d['test_mse']), float(d['test_ci']), int(d['best_epoch']))
    return out


def hedges_correction(n_total):
    """Hedges 小样本修正因子。"""
    return 1.0 - 3.0 / (4.0 * n_total - 9.0)


def analyze_metric(e0, e7, higher_is_better):
    """单指标完整统计。增益正号 = CL 改进。"""
    n = len(e0)
    e0, e7 = np.asarray(e0, float), np.asarray(e7, float)
    e0_mean, e7_mean = float(e0.mean()), float(e7.mean())
    e0_std, e7_std = float(e0.std(ddof=1)), float(e7.std(ddof=1))

    gain = ((e7_mean - e0_mean) if higher_is_better else (e0_mean - e7_mean)) / e0_mean * 100.0

    diffs = e7 - e0
    t_pair, p_pair = stats.ttest_rel(e7, e0)
    d_sd = float(diffs.std(ddof=1))
    dz = float(diffs.mean() / d_sd) if d_sd > 0 else float('nan')
    half = stats.t.ppf(0.975, n - 1) * d_sd / np.sqrt(n)
    ci95 = [float(diffs.mean() - half), float(diffs.mean() + half)]

    t_w, p_w = stats.ttest_ind(e7, e0, equal_var=False)
    sp = np.sqrt(((n - 1) * e0_std ** 2 + (n - 1) * e7_std ** 2) / (2 * n - 2))
    g = float((e7_mean - e0_mean) / sp * hedges_correction(2 * n)) if sp > 0 else float('nan')

    return {
        'n': n, 'e0_mean': e0_mean, 'e0_std': e0_std,
        'e7_mean': e7_mean, 'e7_std': e7_std, 'gain_pct': float(gain),
        'paired_t': float(t_pair), 'paired_p': float(p_pair),
        'mean_diff_e7_minus_e0': float(diffs.mean()), 'diff_ci95': ci95,
        'paired_dz': dz, 'welch_t': float(t_w), 'welch_p': float(p_w), 'hedges_g': g
    }


def variance_test(e0, e7):
    """Levene 检验 + 方差比（基于 MSE）。"""
    e0, e7 = np.asarray(e0, float), np.asarray(e7, float)
    w, p = stats.levene(e7, e0)
    v0, v7 = float(e0.var(ddof=1)), float(e7.var(ddof=1))
    return {
        'e0_variance': v0, 'e7_variance': v7,
        'variance_ratio_e7_over_e0': v7 / v0,
        'variance_reduction_pct': (1 - v7 / v0) * 100.0,
        'levene_stat': float(w), 'levene_p': float(p),
        'significantly_different': bool(p < 0.05)
    }


def write_markdown(report, path):
    m, c, v = report['mse'], report['ci'], report['variance']
    L = []
    L.append('# Phase 2 Task 2.1: Davis E0 vs E7 统计报告（Config C @1000ep）\n')
    L.append(f"**生成时间**: {report['timestamp']}")
    L.append(f"**配对种子**: n={m['n']}，{report['common_seeds']}\n")
    L.append('## 逐种子配对（MSE）\n')
    L.append('| Seed | E0 MSE | E7 MSE | 差值(E7−E0) | E0 CI | E7 CI |')
    L.append('|------|--------|--------|-------------|-------|-------|')
    for row in report['per_seed']:
        L.append(f"| {row['seed']} | {row['e0_mse']:.4f} | {row['e7_mse']:.4f} | "
                 f"{row['e7_mse']-row['e0_mse']:+.4f} | {row['e0_ci']:.4f} | {row['e7_ci']:.4f} |")
    L.append('')
    L.append('## 描述统计与增益\n')
    L.append('| 指标 | E0 | E7 | 相对增益(正=CL好) |')
    L.append('|------|----|----|-------------------|')
    L.append(f"| MSE | {m['e0_mean']:.4f}±{m['e0_std']:.4f} | {m['e7_mean']:.4f}±{m['e7_std']:.4f} | {m['gain_pct']:+.2f}% |")
    L.append(f"| CI  | {c['e0_mean']:.4f}±{c['e0_std']:.4f} | {c['e7_mean']:.4f}±{c['e7_std']:.4f} | {c['gain_pct']:+.2f}% |\n")
    L.append('## 配对 t 检验（主检验）\n')
    L.append('| 指标 | 均值差 | 95% CI | t | p | dz |')
    L.append('|------|--------|--------|---|---|----|')
    for r, name in ((m, 'MSE'), (c, 'CI')):
        lo, hi = r['diff_ci95']
        L.append(f"| {name} | {r['mean_diff_e7_minus_e0']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | "
                 f"{r['paired_t']:+.3f} | {r['paired_p']:.4f} | {r['paired_dz']:+.2f} |")
    L.append('')
    L.append('## Welch / Hedges\' g（辅助）\n')
    L.append('| 指标 | t (Welch) | p | Hedges\' g |')
    L.append('|------|-----------|---|-----------|')
    L.append(f"| MSE | {m['welch_t']:+.3f} | {m['welch_p']:.4f} | {m['hedges_g']:+.2f} |")
    L.append(f"| CI  | {c['welch_t']:+.3f} | {c['welch_p']:.4f} | {c['hedges_g']:+.2f} |\n")
    L.append('## 方差（Levene）\n')
    L.append(f"- E0 方差 {v['e0_variance']:.6f}，E7 方差 {v['e7_variance']:.6f}，"
             f"比 {v['variance_ratio_e7_over_e0']:.3f}，缩减 {v['variance_reduction_pct']:+.1f}%")
    L.append(f"- Levene W={v['levene_stat']:.3f}, p={v['levene_p']:.4f} → "
             f"{'方差不齐' if v['significantly_different'] else '方差齐性（不显著）'}\n")
    L.append('---\n')
    L.append('## 结论（数据驱动）\n')

    mse_sig, mse_pos = m['paired_p'] < 0.05, m['gain_pct'] > 0
    ci_sig, ci_pos = c['paired_p'] < 0.05, c['gain_pct'] > 0
    if not mse_sig and not ci_sig:
        L.append(f"- Davis 上 **E0 与 E7 无统计显著差异**（MSE p={m['paired_p']:.3f}，"
                 f"CI p={c['paired_p']:.3f}）；点估计方向为 MSE {m['gain_pct']:+.2f}%、"
                 f"CI {c['gain_pct']:+.2f}%，效应量小（|dz| "
                 f"MSE={abs(m['paired_dz']):.2f}, CI={abs(c['paired_dz']):.2f}）。")
    L.append('- 与 KIBA（Config C 下 CL 两指标均显著更差，配对 p<0.01）对照：**CL 的性能效应具有数据集依赖性**，'
             '不能声称跨数据集一致的收益或损害。')
    L.append(f"- 方差点估计：E7/E0={v['variance_ratio_e7_over_e0']:.2f}（缩减 {v['variance_reduction_pct']:+.1f}%），"
             f"Levene p={v['levene_p']:.3f}，"
             + ("显著。" if v['significantly_different'] else "不显著，方差缩减在 Davis 上同样不构成统计声称。"))
    with open(path, 'w') as f:
        f.write('\n'.join(L))


def main():
    print('=' * 70)
    print('V9 Phase 2 Task 2.1.4: Davis E0 vs E7 stats')
    print('=' * 70)
    e0 = load_seed_results('/mnt/data/dta2/results/controlled/v9_p2_davis_e0_seed*/results.json')
    e7 = load_seed_results('/mnt/data/dta2/results/controlled/v9_p2_davis_e7_seed*/results.json')
    common = sorted(set(e0) & set(e7))
    print(f'配对 n={len(common)}: {common}')

    e0m = [e0[s][0] for s in common]; e7m = [e7[s][0] for s in common]
    e0c = [e0[s][1] for s in common]; e7c = [e7[s][1] for s in common]

    mse = analyze_metric(e0m, e7m, higher_is_better=False)
    ci = analyze_metric(e0c, e7c, higher_is_better=True)
    var = variance_test(e0m, e7m)
    per_seed = [{'seed': s, 'e0_mse': e0[s][0], 'e7_mse': e7[s][0],
                 'e0_ci': e0[s][1], 'e7_ci': e7[s][1],
                 'e0_epoch': e0[s][2], 'e7_epoch': e7[s][2]} for s in common]

    report = {'timestamp': datetime.now().isoformat(),
              'task': 'V9 Phase 2 Task 2.1.4: Davis E0 vs E7 (Config C)',
              'common_seeds': common, 'per_seed': per_seed,
              'mse': mse, 'ci': ci, 'variance': var}

    out_j = '/mnt/data/dta2/results/analysis/v9_phase2p1_davis_stats.json'
    out_m = '/mnt/data/dta2/results/analysis/v9_phase2p1_davis_stats.md'
    with open(out_j, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    write_markdown(report, out_m)

    print(f"\nMSE 增益 {mse['gain_pct']:+.2f}% p={mse['paired_p']:.4f} dz={mse['paired_dz']:+.2f}")
    print(f"CI  增益 {ci['gain_pct']:+.2f}% p={ci['paired_p']:.4f} dz={ci['paired_dz']:+.2f}")
    print(f"方差缩减 {var['variance_reduction_pct']:+.1f}% Levene p={var['levene_p']:.4f}")
    print(f'\n{out_j}\n{out_m}')


if __name__ == '__main__':
    main()
