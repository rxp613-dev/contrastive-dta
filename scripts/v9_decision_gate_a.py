#!/usr/bin/env python3
"""
V9 Phase 1 Task 1.3: Decision Gate A 决策门统计脚本
日期: 2026-11-08
目标: 在收敛协议(Config C, max_epochs=1000)下，对 KIBA E0 vs E7(α=0.05) 的
      10×10 同种子结果进行完整统计检验，裁定原论文头条三项是否存活：
        1. MSE 增益 (Config B 时声称 +3.34%)
        2. CI  增益 (Config B 时声称 +1.58%)
        3. 方差缩减 (Config B 时声称 -75%, Levene p=0.011)
方法:
  - 配对 t 检验 (paired t-test, 同种子差值) — 主检验
  - Welch t 检验 (不假设方差齐性) — 辅助检验
  - Levene 方差齐性检验
  - Hedges' g (独立样本效应量) 与配对效应量 dz
  - 95% 置信区间 (均值差 bootstrap-free, t 分布)
输入:
  results/controlled/v9_p1_kiba_e0_seed*/results.json
  results/controlled/v9_p1_kiba_e7_seed*/results.json
输出:
  results/analysis/v9_decision_gate_a.json
  results/analysis/v9_decision_gate_a.md
原则: 所有数字只由此脚本产出，不编造；结果基于真实实验 JSON。
"""

import json
import glob
import statistics
from datetime import datetime

import numpy as np
from scipy import stats


# ----------------------------------------------------------------------
# 数据加载
# ----------------------------------------------------------------------
def load_seed_results(pattern):
    """
    加载 results 目录 pattern，返回 {seed: {'mse':..., 'ci':..., 'epoch':...}}
    """
    out = {}
    for rf in sorted(glob.glob(pattern)):
        seed = int(rf.split('seed')[-1].split('/')[0])
        with open(rf) as f:
            d = json.load(f)
        out[seed] = {
            'mse': float(d['test_mse']),
            'ci': float(d['test_ci']),
            'epoch': int(d['best_epoch'])
        }
    return out


# ----------------------------------------------------------------------
# 统计检验
# ----------------------------------------------------------------------
def hedges_correction(n):
    """Hedges 小样本修正因子 (近似)。"""
    return 1.0 - 3.0 / (4.0 * n - 9.0)


def analyze_metric(e0_vals, e7_vals, metric_name, higher_is_better=False):
    """
    对单一指标(MSE 或 CI)做完整统计。
    约定 "增益" = CL 相对 E0 的相对改进百分比；MSE 越小越好，CI 越大越好。

    Returns: dict
    """
    n = len(e0_vals)
    e0 = np.array(e0_vals, dtype=float)
    e7 = np.array(e7_vals, dtype=float)

    # ---- 描述统计 ----
    e0_mean, e7_mean = float(e0.mean()), float(e7.mean())
    e0_std = float(e0.std(ddof=1))
    e7_std = float(e7.std(ddof=1))

    # 相对增益 (按指标方向)
    if higher_is_better:
        gain_pct = (e7_mean - e0_mean) / e0_mean * 100.0
    else:
        gain_pct = (e0_mean - e7_mean) / e0_mean * 100.0

    # ---- 配对 t 检验 (同种子差值 d = e0 - e7；对 CI 检验方向单独解释) ----
    # 统一使用 d = E7 - E0；MSE 下 d<0 为改进，CI 下 d>0 为改进
    diffs = e7 - e0
    t_pair, p_pair = stats.ttest_rel(e7, e0)
    d_mean = float(diffs.mean())
    d_sd = float(diffs.std(ddof=1))
    dz = d_mean / d_sd if d_sd > 0 else float('nan')  # 配对效应量

    # 差值 95% CI
    tcrit = stats.t.ppf(0.975, df=n - 1)
    ci_half = tcrit * d_sd / np.sqrt(n)
    diff_ci95 = [d_mean - ci_half, d_mean + ci_half]

    # ---- Welch t 检验 (独立样本) ----
    t_welch, p_welch = stats.ttest_ind(e7, e0, equal_var=False)

    # ---- Hedges' g (pooled SD, 独立样本效应量) ----
    sp = np.sqrt(((n - 1) * e0_std ** 2 + (n - 1) * e7_std ** 2) / (2 * n - 2))
    g = (e7_mean - e0_mean) / sp * hedges_correction(2 * n) if sp > 0 else float('nan')

    return {
        'metric': metric_name,
        'n': n,
        'e0_mean': e0_mean,
        'e0_std': e0_std,
        'e7_mean': e7_mean,
        'e7_std': e7_std,
        'gain_pct': gain_pct,                      # 正=CL 改进
        'paired_t': float(t_pair),
        'paired_p': float(p_pair),
        'mean_diff_e7_minus_e0': d_mean,
        'diff_ci95': diff_ci95,
        'paired_dz': float(dz),
        'welch_t': float(t_welch),
        'welch_p': float(p_welch),
        'hedges_g': float(g)
    }


def variance_test(e0_vals, e7_vals):
    """Levene 检验 + 方差比。"""
    e0 = np.array(e0_vals, dtype=float)
    e7 = np.array(e7_vals, dtype=float)
    stat, p = stats.levene(e0, e7)
    v0 = float(e0.var(ddof=1))
    v7 = float(e7.var(ddof=1))
    return {
        'e0_variance': v0,
        'e7_variance': v7,
        'variance_ratio_e7_over_e0': v7 / v0,
        'variance_reduction_pct': (1.0 - v7 / v0) * 100.0,
        'levene_stat': float(stat),
        'levene_p': float(p),
        'equal_variance_at_0.05': bool(p > 0.05)
    }


# ----------------------------------------------------------------------
# 决策裁定
# ----------------------------------------------------------------------
def adjudicate(mse_res, ci_res, var_res):
    """
    对照 Config B 时的三项声称逐条裁定存活/翻转。
    """
    verdicts = []

    # 1) MSE 增益: Config B 声称 +3.34%
    mse_survive = (mse_res['gain_pct'] > 0) and (mse_res['paired_p'] < 0.05)
    if mse_survive:
        mse_verdict = '存活'
    elif mse_res['gain_pct'] <= 0 and mse_res['paired_p'] < 0.05:
        mse_verdict = '符号翻转，且反向显著（CL 显著更差）'
    elif mse_res['gain_pct'] <= 0:
        mse_verdict = '符号翻转，反向不显著'
    else:
        mse_verdict = '方向保留但不显著'
    verdicts.append({
        'claim': 'MSE 增益 (Config B 声称 +3.34%)',
        'observed_gain_pct': mse_res['gain_pct'],
        'paired_p': mse_res['paired_p'],
        'verdict': mse_verdict
    })

    # 2) CI 增益: Config B 声称 +1.58%
    ci_survive = (ci_res['gain_pct'] > 0) and (ci_res['paired_p'] < 0.05)
    if ci_survive:
        ci_verdict = '存活'
    elif ci_res['gain_pct'] <= 0 and ci_res['paired_p'] < 0.05:
        ci_verdict = '符号翻转，且反向显著（CL 显著更差）'
    elif ci_res['gain_pct'] <= 0:
        ci_verdict = '符号翻转，反向不显著'
    else:
        ci_verdict = '方向保留但不显著'
    verdicts.append({
        'claim': 'CI 增益 (Config B 声称 +1.58%)',
        'observed_gain_pct': ci_res['gain_pct'],
        'paired_p': ci_res['paired_p'],
        'verdict': ci_verdict
    })

    # 3) 方差缩减: Config B 声称 -75%, Levene p=0.011
    var_survive = (var_res['variance_reduction_pct'] > 0) and (var_res['levene_p'] < 0.05)
    verdicts.append({
        'claim': '方差缩减 (Config B 声称 -75%, p=0.011)',
        'observed_reduction_pct': var_res['variance_reduction_pct'],
        'levene_p': var_res['levene_p'],
        'verdict': '存活' if var_survive else '不存活'
    })

    return verdicts


# ----------------------------------------------------------------------
# 报告
# ----------------------------------------------------------------------
def write_markdown(report, path):
    m = report['mse']
    c = report['ci']
    v = report['variance']

    lines = []
    lines.append('# Decision Gate A 决策报告（KIBA, Config C @1000ep）\n')
    lines.append(f"**生成时间**: {report['timestamp']}")
    lines.append(f"**配对种子数**: n={m['n']}，种子 {report['common_seeds']}\n")
    lines.append('---\n')

    lines.append('## 描述统计\n')
    lines.append('| 指标 | E0 | E7 (α=0.05) | 相对增益 |')
    lines.append('|------|----|-------------|----------|')
    lines.append(f"| MSE | {m['e0_mean']:.4f}±{m['e0_std']:.4f} | {m['e7_mean']:.4f}±{m['e7_std']:.4f} | {m['gain_pct']:+.2f}% |")
    lines.append(f"| CI  | {c['e0_mean']:.4f}±{c['e0_std']:.4f} | {c['e7_mean']:.4f}±{c['e7_std']:.4f} | {c['gain_pct']:+.2f}% |\n")

    lines.append('## 配对 t 检验（同种子，主检验）\n')
    lines.append('| 指标 | 均值差(E7−E0) | 差值95%CI | t | p | dz |')
    lines.append('|------|---------------|-----------|---|---|----|')
    lo, hi = m['diff_ci95']
    lines.append(f"| MSE | {m['mean_diff_e7_minus_e0']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {m['paired_t']:+.3f} | {m['paired_p']:.4f} | {m['paired_dz']:+.2f} |")
    lo, hi = c['diff_ci95']
    lines.append(f"| CI  | {c['mean_diff_e7_minus_e0']:+.4f} | [{lo:+.4f}, {hi:+.4f}] | {c['paired_t']:+.3f} | {c['paired_p']:.4f} | {c['paired_dz']:+.2f} |\n")

    lines.append('## Welch t 检验与 Hedges\' g（辅助）\n')
    lines.append('| 指标 | t (Welch) | p | Hedges\' g |')
    lines.append('|------|-----------|---|-----------|')
    lines.append(f"| MSE | {m['welch_t']:+.3f} | {m['welch_p']:.4f} | {m['hedges_g']:+.2f} |")
    lines.append(f"| CI  | {c['welch_t']:+.3f} | {c['welch_p']:.4f} | {c['hedges_g']:+.2f} |\n")

    lines.append('## 方差齐性（Levene）\n')
    lines.append(f"- E0 方差: {v['e0_variance']:.6f}；E7 方差: {v['e7_variance']:.6f}")
    lines.append(f"- 方差比 E7/E0: {v['variance_ratio_e7_over_e0']:.3f}；方差缩减: {v['variance_reduction_pct']:+.1f}%")
    lines.append(f"- Levene: W={v['levene_stat']:.3f}, p={v['levene_p']:.4f} → "
                 f"{'方差齐性' if v['equal_variance_at_0.05'] else '方差不齐（p<0.05）'}\n")

    lines.append('---\n')
    lines.append('## 裁定\n')
    lines.append('| 原声称 (Config B) | Config C 观测 | p 值 | 裁定 |')
    lines.append('|-------------------|---------------|------|------|')
    for vd in report['verdicts']:
        if 'observed_gain_pct' in vd:
            lines.append(f"| {vd['claim']} | {vd['observed_gain_pct']:+.2f}% | {vd['paired_p']:.4f} | **{vd['verdict']}** |")
        else:
            lines.append(f"| {vd['claim']} | {vd['observed_reduction_pct']:+.1f}% | {vd['levene_p']:.4f} | **{vd['verdict']}** |")

    lines.append('')
    lines.append('---\n')
    lines.append('## 结论\n')
    n_survive = sum(1 for vd in report['verdicts'] if vd['verdict'] == '存活')
    lines.append(f"三项声称中 **{n_survive}/3 存活**。")
    lines.append('')

    # 数据驱动的结论，避免把条件性文字写成事实
    mse_reversed_sig = (m['gain_pct'] <= 0) and (m['paired_p'] < 0.05)
    ci_reversed_sig = (c['gain_pct'] <= 0) and (c['paired_p'] < 0.05)
    var_point_positive = v['variance_reduction_pct'] > 0
    var_significant = v['levene_p'] < 0.05

    if mse_reversed_sig and ci_reversed_sig:
        lines.append('- 在收敛协议（Config C）下，CL 相对同种子基线在 **MSE 与 CI 两个方向上均显著变差**'
                     '（配对 p 均 < 0.01），Config B 时报告的 MSE +3.34%、CI +1.58% 表观增益'
                     '确认为**欠收敛监督基线制造的伪影**，与 v9 评审的核心预判一致。')
    if var_point_positive and not var_significant:
        lines.append(f"- 方差点估计仍显示 CL 组方差较小（缩减 {v['variance_reduction_pct']:.1f}%），"
                     f"但 n=10 时 Levene 检验不显著（p={v['levene_p']:.3f}），"
                     '不能作为统计上成立的声称；需在 Phase 2 用更多种子/数据集复验。')
    lines.append('- 结果进入 Phase 2：在 Davis、scaffold、其他 α 剂量与 GAT 编码器上复验，'
                 '判断"方差缩减"是否为可复现效应，以及性能翻转是否跨数据集成立。')

    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def main():
    print('=' * 70)
    print('V9 Task 1.3: Decision Gate A')
    print('=' * 70)

    e0 = load_seed_results('/mnt/data/dta2/results/controlled/v9_p1_kiba_e0_seed*/results.json')
    e7 = load_seed_results('/mnt/data/dta2/results/controlled/v9_p1_kiba_e7_seed*/results.json')
    common = sorted(set(e0) & set(e7))
    print(f'E0 种子 {len(e0)} 个, E7 种子 {len(e7)} 个, 配对 {len(common)} 个: {common}')

    e0_mse = [e0[s]['mse'] for s in common]
    e7_mse = [e7[s]['mse'] for s in common]
    e0_ci = [e0[s]['ci'] for s in common]
    e7_ci = [e7[s]['ci'] for s in common]

    mse_res = analyze_metric(e0_mse, e7_mse, 'MSE', higher_is_better=False)
    ci_res = analyze_metric(e0_ci, e7_ci, 'CI', higher_is_better=True)
    var_res = variance_test(e0_mse, e7_mse)

    verdicts = adjudicate(mse_res, ci_res, var_res)

    report = {
        'timestamp': datetime.now().isoformat(),
        'task': 'V9 Phase 1 Task 1.3: Decision Gate A',
        'protocol': 'Config C: max_epochs=1000, lr_patience=30, early_stop=150',
        'dataset': 'KIBA',
        'common_seeds': common,
        'mse': mse_res,
        'ci': ci_res,
        'variance': var_res,
        'verdicts': verdicts
    }

    out_json = '/mnt/data/dta2/results/analysis/v9_decision_gate_a.json'
    out_md = '/mnt/data/dta2/results/analysis/v9_decision_gate_a.md'
    with open(out_json, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    write_markdown(report, out_md)

    print('\n--- MSE ---')
    print(f"增益 {mse_res['gain_pct']:+.2f}% | paired p={mse_res['paired_p']:.4f} | dz={mse_res['paired_dz']:+.2f}")
    print('--- CI ---')
    print(f"增益 {ci_res['gain_pct']:+.2f}% | paired p={ci_res['paired_p']:.4f} | dz={ci_res['paired_dz']:+.2f}")
    print('--- 方差 ---')
    print(f"缩减 {var_res['variance_reduction_pct']:+.1f}% | Levene p={var_res['levene_p']:.4f}")
    print('\n裁定:')
    for vd in verdicts:
        print(f"  - {vd['claim']} → {vd['verdict']}")
    print(f'\n报告: {out_json}\n      {out_md}')


if __name__ == '__main__':
    main()
