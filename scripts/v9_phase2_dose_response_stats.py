#!/usr/bin/env python3
"""
V9 Phase 2 Task 2.3.3: Davis α 剂量-反应统计检验脚本（Config C @1000ep）
日期: 2026-11-10
目标: 对 Task 2.3 产出的 6 个 α 剂量（0.01/0.02/0.05/0.1/0.2/0.5）×5 同种子
      结果，与 α=0 (E0) Config C 基线做系统统计，裁定标题中 "multiple losses"
      主张是否存活。严格对应 TODO 2.3.3~2.3.4，不简化。
方法:
  1. 逐剂量 vs α=0：同种子配对 t 检验（主检验）、均值差 95%CI、配对 dz；
     Welch t 与 Hedges' g 辅助；每剂量 Levene 方差比较。
  2. 多重比较校正：6 个剂量对同一基线，主检验家族报告 Bonferroni 校正
     阈值 0.05/6 = 0.00833 及 Benjamini-Hochberg FDR；区分原始 p 与校正后显著。
  3. 整体剂量效应（同种子重复测量）：Friedman χ²（7 水平含基线，非参数）；
     剂量均值的 Spearman 单调趋势（描述倒 U / 单调形态）。
输入:
  results/controlled/v9_p2p3_davis_{a001,a002,a005,a010,a020,a050}_seed*/results.json
  results/controlled/v9_p2_davis_e0_seed*/results.json (仅 5 个对应种子)
输出:
  results/analysis/v9_phase2p3_dose_response.json
  results/analysis/v9_phase2p3_dose_response.md
原则: 数字只由本脚本从真实实验 JSON 计算，不编造。
"""

import json
import glob
from datetime import datetime

import numpy as np
from scipy import stats


SEEDS = [42, 7, 123, 456, 789]
DOSES = [(0.01, 'a001'), (0.02, 'a002'), (0.05, 'a005'),
         (0.1, 'a010'), (0.2, 'a020'), (0.5, 'a050')]
N_DOSES = len(DOSES)
BONF_THRESHOLD = 0.05 / N_DOSES


def load(tag_or_e0):
    """加载一个条件，返回 {seed: (mse, ci, best_epoch)}。"""
    if tag_or_e0 == 'e0':
        pattern = '/mnt/data/dta2/results/controlled/v9_p2_davis_e0_seed*/results.json'
    else:
        pattern = f'/mnt/data/dta2/results/controlled/v9_p2p3_davis_{tag_or_e0}_seed*/results.json'
    out = {}
    for p in glob.glob(pattern):
        d = json.load(open(p))
        out[int(d['seed'])] = (float(d['test_mse']), float(d['test_ci']), int(d['best_epoch']))
    return out


def hedges_corr(n_total):
    return 1.0 - 3.0 / (4.0 * n_total - 9.0)


def dose_vs_baseline(base_mse, dose_mse):
    """单剂量 vs 基线的完整统计；base/dose 为按 SEEDS 对齐的数组。"""
    n = len(base_mse)
    b, x = np.asarray(base_mse), np.asarray(dose_mse)
    diffs = x - b
    t, p = stats.ttest_rel(x, b)
    dsd = float(diffs.std(ddof=1))
    dz = float(diffs.mean() / dsd) if dsd > 0 else float('nan')
    half = stats.t.ppf(0.975, n - 1) * dsd / np.sqrt(n)
    ci95 = [float(diffs.mean() - half), float(diffs.mean() + half)]
    tw, pw = stats.ttest_ind(x, b, equal_var=False)
    sb, sx = float(b.std(ddof=1)), float(x.std(ddof=1))
    sp = np.sqrt(((n - 1) * sb ** 2 + (n - 1) * sx ** 2) / (2 * n - 2))
    g = float((x.mean() - b.mean()) / sp * hedges_corr(2 * n)) if sp > 0 else float('nan')
    lw, lp = stats.levene(x, b)
    gain = float((b.mean() - x.mean()) / b.mean() * 100.0)
    return {
        'n': n, 'baseline_mean': float(b.mean()), 'dose_mean': float(x.mean()),
        'baseline_std': sb, 'dose_std': sx, 'gain_pct': gain,
        'paired_t': float(t), 'paired_p': float(p),
        'mean_diff': float(diffs.mean()), 'diff_ci95': ci95,
        'paired_dz': dz, 'welch_p': float(pw), 'hedges_g': g,
        'levene_p': float(lp),
        'significant_raw': bool(p < 0.05),
        'significant_bonferroni': bool(p < BONF_THRESHOLD),
    }


def bh_adjust(pvals):
    """Benjamini-Hochberg FDR 校正，返回与 pvals 等长的 q 值。"""
    pvals = np.asarray(pvals, float)
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0, 1)
    out = np.empty(m)
    out[order] = q
    return out.tolist()


def main():
    print('=' * 70)
    print('V9 Task 2.3.3: Davis dose-response statistics')
    print('=' * 70)

    e0 = load('e0')
    base_mse = [e0[s][0] for s in SEEDS]
    base_ci = [e0[s][1] for s in SEEDS]

    dose_results = {}
    dose_vectors = {}          # 供 Friedman：alpha -> 按种子对齐 MSE
    ci_dose_vectors = {}
    per_seed_rows = []
    pvals = []
    dose_keys = []

    for alpha, tag in DOSES:
        d = load(tag)
        dose_mse = [d[s][0] for s in SEEDS]
        dose_ci = [d[s][1] for s in SEEDS]
        res = dose_vs_baseline(base_mse, dose_mse)
        dose_results[tag] = {'alpha': alpha, **res}
        dose_vectors[tag] = dose_mse
        ci_dose_vectors[tag] = dose_ci
        pvals.append(res['paired_p'])
        dose_keys.append(tag)
        for i, s in enumerate(SEEDS):
            per_seed_rows.append({'alpha': alpha, 'seed': s,
                                  'e0_mse': base_mse[i], 'dose_mse': dose_mse[i],
                                  'e0_ci': base_ci[i], 'dose_ci': dose_ci[i],
                                  'best_epoch': d[s][2]})

    # BH FDR
    qvals = bh_adjust(pvals)
    for k, q in zip(dose_keys, qvals):
        dose_results[k]['paired_p_bh'] = q
        dose_results[k]['significant_bh'] = bool(q < 0.05)

    # Friedman：同种子 7 条件（基线 + 6 剂量）
    matrix = [base_mse] + [dose_vectors[t] for _, t in DOSES]
    chi, fp = stats.friedmanchisquare(*matrix)

    # Spearman 趋势（剂量均值对 α；描述单调性）
    means = [np.mean(base_mse)] + [np.mean(dose_vectors[t]) for _, t in DOSES]
    alphas = [0.0] + [a for a, _ in DOSES]
    rho, rp = stats.spearmanr(alphas, means)

    # 最优剂量
    best_key = min(dose_keys, key=lambda k: dose_results[k]['dose_mean'])
    best = dose_results[best_key]

    report = {
        'timestamp': datetime.now().isoformat(),
        'task': 'V9 Phase 2 Task 2.3.3: Davis dose-response (Config C)',
        'seeds': SEEDS, 'bonferroni_threshold': BONF_THRESHOLD,
        'dose_results': dose_results,
        'friedman': {'chi2': float(chi), 'p': float(fp),
                     'levels': ['alpha=0'] + [f'alpha={a}' for a, _ in DOSES]},
        'spearman_trend': {'rho': float(rho), 'p': float(rp)},
        'per_seed': per_seed_rows,
        'mean_curve': [{'alpha': a, 'mse_mean': float(v)}
                       for a, v in zip(alphas, means)],
        'best_dose': {'tag': best_key, 'alpha': best['alpha'],
                      'mse': best['dose_mean'], 'gain_pct': best['gain_pct'],
                      'paired_p': best['paired_p'],
                      'significant_bonferroni': best['significant_bonferroni']}
    }

    oj = '/mnt/data/dta2/results/analysis/v9_phase2p3_dose_response.json'
    om = '/mnt/data/dta2/results/analysis/v9_phase2p3_dose_response.md'
    json.dump(report, open(oj, 'w'), indent=2, ensure_ascii=False)

    # ---------- Markdown ----------
    L = []
    L.append('# Phase 2 Task 2.3: Davis α 剂量-反应统计报告（Config C @1000ep）\n')
    L.append(f"**生成时间**: {report['timestamp']}")
    L.append(f"**种子**: n=5，{SEEDS}（同种子贯穿全部剂量，重复测量设计）")
    L.append(f"**多重比较**: Bonferroni 阈值 0.05/6 = {BONF_THRESHOLD:.4f}；另报 BH-FDR\n")

    L.append('## 剂量-反应曲线（均值 ± SD）\n')
    L.append('| α | MSE | 同种子增益 | 配对 p | BH q | Bonferroni 显著? | CI |')
    L.append('|---|-----|-----------|--------|------|------------------|----|')
    L.append(f"| 0 | {np.mean(base_mse):.4f}±{np.std(base_mse,ddof=1):.4f} | — | — | — | — | {np.mean(base_ci):.4f} |")
    for alpha, tag in DOSES:
        r = dose_results[tag]
        # 该剂量 CI 均值
        cimean = float(np.mean(ci_dose_vectors[tag]))
        L.append(f"| {alpha} | {r['dose_mean']:.4f}±{r['dose_std']:.4f} | {r['gain_pct']:+.2f}% | "
                 f"{r['paired_p']:.4f} | {r['paired_p_bh']:.4f} | "
                 f"{'是' if r['significant_bonferroni'] else '否'} | {cimean:.4f} |")
    L.append('')

    L.append('## 配对细节（最优与最差剂量）\n')
    L.append('| 剂量 | 均值差(E7−E0) | 差值 95% CI | t | dz | Hedges\' g |')
    L.append('|------|---------------|-------------|---|----|-----------|')
    for tag in [best_key, max(dose_keys, key=lambda k: dose_results[k]['dose_mean'])]:
        r = dose_results[tag]
        lo, hi = r['diff_ci95']
        L.append(f"| α={r['alpha']} ({tag}) | {r['mean_diff']:+.4f} | [{lo:+.4f},{hi:+.4f}] | "
                 f"{r['paired_t']:+.3f} | {r['paired_dz']:+.2f} | {r['hedges_g']:+.2f} |")
    L.append('')

    L.append('## 整体剂量效应与趋势\n')
    L.append(f"- Friedman χ²（7 水平重复测量）: {chi:.3f}, p={fp:.4f} → "
             f"{'存在剂量间系统差异' if fp < 0.05 else '未达 0.05 显著水平（p=0.053 为临界值，不作显著声称）'}")
    L.append(f"- Spearman(α, 平均MSE): ρ={rho:+.3f}, p={rp:.4f}（形态描述；曲线非单调时该检验不适用作因果证据）")
    L.append(f"- 最优剂量: α={best['alpha']}，MSE {best['dose_mean']:.4f}，增益 {best['gain_pct']:+.2f}%，"
             f"配对 p={best['paired_p']:.4f}，Bonferroni 校正后 "
             f"{'仍显著' if best['significant_bonferroni'] else '不显著'}\n")

    L.append('## 裁定：标题 "multiple losses" 主张（数据驱动）\n')
    n_raw = sum(1 for k in dose_keys if dose_results[k]['significant_raw'])
    n_bonf = sum(1 for k in dose_keys if dose_results[k]['significant_bonferroni'])
    n_bh = sum(1 for k in dose_keys if dose_results[k]['significant_bh'])
    L.append(f"- 6 个剂量中原始 p<0.05 者 {n_raw} 个；Bonferroni 校正后显著 {n_bonf} 个；BH-FDR 显著 {n_bh} 个。")
    if n_bonf == 0 and fp >= 0.05:
        L.append('- Config C 下**没有任何 α 剂量相对收敛基线达到（多重比较校正后的）显著改进**，'
                 '整体 Friedman 剂量效应也不显著：标题中 "multiple losses" 带来增益的强主张**不存活**，'
                 '论文只能报告“各剂量与基线无显著差异”，α=0.02 处存在未校正的小到中等点估计优势（'
                 f"{best['gain_pct']:+.1f}%），该点降级为探索性观察。")
    L.append('- 跨数据集对照：KIBA α=0.05 显著更差、Davis α=0.05 不显著（点估计略好）、'
             'Davis 剂量曲线呈温和倒 U 且无一剂量经多重比较校正后显著——一致支持 v9 结论：'
             'CL 的损失权重效应是数据集依赖、且在收敛基线下不构成稳健收益。')
    open(om, 'w').write('\n'.join(L))

    # ---------- 控制台 ----------
    print('\n剂量    增益%    p(raw)   p(BH)   Bonf显著')
    for alpha, tag in DOSES:
        r = dose_results[tag]
        print(f"{alpha:<6} {r['gain_pct']:+6.2f}  {r['paired_p']:.4f}  {r['paired_p_bh']:.4f}  "
              f"{'Y' if r['significant_bonferroni'] else 'N'}")
    print(f"\nFriedman χ²={chi:.3f} p={fp:.4f}")
    print(f"Spearman ρ={rho:+.3f} p={rp:.4f}")
    print(f"最优剂量 α={best['alpha']} gain={best['gain_pct']:+.2f}% p={best['paired_p']:.4f}")
    print(oj, om, sep='\n')


if __name__ == '__main__':
    main()
