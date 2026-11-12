#!/usr/bin/env python3
"""
V9 Phase 2 Task 2.5.4: GAT Davis E0 vs E7 (Config C @1000ep) 统计检验 + encoders 主张裁定
日期: 2026-11-11
目标: 对 Task 2.5 产出的 Davis 标准随机划分 GAT 编码器 E0 / E7(α=0.05) 各 5 个同种子
      Config C 结果做系统统计，并据此裁定论文标题中 "multiple encoders" 主张是否存活：
        1. MSE / CI 的同种子配对差异（配对 t，主检验）
        2. Welch t（辅助，不假设方差齐性）
        3. Levene 方差检验与方差比
        4. 配对效应量 dz 与 Hedges' g（含小样本修正 1-3/(4n-9)）
        5. 收敛性核查：best_epoch ≥980 视为贴顶（1000 上限下未收敛），须在结论中 flag
        6. 参数量硬核查：total_params 必须为 521,345（GATControlledDTA）
        7. 与 GIN（Task 2.1，同种子）的跨编码器描述性对照
      裁定规则（事先固定，避免事后移动门限）：
        - encoders 主张"存活"：GAT 上 MSE 配对检验 p<0.05 且增益为正（CL 更好），
          CI 方向不矛盾（CI 配对 p>=0.05 或增益为正）。
        - 其余一切情形（不显著 / 反向显著 / CI 反向显著）→ encoders 主张不存活，
          标题须收窄为仅 GIN（或仅报告为负结果/数据集与编码器条件依赖）。
严格对应 TODO_v9_phase1_2_3.md Task 2.5.4~2.5.5，不简化、不改目标。
输入:
  results/controlled/v9_p2p5_gat_e0_seed*/results.json
  results/controlled/v9_p2p5_gat_e7_seed*/results.json
  results/controlled/v9_p2_davis_e0_seed*/results.json    (GIN, Task 2.1, 跨编码器对照)
  results/controlled/v9_p2_davis_e7_seed*/results.json
输出:
  results/analysis/v9_phase2p5_gat_stats.json
  results/analysis/v9_phase2p5_gat_stats.md
原则: 数字只由本脚本从真实实验 JSON 计算，不编造。
"""

import json
import glob
from datetime import datetime

import numpy as np
from scipy import stats


EXPECTED_GAT_PARAMS = 521345
CEILING_EPOCH = 980          # best_epoch >= 980 @1000 cap → 贴顶
GIN_PATTERN_E0 = '/mnt/data/dta2/results/controlled/v9_p2_davis_e0_seed*/results.json'
GIN_PATTERN_E7 = '/mnt/data/dta2/results/controlled/v9_p2_davis_e7_seed*/results.json'
GAT_PATTERN_E0 = '/mnt/data/dta2/results/controlled/v9_p2p5_gat_e0_seed*/results.json'
GAT_PATTERN_E7 = '/mnt/data/dta2/results/controlled/v9_p2p5_gat_e7_seed*/results.json'


def load_seed_results(pattern):
    """返回 {seed: dict(mse, ci, rmse, best_epoch, best_val_mse, total_params)}。"""
    out = {}
    for rf in sorted(glob.glob(pattern)):
        with open(rf) as f:
            d = json.load(f)
        out[int(d['seed'])] = {
            'mse': float(d['test_mse']),
            'ci': float(d['test_ci']),
            'rmse': float(d['test_rmse']),
            'best_epoch': int(d['best_epoch']),
            'best_val_mse': float(d['best_val_mse']),
            'total_params': int(d.get('total_params', -1)),
        }
    return out


def hedges_correction(n_total):
    """Hedges 小样本修正因子 1 - 3/(4n-9)。"""
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


def adjudicate_encoders(mse_res, ci_res, ceiling_hits):
    """
    encoders 主张裁定（规则见模块 docstring，门限事先固定）。
    返回 dict(claim_survives: bool, verdict: str, rationale: str)。
    """
    mse_sig_pos = (mse_res['paired_p'] < 0.05) and (mse_res['gain_pct'] > 0)
    mse_sig_neg = (mse_res['paired_p'] < 0.05) and (mse_res['gain_pct'] <= 0)
    ci_sig_pos = (ci_res['paired_p'] < 0.05) and (ci_res['gain_pct'] > 0)
    ci_sig_neg = (ci_res['paired_p'] < 0.05) and (ci_res['gain_pct'] <= 0)
    has_ceiling = len(ceiling_hits) > 0

    if mse_sig_pos and not ci_sig_neg:
        survives = True
        verdict = 'encoders 主张存活：GAT 上 CL（MSE）显著优于自身收敛基线，且 CI 不反向'
        rationale = (f"MSE 配对 p={mse_res['paired_p']:.4f}<0.05，增益 "
                     f"{mse_res['gain_pct']:+.2f}%；CI p={ci_res['paired_p']:.4f}，"
                     f"增益 {ci_res['gain_pct']:+.2f}%。")
    elif mse_sig_neg or ci_sig_neg:
        survives = False
        if mse_sig_neg and ci_sig_neg:
            verdict = 'encoders 主张不存活：GAT 上 CL 在 MSE 与 CI 上均显著更差'
        elif mse_sig_neg:
            verdict = 'encoders 主张不存活：GAT 上 CL 的 MSE 显著更差'
        else:
            verdict = 'encoders 主张不存活：GAT 上 CL 的 CI 显著更差'
        rationale = (f"MSE p={mse_res['paired_p']:.4f}，增益 {mse_res['gain_pct']:+.2f}%；"
                     f"CI p={ci_res['paired_p']:.4f}，增益 {ci_res['gain_pct']:+.2f}%。"
                     "标题须收窄 encoders 表述。")
    else:
        survives = False
        verdict = 'encoders 主张不存活：GAT 上 E0 与 E7 无统计显著差异'
        rationale = (f"n=5 下 MSE p={mse_res['paired_p']:.4f}、CI p={ci_res['paired_p']:.4f}，"
                     f"点估计 MSE {mse_res['gain_pct']:+.2f}%、CI {ci_res['gain_pct']:+.2f}%；"
                     "CL 在 GAT 上无可检出收益，标题中 multiple encoders 泛化声称须收窄"
                     "（可作为编码器条件依赖/负结果如实报告）。")

    if has_ceiling:
        rationale += (f" 注意：{len(ceiling_hits)} 个 run 贴顶（{ceiling_hits}），"
                      "对应条件的均值受未收敛 run 影响，结论须同时披露该限制。")
    return {'claim_survives': survives, 'verdict': verdict, 'rationale': rationale}


def write_markdown(report, path):
    m, c, v = report['mse'], report['ci'], report['variance']
    adj = report['adjudication']
    L = []
    L.append('# Phase 2 Task 2.5: GAT Davis E0 vs E7 统计与 encoders 主张裁定（Config C @1000ep）\n')
    L.append(f"**生成时间**: {report['timestamp']}")
    L.append(f"**配对种子**: n={m['n']}，{report['common_seeds']}")
    L.append(f"**参数量核查**: total_params={report['param_check']['observed']} "
             f"（期望 {EXPECTED_GAT_PARAMS}）→ "
             f"{'通过' if report['param_check']['pass'] else '未通过'}\n")

    L.append('## 逐种子配对\n')
    L.append('| Seed | E0 MSE (ep) | E7 MSE (ep) | MSE差(E7−E0) | E0 CI | E7 CI |')
    L.append('|------|-------------|-------------|--------------|-------|-------|')
    for row in report['per_seed']:
        L.append(f"| {row['seed']} | {row['e0_mse']:.4f} ({row['e0_epoch']}) | "
                 f"{row['e7_mse']:.4f} ({row['e7_epoch']}) | "
                 f"{row['e7_mse']-row['e0_mse']:+.4f} | {row['e0_ci']:.4f} | {row['e7_ci']:.4f} |")
    L.append('')

    L.append('## 收敛性核查（贴顶 best_epoch ≥ 980）\n')
    ch = report['ceiling_hits']
    if ch:
        L.append(f"- ⚠️ 贴顶 run: {ch}；这些 run 在 1000 epoch 上限下未收敛，结论须附带该限制披露。")
    else:
        L.append('- 全部 10 个 run best_epoch < 980，early stop 正常触发，无贴顶。')
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

    L.append('## 跨编码器对照（同种子，Task 2.1 GIN vs 本批 GAT）\n')
    L.append('| Seed | GIN E0 MSE | GAT E0 MSE | GIN E7 MSE | GAT E7 MSE |')
    L.append('|------|------------|------------|------------|------------|')
    for row in report['cross_encoder']:
        L.append(f"| {row['seed']} | {row['gin_e0_mse']:.4f} | {row['gat_e0_mse']:.4f} | "
                 f"{row['gin_e7_mse']:.4f} | {row['gat_e7_mse']:.4f} |")
    gm = report['cross_encoder_means']
    L.append(f"| **均值** | {gm['gin_e0']:.4f} | {gm['gat_e0']:.4f} | "
             f"{gm['gin_e7']:.4f} | {gm['gat_e7']:.4f} |\n")
    L.append(f"- GAT E0 相对 GIN E0 高 {gm['gat_e0']/gm['gin_e0']*100-100:+.1f}%（MSE，越大越差）。")
    L.append('- GIN 上 CL 效应（Task 2.1，n=10）在 Davis 不显著；GAT 上的裁定见下。\n')

    L.append('---\n')
    L.append('## encoders 主张裁定\n')
    L.append(f"- **裁定**: {'✅ ' + adj['verdict'] if adj['claim_survives'] else '❌ ' + adj['verdict']}")
    L.append(f"- **依据**: {adj['rationale']}")
    L.append(f"- **claim_survives** = {adj['claim_survives']}（该布尔值供 Phase 3 标题/摘要改写直接引用）")

    with open(path, 'w') as f:
        f.write('\n'.join(L))


def main():
    print('=' * 70)
    print('V9 Phase 2 Task 2.5.4: GAT Davis E0 vs E7 stats + encoder adjudication')
    print('=' * 70)

    gat_e0 = load_seed_results(GAT_PATTERN_E0)
    gat_e7 = load_seed_results(GAT_PATTERN_E7)
    common = sorted(set(gat_e0) & set(gat_e7))
    print(f'GAT 配对 n={len(common)}: {common}')
    if len(common) < 5:
        raise SystemExit(f'❌ 配对种子不足 5（当前 {len(common)}），E7 尚未全部完成，拒绝出报告。')

    # 参数量硬核查（10 个 run 全部须为 521,345）
    params = sorted({gat_e0[s]['total_params'] for s in common}
                    | {gat_e7[s]['total_params'] for s in common})
    param_pass = params == [EXPECTED_GAT_PARAMS]
    param_check = {'observed': params, 'expected': EXPECTED_GAT_PARAMS, 'pass': param_pass}
    print('参数量核查:', params, 'pass =', param_pass)
    if not param_pass:
        raise SystemExit('❌ 参数量与期望 GATControlledDTA 521,345 不符，停止以免混入错误模型结果。')

    e0m = [gat_e0[s]['mse'] for s in common]
    e7m = [gat_e7[s]['mse'] for s in common]
    e0c = [gat_e0[s]['ci'] for s in common]
    e7c = [gat_e7[s]['ci'] for s in common]

    mse = analyze_metric(e0m, e7m, higher_is_better=False)
    ci = analyze_metric(e0c, e7c, higher_is_better=True)
    var = variance_test(e0m, e7m)

    # 贴顶清单
    ceiling_hits = []
    for s in common:
        for grp, src in (('e0', gat_e0), ('e7', gat_e7)):
            if src[s]['best_epoch'] >= CEILING_EPOCH:
                ceiling_hits.append(f'{grp}_seed{s}={src[s]["best_epoch"]}')
    print('贴顶 run:', ceiling_hits if ceiling_hits else '无')

    per_seed = [{'seed': s,
                 'e0_mse': gat_e0[s]['mse'], 'e7_mse': gat_e7[s]['mse'],
                 'e0_ci': gat_e0[s]['ci'], 'e7_ci': gat_e7[s]['ci'],
                 'e0_epoch': gat_e0[s]['best_epoch'], 'e7_epoch': gat_e7[s]['best_epoch']}
                for s in common]

    # 跨编码器对照（同 5 种子的 GIN Task 2.1）
    gin_e0 = load_seed_results(GIN_PATTERN_E0)
    gin_e7 = load_seed_results(GIN_PATTERN_E7)
    cross_encoder = []
    for s in common:
        if s in gin_e0 and s in gin_e7:
            cross_encoder.append({'seed': s,
                                  'gin_e0_mse': gin_e0[s]['mse'], 'gat_e0_mse': gat_e0[s]['mse'],
                                  'gin_e7_mse': gin_e7[s]['mse'], 'gat_e7_mse': gat_e7[s]['mse']})
    g_gin_e0 = np.mean([r['gin_e0_mse'] for r in cross_encoder])
    g_gat_e0 = np.mean([r['gat_e0_mse'] for r in cross_encoder])
    g_gin_e7 = np.mean([r['gin_e7_mse'] for r in cross_encoder])
    g_gat_e7 = np.mean([r['gat_e7_mse'] for r in cross_encoder])
    cross_encoder_means = {'gin_e0': float(g_gin_e0), 'gat_e0': float(g_gat_e0),
                           'gin_e7': float(g_gin_e7), 'gat_e7': float(g_gat_e7)}

    adjudication = adjudicate_encoders(mse, ci, ceiling_hits)
    print('\n裁定:', adjudication['verdict'])

    report = {'timestamp': datetime.now().isoformat(),
              'task': 'V9 Phase 2 Task 2.5.4: GAT Davis E0 vs E7 (Config C)',
              'common_seeds': common, 'param_check': param_check,
              'ceiling_hits': ceiling_hits,
              'per_seed': per_seed,
              'mse': mse, 'ci': ci, 'variance': var,
              'cross_encoder': cross_encoder,
              'cross_encoder_means': cross_encoder_means,
              'adjudication': adjudication}

    out_j = '/mnt/data/dta2/results/analysis/v9_phase2p5_gat_stats.json'
    out_m = '/mnt/data/dta2/results/analysis/v9_phase2p5_gat_stats.md'
    with open(out_j, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    write_markdown(report, out_m)

    print(f"\nMSE 增益 {mse['gain_pct']:+.2f}% p={mse['paired_p']:.4f} dz={mse['paired_dz']:+.2f}")
    print(f"CI  增益 {ci['gain_pct']:+.2f}% p={ci['paired_p']:.4f} dz={ci['paired_dz']:+.2f}")
    print(f"方差缩减 {var['variance_reduction_pct']:+.1f}% Levene p={var['levene_p']:.4f}")
    print(f"claim_survives = {adjudication['claim_survives']}")
    print(f'\n{out_j}\n{out_m}')


if __name__ == '__main__':
    main()
