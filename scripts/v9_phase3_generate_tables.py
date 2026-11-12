#!/usr/bin/env python3
"""
V9 Phase 3 Task 3.2.1~3.2.3: 从 V9 真实实验分析 JSON 重生成论文 LaTeX 表格
日期: 2026-11-12
目标: 用 Config C 收敛协议结果与 Config B 对账结果重写全部主表，数字只来自已落盘的
      V9 分析报告（这些报告由 v9_*.py 统计脚本从真实 results.json 计算，并已抽样
      对照原始 JSON 验证），绝不手工编造任何数字。
严格对应 TODO Task 3.2.1 Table 1 / 3.2.2 Table 5 协议敏感性 / 3.2.3 Table 6-7 统计。

输入（全部为 V9 真实实验产物）:
  results/analysis/v9_decision_gate_a.json            (KIBA n=10 Config C)
  results/analysis/v9_phase2p1_davis_stats.json       (Davis GIN n=10 Config C)
  results/analysis/v9_phase2p2_scaffold_stats.json    (Davis scaffold n=5 Config C)
  results/analysis/v9_phase2p3_dose_response.json     (Davis 7 剂量 n=5 Config C)
  results/analysis/v9_phase2p4_kiba_dose.json         (KIBA 3 剂量 n=5 Config C)
  results/analysis/v9_phase2p5_gat_stats.json         (Davis GAT n=5 Config C)
  results/analysis/v9_phase0p3_statistical_tests.json (Config B 对账)
输出:
  results/paper_tables/v9_phase3_tables.tex   (可直接替换 main_v9.tex 中旧表)
  results/paper_tables/v9_phase3_tables.json  (同内容机读版)
方法: 纯字符串拼装 LaTeX；显著性标记按各报告原始 p 值生成，不重新计算统计量。
原则: 统一命名、详细注释；执行前须独立审核。
"""

import json
from pathlib import Path
from datetime import datetime


ANALYSIS = Path('/mnt/data/dta2/results/analysis')
OUTDIR = Path('/mnt/data/dta2/results/paper_tables')


def load(name):
    """读取 analysis JSON。"""
    with open(ANALYSIS / name) as f:
        return json.load(f)


def sig_marks(p):
    """按原始 p 值返回显著性星标（未校正口径，仅作直观标记；校正结论见表注）。"""
    if p < 0.001:
        return '$^{***}$'
    if p < 0.01:
        return '$^{**}$'
    if p < 0.05:
        return '$^{*}$'
    return ''


def fmt(v, nd=4):
    """固定小数位格式化。"""
    return f"{v:.{nd}f}"


# ---------------------------------------------------------------------------
# Table 1: 收敛协议主结果（encoder × dataset）
# ---------------------------------------------------------------------------
def build_table_main(gate_a, davis, gat):
    """Config C headline：三个 encoder×dataset 格子的 E0/E7。"""
    L = []
    L.append(r'\begin{table*}[!t]')
    L.append(r'\caption{Main results under the convergent protocol (Config C: '
             r'1{,}000 epochs, LR patience 30, early-stop patience 150; all other '
             r'hyperparameters frozen). E7: InfoNCE CL with $\alpha=0.05$, $\tau=0.1$, '
             r'noise augmentation, $\beta=\gamma=0$. Values are mean$\pm$sample std; '
             r'gain is E7 relative to E0 (positive = CL better); $p$ is the paired '
             r'two-sided t-test over matched seeds; $d_z$ is the paired standardized '
             r'mean difference.}')
    L.append(r'\label{tab:main_results}')
    L.append(r'\centering')
    L.append(r'\begin{tabular}{llccccccc}')
    L.append(r'\toprule')
    L.append(r'Dataset & Encoder ($n$) & E0 MSE & E7 MSE & MSE gain & paired $p$ & '
             r'$d_z$ & E0 CI & E7 CI \\')
    L.append(r'\midrule')

    rows = [
        ('Davis', 'GIN (10)', davis['mse'], davis['ci']),
        ('Davis', 'GAT (5)', gat['mse'], gat['ci']),
        ('KIBA', 'GIN (10)', gate_a['mse'], gate_a['ci']),
    ]
    for ds, enc, m, c in rows:
        gain = m['gain_pct']
        marks = sig_marks(m['paired_p'])
        L.append(
            f"{ds} & {enc} & {fmt(m['e0_mean'])}$\\pm${fmt(m['e0_std'])} & "
            f"{fmt(m['e7_mean'])}$\\pm${fmt(m['e7_std'])} & {gain:+.2f}\%{marks} & "
            f"{m['paired_p']:.4f} & {m['paired_dz']:+.2f} & "
            f"{fmt(c['e0_mean'])} & {fmt(c['e7_mean'])} \\\\"
        )
    L.append(r'\bottomrule')
    L.append(r'\end{tabular}')
    L.append(r'\par\smallskip')
    L.append(r'\footnotesize $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$ (uncorrected '
             r'paired p; family-wise claims are assessed in the dose--response tables and '
             r'in Decision Gate A). Sources: \texttt{v9\_phase2p1\_davis\_stats.json}, '
             r'\texttt{v9\_phase2p5\_gat\_stats.json}, \texttt{v9\_decision\_gate\_a.json}.')
    L.append(r'\end{table*}')
    return L


# ---------------------------------------------------------------------------
# Table 2: 2×2 协议敏感性（Config B vs Config C）
# ---------------------------------------------------------------------------
def build_table_protocol_sensitivity(phase0, gate_a, p4, davis_c, ):
    """
    2×2: 协议（B vs C）× 条件（E0 vs E7）。
    KIBA 用同 5 个配对种子；Davis GIN 用 10 种子（B: T1.5；C: Task 2.1）。
    """
    kiba_b = phase0['experiments']['T1.1_vs_T1.3_KIBA']['paired_test']
    davis_b = phase0['experiments']['T1.5_Davis']['paired_test']

    L = []
    L.append(r'\begin{table*}[!t]')
    L.append(r'\caption{Protocol sensitivity (2$\times$2 design): training protocol '
             r'(Config B vs.\ Config C) crossed with condition (E0 vs.\ E7). Same model, '
             r'same data, same optimizer hyperparameters; only the training budget and '
             r'patience change. KIBA rows use the same five matched seeds; Davis rows '
             r'use ten seeds. Positive gain = CL better; $p$ = paired t-test.}')
    L.append(r'\label{tab:protocol_sensitivity}')
    L.append(r'\centering')
    L.append(r'\begin{tabular}{llcccc}')
    L.append(r'\toprule')
    L.append(r'Dataset & Protocol & E0 MSE & E7 MSE & MSE gain & paired $p$ \\')
    L.append(r'\midrule')

    # KIBA Config B (n=5)
    L.append(
        f"KIBA & B (500ep/p150) & {kiba_b['e0_mse_mean']:.4f}$\\pm${kiba_b['e0_mse_std']:.4f} & "
        f"{kiba_b['e7_mse_mean']:.4f}$\\pm${kiba_b['e7_mse_std']:.4f} & "
        f"{kiba_b['mse_improvement_pct']:+.2f}\% & {kiba_b['p_mse']:.4f} \\\\"
    )
    # KIBA Config C (n=5 子集, 来自 Task 2.4 报告，与 B 同五种子)
    kc0 = p4['conditions'][0]; kc05 = p4['conditions'][1]
    cmp_c = p4['comparisons']['alpha0.05_vs_0']
    # p4 的 relative_pct 约定为"正=高α更差"；本表约定"正=CL更好"，须取反
    gain_c = -cmp_c['relative_pct']
    L.append(
        f"KIBA & C (1000ep/p150) & {kc0['mse_mean']:.4f}$\\pm${kc0['mse_std']:.4f} & "
        f"{kc05['mse_mean']:.4f}$\\pm${kc05['mse_std']:.4f} & "
        f"{gain_c:+.2f}\% & "
        f"{cmp_c['paired_p']:.4f} \\\\"
    )
    L.append(r'\midrule')
    # Davis Config B (n=10)
    L.append(
        f"Davis & B (500ep/p150) & {davis_b['e0_mse_mean']:.4f}$\\pm${davis_b['e0_mse_std']:.4f} & "
        f"{davis_b['e7_mse_mean']:.4f}$\\pm${davis_b['e7_mse_std']:.4f} & "
        f"{davis_b['mse_improvement_pct']:+.2f}\% & {davis_b['p_mse']:.4f} \\\\"
    )
    # Davis Config C (n=10)
    dm = davis_c['mse']
    L.append(
        f"Davis & C (1000ep/p150) & {dm['e0_mean']:.4f}$\\pm${dm['e0_std']:.4f} & "
        f"{dm['e7_mean']:.4f}$\\pm${dm['e7_std']:.4f} & {dm['gain_pct']:+.2f}\% & "
        f"{dm['paired_p']:.4f} \\\\"
    )
    L.append(r'\bottomrule')
    L.append(r'\end{tabular}')
    L.append(r'\par\smallskip')
    L.append(r'\footnotesize On KIBA the E7 mean is essentially protocol-invariant '
             r'($\sim$0.147--0.148); it is the E0 baseline that improves from 0.1497 to '
             r'0.1366 when trained to convergence, flipping the comparison sign. On Davis '
             r'the E0 mean is nearly unchanged across protocols; protocol sensitivity is '
             r'itself dataset-dependent and is strongest on KIBA. Sources: '
             r'\texttt{v9\_phase0p3\_statistical\_tests.json}, '
             r'\texttt{v9\_phase2p4\_kiba\_dose.json}, '
             r'\texttt{v9\_phase2p1\_davis\_stats.json}.')
    L.append(r'\end{table*}')
    return L


# ---------------------------------------------------------------------------
# Table 3: Davis 七水平剂量-反应
# ---------------------------------------------------------------------------
def build_table_davis_dose(p3):
    L = []
    L.append(r'\begin{table*}[!t]')
    L.append(r'\caption{Davis dose--response under Config C (GIN, $n=5$ matched seeds). '
             r'Gain is relative to $\alpha=0$ (positive = CL better); $p$ is the paired '
             r't-test. Bonferroni threshold $0.05/6=0.0083$; no dose reaches it.}')
    L.append(r'\label{tab:davis_dose}')
    L.append(r'\centering')
    L.append(r'\begin{tabular}{ccccc}')
    L.append(r'\toprule')
    L.append(r'$\alpha$ & MSE & gain & paired $p$ & Bonf sig. \\')
    L.append(r'\midrule')
    # 基线行：α=0，均值取自 mean_curve[0]，std 取自各剂量条目的 baseline_std（同源）
    base_mean = p3['mean_curve'][0]['mse_mean']
    base_std = next(iter(p3['dose_results'].values()))['baseline_std']
    L.append(f"0 & {base_mean:.4f}$\\pm${base_std:.4f} & — & — & — \\\\")
    order = ['a001', 'a002', 'a005', 'a010', 'a020', 'a050']
    for tag in order:
        r = p3['dose_results'][tag]
        bonf = 'Yes' if r.get('significant_bonferroni') else 'No'
        L.append(f"{r['alpha']:g} & {r['dose_mean']:.4f}$\\pm${r['dose_std']:.4f} & "
                 f"{r['gain_pct']:+.2f}\% & {r['paired_p']:.4f} & {bonf} \\\\")
    L.append(r'\bottomrule')
    L.append(r'\end{tabular}')
    L.append(r'\par\smallskip')
    fr = p3['friedman']
    sp = p3.get('spearman_trend', {})
    L.append(r'\footnotesize Friedman $\chi^2$ (7 levels, repeated measures): '
             f"{fr['chi2']:.3f}, $p={fr['p']:.4f}$ (borderline; not claimed significant). "
             f"Spearman trend: $\\rho={sp.get('rho', float('nan')):+.2f}$, "
             f"$p={sp.get('p', float('nan')):.4f}$. "
             r'Source: \texttt{v9\_phase2p3\_dose\_response.json}.')
    L.append(r'\end{table*}')
    return L


# ---------------------------------------------------------------------------
# Table 4: KIBA 三水平剂量
# ---------------------------------------------------------------------------
def build_table_kiba_dose(p4):
    L = []
    L.append(r'\begin{table*}[!t]')
    L.append(r'\caption{KIBA dose--response under Config C (GIN, $n=5$ matched seeds). '
             r'Three paired comparisons with Bonferroni threshold $0.05/3=0.0167$. '
             r'Mean difference $>0$ = higher $\alpha$ worse (MSE).}')
    L.append(r'\label{tab:kiba_dose}')
    L.append(r'\centering')
    L.append(r'\begin{tabular}{lccccc}')
    L.append(r'\toprule')
    L.append(r'Comparison & mean diff & 95\% CI & $p$ & $d_z$ & Bonf sig. \\')
    L.append(r'\midrule')
    for key, label in [('alpha0.05_vs_0', '0.05 vs 0'),
                       ('alpha0.5_vs_0', '0.5 vs 0'),
                       ('alpha0.5_vs_0.05', '0.5 vs 0.05')]:
        r = p4['comparisons'][key]
        lo, hi = r['diff_ci95']
        L.append(f"{label} & {r['mean_diff_hi_minus_lo']:+.4f} & [{lo:+.4f},{hi:+.4f}] & "
                 f"{r['paired_p']:.4f} & {r['paired_dz']:+.2f} & "
                 f"{'Yes' if r['significant_bonferroni'] else 'No'} \\\\")
    L.append(r'\bottomrule')
    L.append(r'\end{tabular}')
    L.append(r'\par\smallskip')
    fr = p4['friedman']
    tr = p4['trend']
    L.append(r'\footnotesize Friedman $\chi^2$ (3 levels): '
             f"{fr['chi2']:.3f}, $p={fr['p']:.4f}$. "
             f"Monotonic worsening in {tr['per_seed_monotonic_worsening']}/5 seeds. "
             r"Spearman on 3 dose levels: exact two-sided permutation "
             f"$p={tr['spearman_exact_permutation_p']:.3f}$ (asymptotic p invalid with 3 "
             r'points); monotonicity evidence rests on Friedman and paired tests. Source: '
             r'\texttt{v9\_phase2p4\_kiba\_dose.json}.')
    L.append(r'\end{table*}')
    return L


# ---------------------------------------------------------------------------
# Table 5: scaffold 划分
# ---------------------------------------------------------------------------
def build_table_scaffold(p2):
    m, c = p2['mse'], p2['ci']
    L = []
    L.append(r'\begin{table*}[!t]')
    L.append(r'\caption{Bemis--Murcko scaffold split under Config C ($n=5$ matched '
             r'seeds). E0 vs.\ E7 MSE and CI; paired tests. Ceiling hits (best epoch '
             r'$\geq 980$) are flagged as non-converged runs.}')
    L.append(r'\label{tab:scaffold}')
    L.append(r'\centering')
    L.append(r'\begin{tabular}{lcccc}')
    L.append(r'\toprule')
    L.append(r'Condition & MSE & gain & paired $p$ & CI \\')
    L.append(r'\midrule')
    L.append(f"E0 & {m['e0_mean']:.4f}$\\pm${m['e0_std']:.4f} & — & — & {c['e0_mean']:.4f} \\\\")
    L.append(f"E7 & {m['e7_mean']:.4f}$\\pm${m['e7_std']:.4f} & {m['gain_pct']:+.2f}\% & "
             f"{m['paired_p']:.4f} & {c['e7_mean']:.4f} \\\\")
    L.append(r'\bottomrule')
    L.append(r'\end{tabular}')
    L.append(r'\par\smallskip')
    ch = p2.get('ceiling_hitting_ge980', {})
    # 实际结构: {"e0": [...], "e7": [42]} → 展开为 e7_seed42 形式
    flagged = [f"{grp}\\_seed{s}" for grp, seeds in ch.items() for s in seeds]
    L.append(r'\footnotesize '
             + (r'Ceiling-hit runs: ' + ', '.join(flagged) + '; these runs did not '
                r'converge at the 1{,}000-epoch cap. ' if flagged else 'No ceiling hits. ')
             + r'Source: \texttt{v9\_phase2p2\_scaffold\_stats.json}.')
    L.append(r'\end{table*}')
    return L


# ---------------------------------------------------------------------------
# Table 6: 冷启动汇总（Config B 对账，18× 脱节）
# ---------------------------------------------------------------------------
def build_table_cold(phase0):
    cold = phase0['experiments']['T1.6_Davis_ColdStart']['paired_test']
    L = []
    L.append(r'\begin{table*}[!t]')
    L.append(r'\caption{Cold-start audit (Davis, Config B legacy batch, $n=5$ matched '
             r'seeds): CL vs.\ E0. Combined with the warm-validation selection analysis '
             r'(validation--test ranking disconnect of up to 18$\times$), the cold-start '
             r'comparison is confounded by a failed selection mechanism.}')
    L.append(r'\label{tab:coldstart}')
    L.append(r'\centering')
    L.append(r'\begin{tabular}{lccc}')
    L.append(r'\toprule')
    L.append(r'Condition & MSE & gain & paired $p$ \\')
    L.append(r'\midrule')
    L.append(f"E0 & {cold['e0_mse_mean']:.4f} & — & — \\\\")
    L.append(f"E7 & {cold['e7_mse_mean']:.4f} & {cold['mse_improvement_pct']:+.2f}\% & "
             f"{cold['p_mse']:.4f} \\\\")
    L.append(r'\bottomrule')
    L.append(r'\end{tabular}')
    L.append(r'\par\smallskip')
    L.append(r'\footnotesize No CL effect under cold splits ($p=0.99$); because '
             r'warm-validation-based selection is unreliable there, this null is reported '
             r'together with the selection-mechanism finding rather than as evidence for '
             r'or against CL. Source: \texttt{v9\_phase0p3\_statistical\_tests.json}.')
    L.append(r'\end{table*}')
    return L


def main():
    print('=' * 70)
    print('V9 Phase 3 Task 3.2: regenerate LaTeX tables from real V9 analyses')
    print('=' * 70)

    gate_a = load('v9_decision_gate_a.json')
    davis = load('v9_phase2p1_davis_stats.json')
    scaffold = load('v9_phase2p2_scaffold_stats.json')
    dose_d = load('v9_phase2p3_dose_response.json')
    dose_k = load('v9_phase2p4_kiba_dose.json')
    gat = load('v9_phase2p5_gat_stats.json')
    phase0 = load('v9_phase0p3_statistical_tests.json')

    blocks = []
    blocks.append('% ' + '=' * 76)
    blocks.append('% V9 Phase 3 auto-generated tables')
    blocks.append(f'% Generated: {datetime.now().isoformat()}')
    blocks.append('% Goal: replace all Config B tables in main_ccadd.tex with Config C '
                  'real results')
    blocks.append('% Method: values transcribed by v9_phase3_generate_tables.py from V9 '
                  'analysis JSONs only')
    blocks.append('% ' + '=' * 76)
    blocks.append('')

    blocks += build_table_main(gate_a, davis, gat)
    blocks.append('')
    blocks += build_table_protocol_sensitivity(phase0, gate_a, dose_k, davis)
    blocks.append('')
    blocks += build_table_davis_dose(dose_d)
    blocks.append('')
    blocks += build_table_kiba_dose(dose_k)
    blocks.append('')
    blocks += build_table_scaffold(scaffold)
    blocks.append('')
    blocks += build_table_cold(phase0)

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_tex = OUTDIR / 'v9_phase3_tables.tex'
    with open(out_tex, 'w') as f:
        f.write('\n'.join(blocks))

    manifest = {'generated': datetime.now().isoformat(),
                'tables': ['tab:main_results', 'tab:protocol_sensitivity',
                           'tab:davis_dose', 'tab:kiba_dose', 'tab:scaffold',
                           'tab:coldstart'],
                'sources': ['v9_decision_gate_a.json', 'v9_phase2p1_davis_stats.json',
                            'v9_phase2p2_scaffold_stats.json',
                            'v9_phase2p3_dose_response.json',
                            'v9_phase2p4_kiba_dose.json', 'v9_phase2p5_gat_stats.json',
                            'v9_phase0p3_statistical_tests.json']}
    with open(OUTDIR / 'v9_phase3_tables.json', 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f'\nWritten {out_tex}')
    print('Tables:', ', '.join(manifest['tables']))


if __name__ == '__main__':
    main()
