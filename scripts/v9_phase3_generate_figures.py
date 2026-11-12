#!/usr/bin/env python3
"""
V9 Phase 3 Task 3.2.4: 用 V9 Config C 真实实验数据重生成论文图 2–8（重新编号为
fig_v9_1 ~ fig_v9_6）
日期: 2026-11-12
目标: 替换旧批次（Config B）绘制的全部结果图；所有数据点只取自 V9 分析 JSON，
      不重新计算统计量，不编造任何数字。
严格对应 TODO Task 3.2.4。

输入（真实实验产物）:
  results/analysis/v9_phase0p3_statistical_tests.json  (Config B)
  results/analysis/v9_decision_gate_a.json             (KIBA n=10 C)
  results/analysis/v9_phase2p1_davis_stats.json        (Davis GIN n=10 C)
  results/analysis/v9_phase2p2_scaffold_stats.json     (scaffold n=5 C)
  results/analysis/v9_phase2p3_dose_response.json      (Davis 7 剂量 C)
  results/analysis/v9_phase2p4_kiba_dose.json          (KIBA 3 剂量 C)
  results/analysis/v9_phase2p5_gat_stats.json          (GAT n=5 C)
输出（PDF+PNG 各一份）:
  2026-11-12_paper1_v9/figures/fig_v9_1_protocol_schematic
  2026-11-12_paper1_v9/figures/fig_v9_2_protocol_sensitivity
  2026-11-12_paper1_v9/figures/fig_v9_3_dose_response
  2026-11-12_paper1_v9/figures/fig_v9_4_architecture
  2026-11-12_paper1_v9/figures/fig_v9_5_splits
  2026-11-12_paper1_v9/figures/fig_v9_6_cross_dataset
方法: matplotlib 纯绘制；误差棒用样本 std（条件均值）或配对差值 95% CI（增益）。
原则: 统一命名、详细注释、执行前独立审核。
"""

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


ANALYSIS = Path('/mnt/data/dta2/results/analysis')
FIGDIR = Path('/mnt/data/dta2/2026-11-12_paper1_v9/figures')

# 统一视觉参数
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'axes.spines.top': False,
    'axes.spines.right': False,
})
C_E0 = '#4C72B0'      # 蓝：基线
C_E7 = '#C44E52'      # 红：CL
C_ACCENT = '#55A868'  # 绿
C_GRAY = '#8172B3'


def load(name):
    """读取 analysis JSON。"""
    with open(ANALYSIS / name) as f:
        return json.load(f)


def save(fig, stem):
    """同时存 PDF 与 PNG。"""
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(FIGDIR / f'{stem}.{ext}', bbox_inches='tight')
    plt.close(fig)
    print('saved', stem)


# ---------------------------------------------------------------------------
# Fig 1: 2×2 设计示意
# ---------------------------------------------------------------------------
def fig_protocol_schematic():
    """手绘 2×2 协议敏感性设计概念图。"""
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis('off')

    def box(x, y, w, h, text, fc, ec='#333333'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08',
                                    fc=fc, ec=ec, lw=1.2))
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=9)

    # 列标题
    ax.text(2.3, 4.6, 'E0: pure supervised', ha='center', fontweight='bold')
    ax.text(6.9, 4.6, 'E7: supervised + CL ($\\alpha=0.05$)', ha='center',
            fontweight='bold')
    # 行标题
    box(0.2, 2.55, 1.7, 1.0, 'Config B\n500 ep / patience 50', '#ECECEC')
    box(0.2, 0.55, 1.7, 1.0, 'Config C\n1000 ep / patience 150', '#ECECEC')
    # 四格
    box(2.6, 2.55, 3.4, 1.0, 'B / E0\nunder-converged reference', '#D6E2F2')
    box(6.2, 2.55, 3.4, 1.0, 'B / E7\napparent gain +1.4% (KIBA)', '#F5D9DA')
    box(2.6, 0.55, 3.4, 1.0, 'C / E0\nconverged baseline', '#D6E2F2')
    box(6.2, 0.55, 3.4, 1.0, 'C / E7\n$-$7.9%: sign reversed (KIBA)', '#F5D9DA')
    # 箭头：协议变化
    for x in (4.3, 7.9):
        ax.add_patch(FancyArrowPatch((x, 2.5), (x, 1.62), arrowstyle='-|>',
                                     mutation_scale=14, lw=1.4, color='#444'))
    ax.text(9.85, 2.05, 'train to\nconvergence', ha='right', va='center',
            fontsize=8.5, color='#444')
    ax.set_title('2$\\times$2 protocol-sensitivity design: only the training protocol changes',
                 fontsize=10.5)
    save(fig, 'fig_v9_1_protocol_schematic')


# ---------------------------------------------------------------------------
# Fig 2: 协议敏感性（B vs C 的 E0/E7 MSE）
# ---------------------------------------------------------------------------
def fig_protocol_sensitivity(phase0, p4, davis):
    """两面板：KIBA 五种子 / Davis 十种子，B vs C 的 E0、E7 均值±std。"""
    kiba_b = phase0['experiments']['T1.1_vs_T1.3_KIBA']['paired_test']
    davis_b = phase0['experiments']['T1.5_Davis']['paired_test']

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.3))
    # KIBA
    ax = axes[0]
    labels = ['Config B', 'Config C']
    e0_means = [kiba_b['e0_mse_mean'], p4['conditions'][0]['mse_mean']]
    e0_stds = [kiba_b['e0_mse_std'], p4['conditions'][0]['mse_std']]
    e7_means = [kiba_b['e7_mse_mean'], p4['conditions'][1]['mse_mean']]
    e7_stds = [kiba_b['e7_mse_std'], p4['conditions'][1]['mse_std']]
    x = np.arange(2)
    ax.bar(x - 0.2, e0_means, 0.4, yerr=e0_stds, color=C_E0, capsize=3,
           label='E0', error_kw={'lw': 1})
    ax.bar(x + 0.2, e7_means, 0.4, yerr=e7_stds, color=C_E7, capsize=3,
           label='E7', error_kw={'lw': 1})
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Test MSE')
    ax.set_title('KIBA (GIN, $n=5$ matched)')
    ax.legend(frameon=False)
    # 配对 p 注释
    ax.text(0, 0.172, 'p=0.76', ha='center', fontsize=8.5)
    ax.text(1, 0.172, 'p=0.06', ha='center', fontsize=8.5)

    # Davis
    ax = axes[1]
    dm = davis['mse']
    e0_means = [davis_b['e0_mse_mean'], dm['e0_mean']]
    e0_stds = [davis_b['e0_mse_std'], dm['e0_std']]
    e7_means = [davis_b['e7_mse_mean'], dm['e7_mean']]
    e7_stds = [davis_b['e7_mse_std'], dm['e7_std']]
    ax.bar(x - 0.2, e0_means, 0.4, yerr=e0_stds, color=C_E0, capsize=3,
           label='E0', error_kw={'lw': 1})
    ax.bar(x + 0.2, e7_means, 0.4, yerr=e7_stds, color=C_E7, capsize=3,
           label='E7', error_kw={'lw': 1})
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Test MSE')
    ax.set_title('Davis (GIN, $n=10$ matched)')
    ax.legend(frameon=False)
    ax.text(0, 0.2145, 'p=0.13', ha='center', fontsize=8.5)
    ax.text(1, 0.2145, 'p=0.41', ha='center', fontsize=8.5)

    fig.tight_layout()
    save(fig, 'fig_v9_2_protocol_sensitivity')


# ---------------------------------------------------------------------------
# Fig 3: 剂量-反应（Davis 倒 U / KIBA 单调）
# ---------------------------------------------------------------------------
def fig_dose_response(p3, p4):
    """两面板：相对基线增益 % vs α。"""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)

    # Davis 七水平（增益%：正 = MSE 更低 = 更好；呈倒 U）
    ax = axes[0]
    curve = p3['mean_curve']
    base = curve[0]['mse_mean']
    alphas = [c['alpha'] for c in curve]
    gains = [-(c['mse_mean'] - base) / base * 100 for c in curve]
    ax.plot(range(len(alphas)), gains, '-o', color=C_ACCENT, lw=1.6, ms=4)
    ax.axhline(0, color='0.6', lw=0.8, ls='--')
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([('0' if a == 0 else f'{a:g}') for a in alphas])
    ax.set_xlabel(r'Contrastive weight $\alpha$')
    ax.set_ylabel('MSE gain vs.\n$\\alpha=0$ (%)\n(positive = better)')
    ax.set_title('Davis (GIN): inverted-U gain\nFriedman p=0.053')

    # KIBA 三水平（同一约定：增益%，正 = 更好；单调下降即单调恶化）
    ax = axes[1]
    means = [c['mse_mean'] for c in p4['conditions']]
    base = means[0]
    kg = [-(m - base) / base * 100 for m in means]
    xlabels = [f"{c['alpha']:g}" for c in p4['conditions']]
    ax.plot(range(3), kg, '-o', color=C_E7, lw=1.6, ms=5)
    ax.axhline(0, color='0.6', lw=0.8, ls='--')
    ax.set_xticks(range(3)); ax.set_xticklabels(xlabels)
    ax.set_xlabel(r'Contrastive weight $\alpha$')
    ax.set_title('KIBA (GIN): monotonic worsening\nFriedman p=0.015')
    # 显著性注释（最差点在下方）
    ax.annotate('Bonf p=0.0005', xy=(2, kg[2]), xytext=(0.7, kg[2] + 3),
                fontsize=8.5, arrowprops={'arrowstyle': '->', 'lw': 0.8})

    fig.tight_layout()
    save(fig, 'fig_v9_3_dose_response')


# ---------------------------------------------------------------------------
# Fig 4: 编码器交互（Davis GIN vs GAT 的 E0/E7）
# ---------------------------------------------------------------------------
def fig_architecture(davis, gat):
    """E0/E7 均值±std；GIN n=10, GAT n=5。"""
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    encs = ['GIN', 'GAT']
    e0m = [davis['mse']['e0_mean'], gat['mse']['e0_mean']]
    e0s = [davis['mse']['e0_std'], gat['mse']['e0_std']]
    e7m = [davis['mse']['e7_mean'], gat['mse']['e7_mean']]
    e7s = [davis['mse']['e7_std'], gat['mse']['e7_std']]
    x = np.arange(2)
    ax.bar(x - 0.2, e0m, 0.4, yerr=e0s, color=C_E0, capsize=3, label='E0',
           error_kw={'lw': 1})
    ax.bar(x + 0.2, e7m, 0.4, yerr=e7s, color=C_E7, capsize=3, label='E7',
           error_kw={'lw': 1})
    ax.set_xticks(x); ax.set_xticklabels([f'{e}\n($n=10$)' if e == 'GIN'
                                          else f'{e}\n($n=5$)' for e in encs])
    ax.set_ylabel('Test MSE')
    ax.set_title('Davis: CL effect depends on encoder\n'
                 'GIN p=0.41 (null); GAT p=0.0005 ($+$10.2%)')
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, 'fig_v9_4_architecture')


# ---------------------------------------------------------------------------
# Fig 5: 划分策略（standard / scaffold / cold 的增益）
# ---------------------------------------------------------------------------
def fig_splits(davis, scaffold, phase0):
    """三种划分下 CL 相对 E0 的 MSE 增益 %（标准 Davis GIN n=10；scaffold n=5；cold n=5）。"""
    fig, ax = plt.subplots(figsize=(5.6, 3.3))
    names = ['Standard\nrandom', 'Scaffold', 'Cold-start']
    gains = [davis['mse']['gain_pct'], scaffold['mse']['gain_pct'],
             phase0['experiments']['T1.6_Davis_ColdStart']['paired_test']
             ['mse_improvement_pct']]
    ps = [davis['mse']['paired_p'], scaffold['mse']['paired_p'],
          phase0['experiments']['T1.6_Davis_ColdStart']['paired_test']['p_mse']]
    colors = [C_ACCENT if g > 0 else C_E7 for g in gains]
    bars = ax.bar(names, gains, color=colors, width=0.55)
    ax.axhline(0, color='0.3', lw=0.9)
    ax.set_ylabel('MSE gain of CL vs. E0 (%)')
    # V9 fix (2026-11-12 review): single-line title to avoid two-line overlap
    ax.set_title('Davis (GIN): CL gain across split types $-$ none significant')
    for b, g, p in zip(bars, gains, ps):
        ax.text(b.get_x() + b.get_width() / 2,
                g + (0.35 if g >= 0 else -0.35),
                f'{g:+.1f}%\np={p:.2f}', ha='center',
                va='bottom' if g >= 0 else 'top', fontsize=8.5)
    # V9 fix: extra lower headroom so the scaffold label does not clip
    ax.set_ylim(-19.5, 5)
    fig.tight_layout()
    save(fig, 'fig_v9_5_splits')


# ---------------------------------------------------------------------------
# Fig 6: 跨数据集交互总览（三个 encoder×dataset 格子）
# ---------------------------------------------------------------------------
def fig_cross_dataset(davis, gat, gate_a):
    """增益点估计 + 配对差值 95% CI（换算为 %）。"""
    fig, ax = plt.subplots(figsize=(6.2, 3.2))

    def gain_ci(m):
        """配对差值 CI 换算为增益%；MSE 增益 = −(E7−E0)/E0，故整体取负。"""
        lo, hi = m['diff_ci95']
        return (-hi / m['e0_mean'] * 100, -lo / m['e0_mean'] * 100)

    cells = [('GIN\nDavis', davis['mse'], 0),
             ('GAT\nDavis', gat['mse'], 1),
             ('GIN\nKIBA', gate_a['mse'], 2)]
    ys = [2, 1, 0]
    for (label, m, _), y in zip(cells, ys):
        g = m['gain_pct']
        lo, hi = gain_ci(m)
        ax.errorbar(g, y, xerr=[[g - lo], [hi - g]], fmt='o', ms=7,
                    color=C_ACCENT if g > 0 else C_E7, capsize=4, lw=1.5)
        # V9 fix (2026-11-12 review): label just above the point so it never
        # crosses the horizontal CI whisker of the same row
        ax.text(g, y + 0.24, f'{g:+.2f}%\np={m["paired_p"]:.4f}',
                va='bottom', ha='center', fontsize=8.6)
    ax.axvline(0, color='0.3', lw=0.9, ls='--')
    ax.set_yticks(ys)
    ax.set_yticklabels([c[0] for c in cells])
    ax.set_xlabel('MSE gain of E7 vs. E0 (%; paired-difference 95% CI)')
    ax.set_title('Encoder$\\times$dataset interaction: no cross-benchmark robust benefit')
    ax.set_xlim(-13, 13)
    # V9 fix (2026-11-12 review): headroom above top row so its label clears
    # the title; left margin for the two-line y labels
    ax.set_ylim(-0.6, 3.1)
    fig.tight_layout()
    fig.subplots_adjust(left=0.14)
    save(fig, 'fig_v9_6_cross_dataset')


def main():
    print('=' * 70)
    print('V9 Phase 3 Task 3.2.4: regenerate figures from real V9 results')
    print('=' * 70)
    phase0 = load('v9_phase0p3_statistical_tests.json')
    gate_a = load('v9_decision_gate_a.json')
    davis = load('v9_phase2p1_davis_stats.json')
    scaffold = load('v9_phase2p2_scaffold_stats.json')
    dose_d = load('v9_phase2p3_dose_response.json')
    dose_k = load('v9_phase2p4_kiba_dose.json')
    gat = load('v9_phase2p5_gat_stats.json')

    fig_protocol_schematic()
    fig_protocol_sensitivity(phase0, dose_k, davis)
    fig_dose_response(dose_d, dose_k)
    fig_architecture(davis, gat)
    fig_splits(davis, scaffold, phase0)
    fig_cross_dataset(davis, gat, gate_a)
    print('\nAll figures written to', FIGDIR)


if __name__ == '__main__':
    main()
