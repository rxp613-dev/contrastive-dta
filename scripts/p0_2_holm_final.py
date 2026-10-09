#!/usr/bin/env python3
"""
P0-2: KIBA Holm家族规模修正 — 最终版
时间: 2026-09-09
目标:
  1. 从原始per-seed数据计算Welch t-test + Holm(family=6)
  2. 产出Table 16正确的p值
  3. 产出§3.8修正文本
方法:
  - scipy.stats.ttest_ind (Welch, equal_var=False, 双尾)
  - statsmodels.stats.multitest.multipletests (Holm)
  - 数据来源: results/controlled/GIN_KIBA_alpha* 和 KIBA_E0_lr0.0001_seed*
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests

def get_per_seed_mse(pattern):
    base = Path("/home/rxp/dta2/results/controlled")
    mses = []
    for d in sorted(base.glob(pattern)):
        with open(d / 'results.json') as f:
            data = json.load(f)
        mses.append(data.get('test_mse'))
    return np.array(mses)

def main():
    print("=" * 80)
    print("P0-2: KIBA Holm(family=6) 最终重算")
    print("=" * 80)

    # 1. 读取per-seed数据
    e0_mses = get_per_seed_mse("KIBA_E0_lr0.0001_seed*")
    
    alpha_patterns = {
        0.001: "GIN_KIBA_alpha001_seed*",
        0.005: "GIN_KIBA_alpha005_seed*",
        0.010: "GIN_KIBA_alpha10_seed*",
        0.030: "GIN_KIBA_alpha30_seed*",
        0.050: "GIN_KIBA_alpha50_seed*",
        0.100: "GIN_KIBA_alpha100_seed*",
    }
    
    per_seed = {}
    for alpha, pattern in alpha_patterns.items():
        per_seed[alpha] = get_per_seed_mse(pattern)
    
    # 2. Welch t-test (双尾)
    raw_pvals = {}
    for a in sorted(per_seed.keys()):
        t, p = stats.ttest_ind(per_seed[a], e0_mses, equal_var=False)
        raw_pvals[a] = p
    
    # 3. Holm校正 (family=6)
    alphas = sorted(raw_pvals.keys())
    pvals_list = [raw_pvals[a] for a in alphas]
    reject, pvals_holm, _, _ = multipletests(pvals_list, alpha=0.05, method='holm')
    
    # 4. 计算effect size (Hedges' g) 和百分比变化
    e0_mean = e0_mses.mean()
    
    print("\n" + "=" * 80)
    print("Table 16 正确值 (Holm family=6):")
    print("=" * 80)
    print(f"{'α':<8} {'Test MSE':<20} {'vs E0':<12} {'Holm p':<12} {'Sig?'}")
    print("-" * 70)
    
    table16_correct = {}
    for i, a in enumerate(alphas):
        cl_mean = per_seed[a].mean()
        cl_std = per_seed[a].std(ddof=1)
        pct = (e0_mean - cl_mean) / e0_mean * 100
        p_holm = pvals_holm[i]
        sig = "*" if reject[i] else ""
        
        print(f"{a:<8.3f} {cl_mean:.4f}±{cl_std:.4f}  {pct:>+7.2f}%   {p_holm:<12.6f} {sig}")
        
        table16_correct[a] = {
            'mean': float(cl_mean),
            'std': float(cl_std),
            'pct_change': float(pct),
            'holm_p': float(p_holm),
            'significant': bool(reject[i]),
        }
    
    # 5. 与Table 16当前值比对
    table16_current = {
        0.001: 0.008, 0.005: 0.008, 0.010: 0.063,
        0.030: 0.063, 0.050: 0.218, 0.100: 0.350,
    }
    
    print("\n" + "=" * 80)
    print("差异分析:")
    print("=" * 80)
    print(f"{'α':<8} {'当前p值':<12} {'正确p值':<12} {'差异':<12}")
    for a in alphas:
        curr = table16_current[a]
        corr = table16_correct[a]['holm_p']
        print(f"{a:<8.3f} {curr:<12.6f} {corr:<12.6f} {abs(curr-corr):<12.6f}")
    
    # 6. 裁决
    print("\n" + "=" * 80)
    print("裁决:")
    print("=" * 80)
    print("Table 16当前p值与Holm(family=6)重算结果不一致。")
    print("当前p值来源不明,可能来自旧版n=10数据或其他已废弃的计算。")
    print("")
    print("修正方案:")
    print("  1. §3.8: 'family size = 4' → 'family size = 6'")
    print("  2. Table 16: 更新所有p值为Holm(family=6)正确值")
    print("  3. 正文引用: 更新所有引用Table 16 p值的段落")
    
    # 7. 检查正文中引用Table 16 p值的位置
    print("\n" + "=" * 80)
    print("正文中引用Table 16 p值的位置:")
    print("=" * 80)
    
    paper = Path("/home/rxp/dta2/paper/main_paper.tex").read_text()
    import re
    
    # 查找p=0.008, p=0.063, p=0.218, p=0.350
    for p_val in [0.008, 0.063, 0.218, 0.350]:
        pattern = f"p\\s*=\\s*{p_val}"
        for m in re.finditer(pattern, paper):
            line_num = paper[:m.start()].count('\n') + 1
            start = max(0, m.start() - 50)
            end = min(len(paper), m.end() + 50)
            context = paper[start:end].replace('\n', ' ')
            print(f"  Line {line_num}: p={p_val} → {context}")
    
    # 8. 保存结果
    out = {
        'timestamp': '2026-09-09',
        'method': 'Welch t-test (2-sided) + Holm correction, family=6',
        'e0_baseline': {
            'mean': float(e0_mean),
            'std': float(e0_mses.std(ddof=1)),
            'n': len(e0_mses),
        },
        'holm_results': table16_correct,
        'corrections_needed': {
            'section_3_8': 'family size = 4 → family size = 6',
            'table_16_pvalues': {str(a): table16_correct[a]['holm_p'] for a in alphas},
        },
    }
    out_path = Path("/home/rxp/dta2/results/analysis/p0_2_holm_final.json")
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n结果保存: {out_path}")

if __name__ == "__main__":
    main()
