#!/usr/bin/env python3
"""
E1.2: KIBA n=10 权威重跑——逐种子诊断与统计分析脚本
================================================================================
时间: 2026-10-08
任务来源: 《实验复核意见与下一阶段实验方案.md》实验E1步骤4 (R1/R2/R3)
目标:
  1. 汇总 KIBA α扫描(β=0.0) n=10 全部结果（原5种子{42,3407,2024,7,123}
     + 新5种子{251,1496,5934,7350,9889}），E0基线同步扩展至 n=10
  2. R2逐种子诊断: 未收敛(best_epoch顶到500) / 离群(|x-mean|>2sd) /
     异常高MSE(>0.5) 标记；α=0.1组重点核查（R2: n=5时sd=±0.1662, CV=48.6%）
  3. 统计检验: 各α vs E0 的 Welch t 检验(KIBA既定口径) + Holm(family=6)校正
     改善% = (E0_mean - α_mean)/E0_mean × 100 (Convention B: 正=改善)
  4. 批次效应检查: 原5种子与新5种子分组均值对照
  5. 敏感性分析: 剔除离群种子后结论是否翻转

数据装载原则(Round 22教训): 以 results.json 的 config 字段为准，不信目录名。
  - 处理组: GIN_KIBA_alpha*_seed* 中 config β=0.0 且 γ=0.0 的run
    (旧命名 alpha001/alpha005 = 原5种子，新命名 alpha1/alpha5 = 新5种子；
     alpha10/30/50/100 为新旧种子同前缀混合——分组完全按 config.alpha，
     批次按 config.seed 区分)
  - 显式排除: GIN_KIBA_alpha0_seed* (config实证 α=0.1, β=0.05, γ=0.05 污染)
  - E0基线: KIBA_E0_seed* (config.use_contrastive=False, augmentation=none)

输入: results/controlled/{KIBA_E0,GIN_KIBA_alpha*}_seed*/results.json
输出: results/analysis/e1_2_kiba_n10_analysis.json + 终端报告
判读预案(预先承诺, 防事后合理化):
  - α=0.1 n=10 Holm p<0.05 → Finding #5 保留, 样本量表述改n=10
  - 不显著但方向为正 → "方向一致但功效不足", 主结论让位符号反转
  - 方向为负 → 符号反转成为KIBA唯一结论
================================================================================
"""

import json
import glob
import numpy as np
from scipy import stats
from pathlib import Path
from collections import defaultdict

RESULTS = Path('/home/rxp/dta2/results/controlled')
OUT = Path('/home/rxp/dta2/results/analysis/e1_2_kiba_n10_analysis.json')

OLD_SEEDS = {42, 3407, 2024, 7, 123}
NEW_SEEDS = {251, 1496, 5934, 7350, 9889}
# 论文声明的KIBA检验族: 6个α配置 (§Testing protocol, Holm family=6)
ALPHA_ORDER = [0.001, 0.005, 0.01, 0.03, 0.05, 0.1]

report = {'script': 'scripts/e1_2_kiba_n10_analysis.py',
          'protocol_excluded': [], 'groups': {}, 'stats': {},
          'holm_family': len(ALPHA_ORDER)}


def protocol_check_treatment(c):
    """处理组协议断言: β=0, γ=0, lr=5e-4, aug=noise"""
    return (float(c.get('beta', -1)) == 0.0
            and float(c.get('gamma', -1)) == 0.0
            and abs(float(c.get('learning_rate', -1)) - 5e-4) < 1e-9
            and str(c.get('augmentation')) == 'noise'
            and bool(c.get('use_contrastive')) is True)


def protocol_check_e0(c):
    """E0基线协议断言: 纯监督, aug=none, lr=5e-4"""
    return (bool(c.get('use_contrastive')) is False
            and str(c.get('augmentation')) == 'none'
            and abs(float(c.get('learning_rate', -1)) - 5e-4) < 1e-9)


def load_group(pattern, check_fn, label):
    """装载一组实验，以config为准做协议断言。返回 {seed: record}"""
    rows = {}
    for f in sorted(glob.glob(str(RESULTS / pattern / 'results.json'))):
        d = Path(f).parent.name
        with open(f) as fh:
            r = json.load(fh)
        c = r.get('config', {})
        if not check_fn(c):
            report['protocol_excluded'].append(
                f'{d}: 协议不符(α={c.get("alpha")},β={c.get("beta")},'
                f'γ={c.get("gamma")},lr={c.get("learning_rate")},'
                f'aug={c.get("augmentation")},uc={c.get("use_contrastive")})')
            continue
        seed = c.get('seed')
        if seed in rows:
            report['protocol_excluded'].append(f'{d}: seed={seed}重复，忽略')
            continue
        if any(k not in r for k in ('test_mse', 'test_ci', 'best_epoch')):
            report['protocol_excluded'].append(f'{d}: 结果字段缺失，忽略')
            continue
        # CI口径: 优先 test_ci_standard (pair-counting, 论文口径, 2026-10-09修订);
        # 旧文件该字段缺失时回退 test_ci
        ci = r.get('test_ci_standard')
        if ci is None:
            ci = r['test_ci']
        rows[seed] = dict(dir=d, seed=seed, alpha=c.get('alpha'),
                          test_mse=r['test_mse'], test_ci=ci,
                          best_epoch=r['best_epoch'],
                          best_val_mse=r.get('best_val_mse'))
    return rows


def diagnose(rows, label):
    """R2逐种子诊断: 未收敛/离群/异常高MSE"""
    ms = np.array([r['test_mse'] for r in rows.values()])
    mean, sd = ms.mean(), ms.std(ddof=1) if len(ms) > 1 else 0.0
    flags = {}
    for seed, r in sorted(rows.items()):
        f = []
        if r['best_epoch'] >= 499:
            f.append('NOT_CONVERGED')
        if sd > 0 and abs(r['test_mse'] - mean) > 2 * sd:
            f.append('OUTLIER(>2sd)')
        if r['test_mse'] > 0.5:
            f.append('HIGH_MSE(>0.5)')
        if f:
            flags[seed] = f
    return flags, mean, sd


def welch_vs_baseline(treat_mse, base_mse):
    """Welch t检验 + Convention B改善%"""
    a, b = np.asarray(treat_mse), np.asarray(base_mse)
    t, p = stats.ttest_ind(a, b, equal_var=False)
    imp = (b.mean() - a.mean()) / b.mean() * 100
    return float(t), float(p), float(imp)


def holm_correction(pvals_dict):
    """Holm校正: p升序 × (m-rank)，running max，cap 1.0
    输入 {label: raw_p}，返回 {label: holm_p}"""
    items = sorted(pvals_dict.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted, running = {}, 0.0
    for rank, (label, p) in enumerate(items, start=1):
        adj = min(1.0, (m - rank + 1) * p)
        running = max(running, adj)
        adjusted[label] = running
    return adjusted


# ================= 1. 装载 =================
print('=' * 72)
print('E1.2: KIBA n=10 逐种子诊断与统计分析')
print('=' * 72)

e0 = load_group('KIBA_E0_seed*', protocol_check_e0, 'E0')
treat = {}
for pat in ('GIN_KIBA_alpha001_seed*', 'GIN_KIBA_alpha005_seed*',
            'GIN_KIBA_alpha0_seed*',   # 污染批次，协议断言会剔除
            'GIN_KIBA_alpha1_seed*', 'GIN_KIBA_alpha5_seed*',
            'GIN_KIBA_alpha10_seed*', 'GIN_KIBA_alpha30_seed*',
            'GIN_KIBA_alpha50_seed*', 'GIN_KIBA_alpha100_seed*'):
    for seed, r in load_group(pat, protocol_check_treatment, 'T').items():
        if r['alpha'] is None:
            report['protocol_excluded'].append(f'{r["dir"]}: config缺alpha，忽略')
            continue
        a = float(r['alpha'])
        if seed in treat.get(a, {}):
            report['protocol_excluded'].append(
                f'{r["dir"]}: α={a} seed={seed}重复，忽略')
            continue
        treat.setdefault(a, {})[seed] = r

print('\n[协议剔除记录]')
for x in report['protocol_excluded']:
    print(f'  - {x}')

# ================= 2. 逐种子诊断 (R2) =================
print('\n' + '=' * 72)
print('[R2 逐种子诊断] 标记: NOT_CONVERGED / OUTLIER(>2sd) / HIGH_MSE(>0.5)')
print('=' * 72)
diag = {}
for label, rows in ([('E0', e0)] + [(f'α={a}', treat.get(a, {}))
                                    for a in ALPHA_ORDER]):
    flags, mean, sd = diagnose(rows, label)
    diag[label] = flags
    n_old = sum(1 for s in rows if s in OLD_SEEDS)
    n_new = sum(1 for s in rows if s in NEW_SEEDS)
    print(f'\n{label}: n={len(rows)} (原{n_old}+新{n_new})  '
          f'MSE={mean:.4f}±{sd:.4f}' if len(rows) else f'\n{label}: 无数据')
    for seed, r in sorted(rows.items()):
        batch = '原' if seed in OLD_SEEDS else '新'
        flag_str = f'  ⚠ {",".join(flags.get(seed, []))}' if seed in flags else ''
        print(f'  [{batch}] seed={seed:<6} MSE={r["test_mse"]:.4f} '
              f'CI={r["test_ci"]:.4f} best_ep={r["best_epoch"]}{flag_str}')
    report['groups'][label] = dict(
        n=len(rows), n_old=n_old, n_new=n_new, mean=float(mean),
        sd=float(sd), flags={str(k): v for k, v in flags.items()},
        per_seed={str(s): dict(mse=r['test_mse'], ci=r['test_ci'],
                               epoch=r['best_epoch'])
                  for s, r in rows.items()})

# ================= 3. 批次效应检查 =================
print('\n' + '=' * 72)
print('[批次效应检查] 原5种子 vs 新5种子')
print('=' * 72)
batch_check = {}
for label, rows in ([('E0', e0)] + [(f'α={a}', treat.get(a, {}))
                                    for a in ALPHA_ORDER]):
    old = [r['test_mse'] for s, r in rows.items() if s in OLD_SEEDS]
    new = [r['test_mse'] for s, r in rows.items() if s in NEW_SEEDS]
    if old and new:
        t, p = stats.ttest_ind(new, old, equal_var=False)
        batch_check[label] = dict(old_mean=float(np.mean(old)),
                                  new_mean=float(np.mean(new)),
                                  welch_p=float(p))
        print(f'  {label:<8} 原={np.mean(old):.4f}  新={np.mean(new):.4f}  '
              f'Welch p={p:.4f}')
report['batch_effect'] = batch_check

# ================= 4. 统计检验 + Holm =================
print('\n' + '=' * 72)
print('[统计检验] 各α vs E0 (Welch t), Holm family=6, Convention B(正=改善)')
print('=' * 72)
e0_mse = [r['test_mse'] for r in e0.values()]
raw_p, stats_rows = {}, []
for a in ALPHA_ORDER:
    rows = treat.get(a, {})
    if len(rows) < 2 or len(e0_mse) < 2:
        print(f'  α={a}: 数据不足，跳过')
        continue
    t, p, imp = welch_vs_baseline([r['test_mse'] for r in rows.values()],
                                  e0_mse)
    raw_p[f'α={a}'] = p
    ms = np.array([r['test_mse'] for r in rows.values()])
    cis = np.array([r['test_ci'] for r in rows.values()])
    stats_rows.append((a, ms, cis, t, p, imp))

holm = holm_correction(raw_p)
print(f'\nE0 (n={len(e0_mse)}): MSE={np.mean(e0_mse):.4f}±{np.std(e0_mse, ddof=1):.4f}')
print(f'{"α":<7} {"n":>3} {"MSE":>18} {"CI":>18} {"改善%":>8} '
      f'{"raw p":>8} {"Holm p":>8} {"显著":>5}')
for a, ms, cis, t, p, imp in stats_rows:
    hp = holm[f'α={a}']
    sig = '✓' if hp < 0.05 else '—'
    print(f'{a:<7} {len(ms):>3} {f"{ms.mean():.4f}±{ms.std(ddof=1):.4f}":>18} '
          f'{f"{cis.mean():.4f}±{cis.std(ddof=1):.4f}":>18} '
          f'{imp:>+7.2f}% {p:>8.4f} {hp:>8.4f} {sig:>5}')
    report['stats'][f'α={a}'] = dict(
        n=len(ms), mse_mean=float(ms.mean()), mse_sd=float(ms.std(ddof=1)),
        ci_mean=float(cis.mean()), ci_sd=float(cis.std(ddof=1)),
        improvement_pct=float(imp), welch_t=float(t), raw_p=float(p),
        holm_p=float(hp), significant=bool(hp < 0.05))
report['E0'] = dict(n=len(e0_mse), mse_mean=float(np.mean(e0_mse)),
                    mse_sd=float(np.std(e0_mse, ddof=1)))

# ================= 5. 敏感性分析: 剔除离群种子 =================
print('\n' + '=' * 72)
print('[敏感性分析] 剔除被标记种子后重算')
print('=' * 72)
sensitivity = {}
for a in ALPHA_ORDER:
    rows = treat.get(a, {})
    flagged = set(diag.get(f'α={a}', {}))
    clean = {s: r for s, r in rows.items() if s not in flagged}
    if len(clean) >= 2 and len(clean) < len(rows):
        t, p, imp = welch_vs_baseline([r['test_mse'] for r in clean.values()],
                                      e0_mse)
        ms = np.array([r['test_mse'] for r in clean.values()])
        sensitivity[f'α={a}'] = dict(
            excluded=sorted(flagged), n=len(clean),
            mse_mean=float(ms.mean()), mse_sd=float(ms.std(ddof=1)),
            improvement_pct=float(imp), raw_p=float(p))
        print(f'  α={a}: 剔除{sorted(flagged)} → n={len(clean)} '
              f'MSE={ms.mean():.4f}±{ms.std(ddof=1):.4f} '
              f'改善{imp:+.2f}% raw p={p:.4f}')
    elif len(clean) == len(rows):
        print(f'  α={a}: 无标记种子，结论不变')
report['sensitivity'] = sensitivity

# ================= 6. 判读预案 =================
print('\n' + '=' * 72)
print('[判读预案] (预先承诺)')
print('=' * 72)
key = report['stats'].get('α=0.1')
if key:
    if key['significant'] and key['improvement_pct'] > 0:
        verdict = 'n=10显著改善 → Finding #5 保留，样本量表述改n=10'
    elif key['improvement_pct'] > 0:
        verdict = 'n=10不显著但方向为正 → "方向一致但功效不足"，主结论让位符号反转'
    else:
        verdict = 'n=10方向为负 → 符号反转成为KIBA唯一结论'
    print(f'  α=0.1: 改善{key["improvement_pct"]:+.2f}%, Holm p={key["holm_p"]:.4f}')
    print(f'  → {verdict}')
    report['verdict'] = verdict
else:
    print('  α=0.1数据缺失，无法判读')

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f'\n报告已保存: {OUT}')
