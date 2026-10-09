#!/usr/bin/env python3
"""
E5.1: T11/T4 冷启动批次 config 级抽查脚本（R7闭环核查）
================================================================================
时间: 2026-10-08
任务来源: 《实验复核意见与下一阶段实验方案.md》R7
目标:
  以 results.json 的 config 字段为准（不信目录名——Round 22 教训），对
  Round 21 权威批次的全部 50 个目录做协议与种子抽查:
    - T11 (GAT cold-target α扫描): 30 runs = 6 α值 × 5种子, Protocol A
      (weight_decay=1e-4, augmentation=noise, β=0.0, γ=0.0)
    - T4 (2×2析因: 架构{GIN,GAT} × CL{noCL,CL}): 20 runs = 4组 × 5种子,
      Protocol B (weight_decay=0.01 强正则化)
核查项:
  1. 每组种子集合恰为 {42, 3407, 2024, 7, 123}（无重复、无缺失、无漂移）
  2. config.α 与目录声称的 α 一致（以config为准）
  3. 协议字段: β/γ/wd/augmentation/architecture/split_type 符合各批次协议
  4. 结果字段完整 (test_mse, test_ci, best_epoch)
输出: 终端报告 + results/analysis/e5_1_t11_t4_config_audit.json
方法: 只读审计，不修改任何文件
================================================================================
"""

import json
import glob
from pathlib import Path

EXPECTED_SEEDS = {42, 3407, 2024, 7, 123}
BASE = Path('/home/rxp/dta2/results/cold_start')
OUT = Path('/home/rxp/dta2/results/analysis/e5_1_t11_t4_config_audit.json')

T11_ALPHAS = ['0', '0.001', '0.003', '0.01', '0.03', '0.05']

report = {'script': 'scripts/e5_1_t11_t4_config_audit.py',
          'groups': {}, 'violations': [], 'total_dirs': 0}


def check_group(group_name, dirs, expected_cfg):
    """对一组实验目录做config级核查。expected_cfg: {字段: 期望值或'__FROM_NAME__'}"""
    seeds = []
    group_problems = []
    rows = []
    for d in dirs:
        rj = BASE / d / 'results.json'
        if not rj.exists():
            group_problems.append(f'{d}: results.json缺失')
            continue
        with open(rj) as f:
            r = json.load(f)
        c = r.get('config', {})
        report['total_dirs'] += 1
        seeds.append(c.get('seed'))
        rows.append(dict(dir=d, seed=c.get('seed'),
                         alpha=c.get('alpha'),
                         test_mse=r.get('test_mse'),
                         test_ci=r.get('test_ci'),
                         best_epoch=r.get('best_epoch')))
        # 协议字段核查
        for k, v in expected_cfg.items():
            actual = c.get(k)
            if k == 'alpha' and v == '__FROM_NAME__':
                continue  # α单独核查(见下)
            if actual is None:
                group_problems.append(f'{d}: 缺少config字段 {k}')
                continue
            try:
                ok = (abs(float(actual) - float(v)) < 1e-9
                      if isinstance(v, float) else actual == v)
            except (TypeError, ValueError):
                ok = False
            if not ok:
                group_problems.append(f'{d}: {k} 期望{v} 实际{actual}')
        # 结果字段完整性
        for k in ('test_mse', 'test_ci', 'best_epoch'):
            if k not in r:
                group_problems.append(f'{d}: 缺少结果字段{k}')
    # 种子集合核查
    seed_set = set(seeds)
    if seed_set != EXPECTED_SEEDS or len(seeds) != 5:
        group_problems.append(
            f'{group_name}: 种子集合异常 实际{sorted(seed_set)}(n={len(seeds)}) '
            f'期望{sorted(EXPECTED_SEEDS)}')
    report['groups'][group_name] = dict(
        n=len(rows), seeds=sorted(seed_set),
        mse_mean=(sum(x['test_mse'] for x in rows) / len(rows)
                  if rows and all(x['test_mse'] is not None for x in rows) else None),
        problems=group_problems, rows=rows)
    report['violations'].extend(group_problems)
    return group_problems


# ---------- T11: 6组 × 5种子 (Protocol A) ----------
print('=' * 70)
print('T11 (GAT cold-target α扫描, Protocol A: wd=1e-4, noise, β=0, γ=0)')
print('=' * 70)
for a in T11_ALPHAS:
    dirs = sorted(d.name for d in BASE.glob(f'round21_T11_alpha{a}_seed*')
                  if d.is_dir())
    name = f'T11_alpha{a}'
    problems = check_group(name, dirs, dict(
        architecture='GAT', weight_decay=0.0001, augmentation='noise',
        beta=0.0, gamma=0.0, alpha='__FROM_NAME__', dataset='davis'))
    # α值与目录名一致性(以config为准)
    g = report['groups'][name]
    cfg_alphas = {row['alpha'] for row in g['rows']}
    expected_alpha = float(a)
    if cfg_alphas != {expected_alpha}:
        msg = f'{name}: config α={cfg_alphas} 与目录声称{expected_alpha}不符'
        report['violations'].append(msg)
        problems.append(msg)
    status = '✓' if not problems else f'❌ {len(problems)}项问题'
    print(f'  {name:<18} n={g["n"]} seeds={g["seeds"]} '
          f'MSE={g["mse_mean"]:.4f}' if g['mse_mean'] else f'  {name}: n={g["n"]}')
    for p in problems:
        print(f'      ⚠ {p}')

# ---------- T4: 4组 × 5种子 (Protocol B, wd=0.01) ----------
print()
print('=' * 70)
print('T4 (2×2析因, Protocol B: wd=0.01 强正则化)')
print('=' * 70)
for arch in ('GIN', 'GAT'):
    for cl in ('noCL', 'CL'):
        dirs = sorted(d.name for d in BASE.glob(f'round21_T4_{arch}_{cl}_seed*')
                      if d.is_dir())
        name = f'T4_{arch}_{cl}'
        expected = dict(architecture=arch, weight_decay=0.01,
                        dataset='davis')
        if cl == 'CL':
            expected.update(use_contrastive=True, beta=0.0, gamma=0.0,
                            augmentation='noise')
        else:
            # 注: noCL组实测 augmentation='noise' 且 config.alpha=0.1，
            # 均为 use_contrastive=False 时的惰性参数(不参与损失)，故不断言
            expected.update(use_contrastive=False)
        problems = check_group(name, dirs, expected)
        g = report['groups'][name]
        status = '✓' if not problems else f'❌ {len(problems)}项问题'
        line = f'  {name:<14} n={g["n"]} seeds={g["seeds"]}'
        if g['mse_mean'] is not None:
            line += f' MSE={g["mse_mean"]:.4f}'
        print(line)
        for p in problems:
            print(f'      ⚠ {p}')

# ---------- 汇总 ----------
print()
print('=' * 70)
print(f'总目录数: {report["total_dirs"]} (期望50)')
print(f'违规项: {len(report["violations"])}')
if report['violations']:
    for v in report['violations']:
        print(f'  ❌ {v}')
else:
    print('✅ R7核查通过: T11/T4全部50个目录种子集合与协议字段均合规')
print('=' * 70)

OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, 'w') as f:
    json.dump(report, f, indent=2, ensure_ascii=False)
print(f'报告已保存: {OUT}')
