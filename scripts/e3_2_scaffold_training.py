#!/usr/bin/env python3
"""
E3.2: Davis scaffold 冷启动划分训练驱动脚本 (α∈{0,0.05} × 5种子 = 10 runs)
================================================================================
时间: 2026-10-08
任务来源: 《实验复核意见与下一阶段实验方案.md》实验E3
目标:
  在 Bemis-Murcko scaffold 划分(真正 scaffold cold-drug)上检验对比学习收益
  是否保持: E0(纯监督) vs E7(α=0.05, 仅InfoNCE), 各5种子，配对比较。
  统计口径沿用论文: 改善% (Convention B) + 配对t + (报告)Holm。
背景(为何不用已有 cold_scaffold_E0/E7_seed{42,3407,2024}):
  既有6个scaffold实验为旧协议(β=0.05, γ=0.05, α=0.003)，不满足Finding 1
  协议(β=0.0, γ=0.0)，且α≠0.05，种子仅3个 —— 全部弃用，重新跑10个。

协议(与Round 21权威冷启动协议A一致):
  dataset=davis, split_type=scaffold, architecture=GIN(498,177参数),
  lr=5e-4, wd=1e-4, bs=512, ep=500, patience=50, temp=0.1, joint
  E0: 不传--use_contrastive, aug=none
  E7: --use_contrastive --alpha 0.05 --beta 0.0 --gamma 0.0, aug=noise
种子: {42, 3407, 2024, 7, 123} (项目标准五种子)
命名(避免与旧协议实验目录冲突):
  e3_scaffold_E0_seed{X} / e3_scaffold_E7a05_seed{X}
  (输出至 results/cold_start/, 由 train_controlled.py 按 split_type 路由)

方法: subprocess调用 scripts/train_controlled.py；每个run完成后回读
  results.json 做协议断言；幂等跳过；GPU训练。
输入特征: data/features/davis_scaffold_{train,val,test}.pt (已存在)
输出: results/cold_start/e3_scaffold_*/ + results/analysis/e3_2_summary.json
================================================================================
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = '/home/rxp/dta2'
TRAIN_SCRIPT = f'{PROJECT_ROOT}/scripts/train_controlled.py'
RESULTS_DIR = Path(f'{PROJECT_ROOT}/results/cold_start')
SUMMARY_FILE = Path(f'{PROJECT_ROOT}/results/analysis/e3_2_scaffold_training_summary.json')
TIMEOUT_PER_RUN = 7200

SEEDS = [42, 3407, 2024, 7, 123]

PROTOCOL = dict(
    dataset='davis',
    split_type='scaffold',
    architecture='GIN',
    learning_rate='0.0005',
    weight_decay='0.0001',
    batch_size='512',
    max_epochs='500',
    early_stop_patience='50',
    temperature='0.1',
    training_strategy='joint',
)

# (experiment_id模板, use_contrastive, alpha, augmentation)
GROUPS = [
    ('e3_scaffold_E0_seed{}', False, None, 'none'),
    ('e3_scaffold_E7a05_seed{}', True, 0.05, 'noise'),
]


def build_command(exp_id, seed, uc, alpha, aug):
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        '--experiment_id', exp_id,
        '--dataset', PROTOCOL['dataset'],
        '--split_type', PROTOCOL['split_type'],
        '--architecture', PROTOCOL['architecture'],
        '--seed', str(seed),
        '--learning_rate', PROTOCOL['learning_rate'],
        '--weight_decay', PROTOCOL['weight_decay'],
        '--batch_size', PROTOCOL['batch_size'],
        '--max_epochs', PROTOCOL['max_epochs'],
        '--early_stop_patience', PROTOCOL['early_stop_patience'],
        '--temperature', PROTOCOL['temperature'],
        '--augmentation', aug,
        '--mask_ratio', '0.2',
        '--training_strategy', PROTOCOL['training_strategy'],
    ]
    if uc:
        cmd += ['--use_contrastive', '--alpha', str(alpha),
                '--beta', '0.0', '--gamma', '0.0']
    return cmd


def verify(exp_id, seed, uc, alpha, aug):
    """完成后回读 results.json 做协议断言。返回 (ok, msg)"""
    rj = RESULTS_DIR / exp_id / 'results.json'
    if not rj.exists():
        return False, 'results.json不存在'
    try:
        with open(rj) as f:
            r = json.load(f)
    except Exception as e:
        return False, f'解析失败: {e}'
    c = r.get('config', {})
    checks = [
        (c.get('seed') == seed, f"seed {c.get('seed')}≠{seed}"),
        (c.get('dataset') == 'davis', f"dataset {c.get('dataset')}"),
        (str(c.get('split_type')) == 'scaffold', f"split_type {c.get('split_type')}"),
        (abs(float(c.get('learning_rate', -1)) - 5e-4) < 1e-9,
         f"lr {c.get('learning_rate')}"),
        (abs(float(c.get('weight_decay', -1)) - 1e-4) < 1e-9,
         f"wd {c.get('weight_decay')}"),
        (str(c.get('augmentation')) == aug, f"aug {c.get('augmentation')}"),
        (bool(c.get('use_contrastive')) is uc, f"uc {c.get('use_contrastive')}"),
        (str(c.get('architecture')) == 'GIN', f"arch {c.get('architecture')}"),
        (str(c.get('training_strategy')) == 'joint',
         f"strategy {c.get('training_strategy')}"),
        (int(c.get('batch_size', -1)) == 512, f"bs {c.get('batch_size')}"),
        (int(c.get('max_epochs', -1)) == 500, f"epochs {c.get('max_epochs')}"),
    ]
    if uc:
        checks += [
            (abs(float(c.get('alpha', -1)) - alpha) < 1e-9, f"alpha {c.get('alpha')}"),
            (float(c.get('beta', -1)) == 0.0, f"beta {c.get('beta')}"),
            (float(c.get('gamma', -1)) == 0.0, f"gamma {c.get('gamma')}"),
        ]
    for k in ('test_mse', 'test_ci', 'best_epoch'):
        checks.append((k in r, f'缺字段{k}'))
    bad = [m for ok, m in checks if not ok]
    if bad:
        return False, '; '.join(bad)
    return True, f"OK mse={r['test_mse']:.4f} ci={r['test_ci']:.4f} ep={r['best_epoch']}"


def main():
    print('=' * 70)
    print('E3.2: scaffold划分 E0 vs E7(α=0.05) 训练 (10 runs)')
    print(f'开始: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 70)

    assert Path(TRAIN_SCRIPT).exists()
    import torch
    assert torch.cuda.is_available(), 'GPU不可用'
    print(f'GPU: {torch.cuda.get_device_name(0)} ✓')

    results = []
    for tpl, uc, alpha, aug in GROUPS:
        for seed in SEEDS:
            exp_id = tpl.format(seed)
            print(f"\n{'─'*70}\n实验: {exp_id}\n{'─'*70}")
            ok, msg = verify(exp_id, seed, uc, alpha, aug)
            if ok:
                print(f'⚠️  已存在且合规，跳过: {msg}')
                results.append(dict(exp_id=exp_id, status='skipped', detail=msg))
                continue
            cmd = build_command(exp_id, seed, uc, alpha, aug)
            print(f'命令: {" ".join(cmd)}')
            t0 = datetime.now()
            try:
                proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True,
                                      text=True, timeout=TIMEOUT_PER_RUN)
            except subprocess.TimeoutExpired:
                print('⏱️ 超时')
                results.append(dict(exp_id=exp_id, status='timeout'))
                continue
            el = (datetime.now() - t0).total_seconds()
            if proc.returncode != 0:
                print(f'❌ 失败: {proc.stderr[-500:]}')
                results.append(dict(exp_id=exp_id, status='failed',
                                    detail=proc.stderr[-500:]))
                continue
            ok, msg = verify(exp_id, seed, uc, alpha, aug)
            print(f'{"✓" if ok else "❌协议断言"}: {msg} ({el/60:.1f}min)')
            results.append(dict(exp_id=exp_id,
                                status='success' if ok else 'protocol_violation',
                                detail=msg, elapsed_min=round(el / 60, 1)))

    summary = dict(script='scripts/e3_2_scaffold_training.py',
                   timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                   protocol=PROTOCOL, seeds=SEEDS, total=len(results),
                   success=sum(1 for r in results if r['status'] in ('success', 'skipped')),
                   failed=sum(1 for r in results if r['status'] not in ('success', 'skipped')),
                   results=results)
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_FILE, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*70}\n完成 {summary['success']}/{summary['total']}, "
          f"失败 {summary['failed']}\n汇总: {SUMMARY_FILE}\n{'='*70}")
    sys.exit(0 if summary['failed'] == 0 else 1)


if __name__ == '__main__':
    main()
