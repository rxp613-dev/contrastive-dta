#!/usr/bin/env python3
"""
V9 Phase 2 Task 2.4: KIBA E8 (α=0.5) @1000ep, Config C, 5种子
日期: 2026-11-10
目标: 用收敛协议 Config C 在 KIBA 标准随机划分上运行 CL 损失权重 α=0.5，
      补全 KIBA 剂量-反应曲线第三点（α=0 与 α=0.05 的 Config C 结果已在
      Phase 1 产出），复核高剂量 CL 在收敛基线下的作用。严格对应
      TODO_v9_phase1_2_3.md Task 2.4.1~2.4.4，不简化、不改目标。
历史溯源:
  - v9_t1p4_e7_alpha05_seed{42,7,123,456,789}: Config B @500ep，best_epoch
    472-493（贴上限边缘，截断时仍在改进）→ 被 v9 作废，需 Config C 重跑。
  - 更早 GIN_KIBA_alpha500_seed{42,7,123,2024,3407}: 亦 Config B。
方法: 顺序调用 scripts/train_controlled.py，除 α=0.5 外参数与 Phase 1 E7 一致：
  dataset=kiba, GIN, batch 512, lr 5e-4, wd 1e-4, max_epochs=1000,
  lr_patience=30, early_stop_patience=150, temperature=0.1, noise, β=γ=0。
种子: [42, 7, 123, 456, 789]（与 Phase 1 KIBA E0/E7 及 v9_t1p4 历史批次配对）
产出:
  - 每 run: results/controlled/v9_p2p4_kiba_a050_seed{seed}/results.json
  - 批次目录: results/controlled/v9_phase2_kiba_alpha05_{timestamp}/
      config.json / progress.json / summary.json
运行: GPU（自动 CUDA）顺序不并发；KIBA run 较慢，预计 ~22 GPU小时（5 run）。
原则: 真实实验出数；执行前须独立审核；不编造。
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime


SEEDS = [42, 7, 123, 456, 789]
REPO = '/mnt/data/dta2'


def build_cmd(seed):
    """构造 α=0.5 单种子 run 命令行。"""
    exp_id = f"v9_p2p4_kiba_a050_seed{seed}"
    return [
        sys.executable, 'scripts/train_controlled.py',
        '--experiment_id', exp_id,
        '--dataset', 'kiba',
        '--seed', str(seed),
        '--architecture', 'GIN',
        '--batch_size', '512',
        '--max_epochs', '1000',          # Config C
        '--lr_patience', '30',
        '--early_stop_patience', '150',
        '--num_workers', '4',
        '--learning_rate', '0.0005',
        '--weight_decay', '0.0001',
        '--use_contrastive',
        '--alpha', '0.5',
        '--beta', '0.0',
        '--gamma', '0.0',
        '--temperature', '0.1',
        '--augmentation', 'noise'
    ]


def run_one(seed):
    """运行单种子，返回 (success, returncode)。"""
    cmd = build_cmd(seed)
    print("=" * 80)
    print(f"α=0.5 seed={seed}  命令: {' '.join(cmd)}")
    print("=" * 80, flush=True)
    try:
        result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    except Exception as exc:
        print(f"❌ seed={seed} 调用异常: {exc}", flush=True)
        return False, -1

    print('\n'.join(result.stdout.split('\n')[-25:]), flush=True)
    if result.stderr.strip():
        print("--- stderr ---", flush=True)
        print(result.stderr[-2000:], flush=True)

    ok = (result.returncode == 0)
    print(f"{'✅ 完成' if ok else '❌ 失败'}: seed={seed} rc={result.returncode}", flush=True)
    return ok, result.returncode


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_dir = Path(REPO) / 'results/controlled' / f'v9_phase2_kiba_alpha05_{timestamp}'
    batch_dir.mkdir(parents=True, exist_ok=True)

    config_record = {
        'config_version': 'Config C', 'dataset': 'kiba', 'architecture': 'GIN',
        'batch_size': 512, 'learning_rate': 5e-4, 'weight_decay': 1e-4,
        'lr_patience': 30, 'early_stop_patience': 150, 'max_epochs': 1000,
        'contrastive': {'alpha': 0.5, 'beta': 0.0, 'gamma': 0.0,
                        'temperature': 0.1, 'augmentation': 'noise'},
        'seeds': SEEDS,
        'historical_reference': 'v9_t1p4_e7_alpha05 (Config B, best_epoch 472-493)',
        'task': 'V9 Phase 2 Task 2.4',
        'goal': 'KIBA alpha=0.5 @1000ep, 5 seeds; dose-response third point',
        'timestamp': timestamp
    }
    with open(batch_dir / 'config.json', 'w') as f:
        json.dump(config_record, f, indent=2)

    print("V9 Phase 2 Task 2.4: KIBA α=0.5 @1000ep, Config C", flush=True)
    print(f"批次目录: {batch_dir}", flush=True)

    records = []
    for i, seed in enumerate(SEEDS, 1):
        print(f"\n>>> 进度 {i}/{len(SEEDS)}", flush=True)
        ok, rc = run_one(seed)
        records.append({'seed': seed, 'success': ok, 'returncode': rc})
        with open(batch_dir / 'progress.json', 'w') as f:
            json.dump({'completed': i, 'total': len(SEEDS), 'records': records,
                       'last_update': datetime.now().isoformat()}, f, indent=2)

    n_ok = sum(1 for r in records if r['success'])
    with open(batch_dir / 'summary.json', 'w') as f:
        json.dump({'timestamp': timestamp, 'task': 'V9 Phase 2 Task 2.4',
                   'completed': n_ok, 'total': len(SEEDS), 'records': records},
                  f, indent=2)

    print("\n" + "=" * 80)
    print(f"Task 2.4 批次结束: {n_ok}/{len(SEEDS)} runs 成功")
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
