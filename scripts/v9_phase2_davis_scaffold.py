#!/usr/bin/env python3
"""
V9 Phase 2 Task 2.2: Davis scaffold 划分 E0 + E7 @1000ep, Config C, 各5种子
日期: 2026-11-09
目标: 用收敛协议 Config C 重跑被作废(Config B, 500ep)的 Davis scaffold 实验。
      历史批次 e3_scaffold_E0/E7a05 (results/cold_start/) 在 Config B 下 5 种子
      配对给出 CL 比基线差 38.22%（配对 p≈0.075）；Task 2.2 用相同 5 种子在
      Config C 下复核该负向结果是否复现。严格对应 TODO 2.2.1~2.2.4，不简化。
方法: 顺序调用 scripts/train_controlled.py，统一传 --split_type scaffold：
        阶段一 E0 ×5（无对比学习）
        阶段二 E7 ×5（--use_contrastive --alpha 0.05 --temperature 0.1 --augmentation noise）
      注意: train_controlled.py 对 split_type 非 None 的 run 将结果写到
            results/cold_start/{experiment_id}/，故本批次结果目录在 cold_start 下。
配置 (与 Task 2.1 相同，仅划分与种子数不同):
  dataset=davis, GIN, batch 512, lr 5e-4, wd 1e-4,
  max_epochs=1000 (Config C), lr_patience=30, early_stop_patience=150, num_workers=4
种子: [42, 7, 123, 2024, 3407]（与 e3_scaffold 历史批次一致，E0/E7 对称）
产出:
  - 每 run: results/cold_start/v9_p2_scaffold_{e0,e7}_seed{seed}/results.json
  - 批次元数据: results/cold_start/v9_phase2_scaffold_{timestamp}/
      config.json / progress_e0.json / progress_e7.json / summary.json
运行: GPU（自动 CUDA）；预计数小时（历史 run 多在 epoch <175 提前停止）。
原则: 真实实验出数；执行前须独立审核。
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime


SEEDS = [42, 7, 123, 2024, 3407]
REPO = '/mnt/data/dta2'


def build_cmd(group, seed):
    """构造单个 run 命令；group: 'e0' | 'e7'。"""
    exp_id = f"v9_p2_scaffold_{group}_seed{seed}"
    cmd = [
        sys.executable, 'scripts/train_controlled.py',
        '--experiment_id', exp_id,
        '--dataset', 'davis',
        '--split_type', 'scaffold',
        '--seed', str(seed),
        '--architecture', 'GIN',
        '--batch_size', '512',
        '--max_epochs', '1000',          # Config C
        '--lr_patience', '30',
        '--early_stop_patience', '150',
        '--num_workers', '4',
        '--learning_rate', '0.0005',
        '--weight_decay', '0.0001',
    ]
    if group == 'e7':
        cmd += [
            '--use_contrastive',
            '--alpha', '0.05',
            '--beta', '0.0',
            '--gamma', '0.0',
            '--temperature', '0.1',
            '--augmentation', 'noise'
        ]
    return cmd


def run_one(group, seed):
    """运行单个种子，返回 (success, returncode)。"""
    cmd = build_cmd(group, seed)
    print("=" * 80)
    print(f"[{group.upper()}] seed={seed}  命令: {' '.join(cmd)}")
    print("=" * 80, flush=True)
    try:
        result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    except Exception as exc:
        print(f"❌ seed={seed} 调用异常: {exc}", flush=True)
        return False, -1

    print('\n'.join(result.stdout.split('\n')[-30:]), flush=True)
    if result.stderr.strip():
        print("--- stderr ---", flush=True)
        print(result.stderr[-2000:], flush=True)

    ok = (result.returncode == 0)
    print(f"{'✅ 完成' if ok else '❌ 失败'}: {group} seed={seed} rc={result.returncode}", flush=True)
    return ok, result.returncode


def run_group(group, batch_dir):
    """运行一个组的全部种子并逐步写 progress。"""
    records = []
    for i, seed in enumerate(SEEDS, 1):
        print(f"\n>>> 进度 {group}: {i}/{len(SEEDS)}", flush=True)
        ok, rc = run_one(group, seed)
        records.append({'seed': seed, 'success': ok, 'returncode': rc})
        with open(batch_dir / f'progress_{group}.json', 'w') as f:
            json.dump({'group': group, 'completed': i, 'total': len(SEEDS),
                       'records': records, 'last_update': datetime.now().isoformat()},
                      f, indent=2)
    return records


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 批次元数据与 run 结果同放 cold_start（trainer 对 scaffold 的输出位置）
    batch_dir = Path(REPO) / 'results/cold_start' / f'v9_phase2_scaffold_{timestamp}'
    batch_dir.mkdir(parents=True, exist_ok=True)

    config_record = {
        'config_version': 'Config C', 'dataset': 'davis', 'split_type': 'scaffold',
        'architecture': 'GIN', 'batch_size': 512, 'learning_rate': 5e-4,
        'weight_decay': 1e-4, 'lr_patience': 30, 'early_stop_patience': 150,
        'max_epochs': 1000,
        'e7_extra': {'use_contrastive': True, 'alpha': 0.05, 'beta': 0.0,
                     'gamma': 0.0, 'temperature': 0.1, 'augmentation': 'noise'},
        'seeds': SEEDS, 'task': 'V9 Phase 2 Task 2.2',
        'goal': 'Davis scaffold E0 + E7 @1000ep, 5 matched seeds',
        'historical_reference': 'e3_scaffold_* (Config B): CL -38.22%, paired p~0.075',
        'timestamp': timestamp
    }
    with open(batch_dir / 'config.json', 'w') as f:
        json.dump(config_record, f, indent=2)

    print("V9 Phase 2 Task 2.2: Davis scaffold E0+E7 @1000ep, Config C", flush=True)
    print(f"批次目录: {batch_dir}", flush=True)

    rec_e0 = run_group('e0', batch_dir)
    rec_e7 = run_group('e7', batch_dir)

    summary = {'timestamp': timestamp, 'task': 'V9 Phase 2 Task 2.2',
               'e0_completed': sum(1 for r in rec_e0 if r['success']),
               'e7_completed': sum(1 for r in rec_e7 if r['success']),
               'total': len(SEEDS) * 2, 'records_e0': rec_e0, 'records_e7': rec_e7}
    with open(batch_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Task 2.2 批次结束: E0 {summary['e0_completed']}/5, E7 {summary['e7_completed']}/5")
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
