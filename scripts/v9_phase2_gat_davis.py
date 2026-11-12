#!/usr/bin/env python3
"""
V9 Phase 2 Task 2.5: GAT E0 + E7 Davis 标准随机划分 @1000ep, Config C, 各5种子
日期: 2026-11-11
目标: 用收敛协议 Config C 在 Davis 上重跑 GAT 编码器（替代 GIN）的基线与对比学习
      条件，复核论文标题中 "multiple encoders" 主张是否在基线收敛后仍然存活：
        - 若 GAT 上 E7 显著优于 E0 → 编码器泛化主张存活
        - 若无显著差异/反向 → 收窄标题中 encoders 主张
      严格对应 TODO_v9_phase1_2_3.md Task 2.5.1~2.5.5，不简化、不改目标。
历史溯源:
  - 旧 GAT_E0 / GAT_E7_alpha* 批次为早期实验：early_stop_patience=50、config 字段
    不完整、test_ci 记录的是已确认有 bug 的值（文件内另有 test_ci_standard）→ 全部
    不具可比性，不能作为 encoders 主张依据，需 Config C 完整重跑。
  - GATControlledDTA 参数量 521,345（与 GIN 498,177 不同属正常）。
方法: 顺序调用 scripts/train_controlled.py --architecture GAT：
  阶段一 E0 ×5（无对比学习）；阶段二 E7 ×5（α=0.05, temp=0.1, noise, β=γ=0）。
  其余超参与 GIN Davis 批次一致：batch 512, lr 5e-4, wd 1e-4, max_epochs=1000,
  lr_patience=30, early_stop_patience=150。
种子: [42, 7, 123, 456, 789]（与 Phase 2 其他任务一致，E0/E7 对称）
产出:
  - 每 run: results/controlled/v9_p2p5_gat_{e0,e7}_seed{seed}/results.json
  - 批次目录: results/controlled/v9_phase2_gat_davis_{timestamp}/
      config.json / progress_e0.json / progress_e7.json / summary.json
运行: GPU（自动 CUDA）顺序不并发；预计 ~11 GPU小时（10 run）。
原则: 真实实验出数；执行前须独立审核；不编造。
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime


SEEDS = [42, 7, 123, 456, 789]
REPO = '/mnt/data/dta2'


def build_cmd(group, seed):
    """构造单个 run 命令；group: 'e0' | 'e7'。"""
    exp_id = f"v9_p2p5_gat_{group}_seed{seed}"
    cmd = [
        sys.executable, 'scripts/train_controlled.py',
        '--experiment_id', exp_id,
        '--dataset', 'davis',
        '--seed', str(seed),
        '--architecture', 'GAT',          # Task 2.5 关键：GAT 编码器
        '--batch_size', '512',
        '--max_epochs', '1000',           # Config C
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
    """运行单种子，返回 (success, returncode)。"""
    cmd = build_cmd(group, seed)
    print("=" * 80)
    print(f"[{group.upper()}] seed={seed}  命令: {' '.join(cmd)}")
    print("=" * 80, flush=True)
    try:
        result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    except Exception as exc:
        print(f"❌ {group} seed={seed} 调用异常: {exc}", flush=True)
        return False, -1

    print('\n'.join(result.stdout.split('\n')[-25:]), flush=True)
    if result.stderr.strip():
        print("--- stderr ---", flush=True)
        print(result.stderr[-2000:], flush=True)

    ok = (result.returncode == 0)
    print(f"{'✅ 完成' if ok else '❌ 失败'}: {group} seed={seed} rc={result.returncode}", flush=True)
    return ok, result.returncode


def run_group(group, batch_dir):
    """运行一组全部种子并逐步写 progress。"""
    records = []
    for i, seed in enumerate(SEEDS, 1):
        print(f"\n>>> 进度 {group}: {i}/{len(SEEDS)}", flush=True)
        ok, rc = run_one(group, seed)
        records.append({'seed': seed, 'success': ok, 'returncode': rc})
        with open(batch_dir / f'progress_{group}.json', 'w') as f:
            json.dump({'group': group, 'completed': i, 'total': len(SEEDS),
                       'records': records,
                       'last_update': datetime.now().isoformat()}, f, indent=2)
    return records


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_dir = Path(REPO) / 'results/controlled' / f'v9_phase2_gat_davis_{timestamp}'
    batch_dir.mkdir(parents=True, exist_ok=True)

    config_record = {
        'config_version': 'Config C', 'dataset': 'davis', 'architecture': 'GAT',
        'batch_size': 512, 'learning_rate': 5e-4, 'weight_decay': 1e-4,
        'lr_patience': 30, 'early_stop_patience': 150, 'max_epochs': 1000,
        'e7_extra': {'use_contrastive': True, 'alpha': 0.05, 'beta': 0.0,
                     'gamma': 0.0, 'temperature': 0.1, 'augmentation': 'noise'},
        'seeds': SEEDS,
        'expected_gat_params': 521345,
        'task': 'V9 Phase 2 Task 2.5',
        'goal': 'GAT Davis E0 + E7 @1000ep, 5 matched seeds; encoders claim check',
        'timestamp': timestamp
    }
    with open(batch_dir / 'config.json', 'w') as f:
        json.dump(config_record, f, indent=2)

    print("V9 Phase 2 Task 2.5: GAT Davis E0+E7 @1000ep, Config C", flush=True)
    print(f"批次目录: {batch_dir}", flush=True)

    rec_e0 = run_group('e0', batch_dir)
    rec_e7 = run_group('e7', batch_dir)

    summary = {'timestamp': timestamp, 'task': 'V9 Phase 2 Task 2.5',
               'e0_completed': sum(1 for r in rec_e0 if r['success']),
               'e7_completed': sum(1 for r in rec_e7 if r['success']),
               'total': len(SEEDS) * 2,
               'records_e0': rec_e0, 'records_e7': rec_e7}
    with open(batch_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Task 2.5 批次结束: E0 {summary['e0_completed']}/5, E7 {summary['e7_completed']}/5")
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
