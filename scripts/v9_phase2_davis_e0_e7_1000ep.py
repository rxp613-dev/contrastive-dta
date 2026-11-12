#!/usr/bin/env python3
"""
V9 Phase 2 Task 2.1: Davis E0 + E7 @1000ep, Config C, 各10种子
日期: 2026-11-08
目标: 用收敛协议 Config C 在 Davis 标准随机划分上重跑被作废(Config B, 500ep)的
      E0(基线) 与 E7(对比学习 α=0.05)，各 10 个同种子配对：
        - 检验 KIBA 上观察到的"CL 显著更差"是否跨数据集成立
        - 检验"方差缩减"点估计在 Davis 上是否重现/能否达到显著
      严格对应 TODO_v9_phase1_2_3.md Task 2.1.1~2.1.4，不简化、不改目标。
方法: 顺序调用 scripts/train_controlled.py：
        阶段一: E0 ×10 (无对比学习)
        阶段二: E7 ×10 (--use_contrastive --alpha 0.05 --temperature 0.1 --augmentation noise)
      除 max_epochs 500→1000 外，全部超参与 Config B 完全一致(Config C 冻结协议)。
配置:
  dataset=davis, architecture=GIN, batch_size=512,
  learning_rate=5e-4, weight_decay=1e-4,
  max_epochs=1000 (Config C), lr_patience=30, early_stop_patience=150, num_workers=4
种子: [42, 7, 123, 456, 789, 101, 202, 303, 404, 505] (E0/E7 对称)
产出:
  - 每个种子: results/controlled/v9_p2_davis_{e0,e7}_seed{seed}/results.json (train_controlled 写出)
  - 本批次目录: results/controlled/v9_phase2_davis_e0_e7_1000ep_{timestamp}/
      内含 config.json / progress_e0.json / progress_e7.json / summary.json
运行: GPU (train_controlled.py 自动使用 CUDA)；预计 ~22 GPU小时(20 个 run)。
原则: 真实实验出数；脚本执行前须经独立审核。
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime


SEEDS = [42, 7, 123, 456, 789, 101, 202, 303, 404, 505]
REPO = '/mnt/data/dta2'


def build_cmd(group, seed):
    """
    构造单个 run 的命令行。
    group: 'e0'（基线）或 'e7'（对比学习 α=0.05）。
    """
    exp_id = f"v9_p2_davis_{group}_seed{seed}"
    cmd = [
        sys.executable, 'scripts/train_controlled.py',
        '--experiment_id', exp_id,
        '--dataset', 'davis',
        '--seed', str(seed),
        '--architecture', 'GIN',
        '--batch_size', '512',
        '--max_epochs', '1000',          # Config C（唯一变化项）
        '--lr_patience', '30',
        '--early_stop_patience', '150',
        '--num_workers', '4',
        '--learning_rate', '0.0005',
        '--weight_decay', '0.0001',
    ]
    if group == 'e7':
        # E7: 启用对比学习，其余对比项 β/γ 关闭，隔离 α 单一损失效应
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
    """运行单个种子；subprocess 完成后返回 (success, returncode)。"""
    cmd = build_cmd(group, seed)
    print("=" * 80)
    print(f"[{group.upper()}] seed={seed}  命令: {' '.join(cmd)}")
    print("=" * 80, flush=True)

    try:
        result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    except Exception as exc:  # 启动级异常不应杀死整批，记录后继续
        print(f"❌ seed={seed} 调用异常: {exc}", flush=True)
        return False, -1

    # 打印尾部输出，便于在 nohup 日志中回溯
    tail = '\n'.join(result.stdout.split('\n')[-30:])
    print(tail, flush=True)
    if result.stderr.strip():
        print("--- stderr ---", flush=True)
        print(result.stderr[-2000:], flush=True)

    ok = (result.returncode == 0)
    print(f"{'✅ 完成' if ok else '❌ 失败'}: {group} seed={seed} rc={result.returncode}", flush=True)
    return ok, result.returncode


def run_group(group, batch_dir):
    """运行一个组（e0 或 e7）的全部 10 个种子，逐步落盘 progress。"""
    records = []
    for i, seed in enumerate(SEEDS, 1):
        print(f"\n>>> 进度 {group}: {i}/{len(SEEDS)} ({i/len(SEEDS)*100:.0f}%)", flush=True)
        ok, rc = run_one(group, seed)
        records.append({'seed': seed, 'success': ok, 'returncode': rc})

        with open(batch_dir / f'progress_{group}.json', 'w') as f:
            json.dump({
                'group': group,
                'completed': i,
                'total': len(SEEDS),
                'records': records,
                'last_update': datetime.now().isoformat()
            }, f, indent=2)
    return records


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_dir = Path(REPO) / 'results/controlled' / f'v9_phase2_davis_e0_e7_1000ep_{timestamp}'
    batch_dir.mkdir(parents=True, exist_ok=True)

    config_record = {
        'config_version': 'Config C',
        'dataset': 'davis',
        'architecture': 'GIN',
        'batch_size': 512,
        'learning_rate': 5e-4,
        'weight_decay': 1e-4,
        'lr_patience': 30,
        'early_stop_patience': 150,
        'max_epochs': 1000,
        'e7_extra': {'use_contrastive': True, 'alpha': 0.05, 'beta': 0.0,
                     'gamma': 0.0, 'temperature': 0.1, 'augmentation': 'noise'},
        'seeds': SEEDS,
        'task': 'V9 Phase 2 Task 2.1',
        'goal': 'Davis E0 + E7 @1000ep, Config C, 10 matched seeds each',
        'timestamp': timestamp
    }
    with open(batch_dir / 'config.json', 'w') as f:
        json.dump(config_record, f, indent=2)

    print(f"V9 Phase 2 Task 2.1: Davis E0+E7 @1000ep, Config C", flush=True)
    print(f"批次目录: {batch_dir}", flush=True)

    # Task 2.1.2: E0 ×10；Task 2.1.3: E7 ×10（顺序执行，避免两个 run 抢 GPU）
    rec_e0 = run_group('e0', batch_dir)
    rec_e7 = run_group('e7', batch_dir)

    summary = {
        'timestamp': timestamp,
        'task': 'V9 Phase 2 Task 2.1',
        'e0_completed': sum(1 for r in rec_e0 if r['success']),
        'e7_completed': sum(1 for r in rec_e7 if r['success']),
        'total': len(SEEDS) * 2,
        'records_e0': rec_e0,
        'records_e7': rec_e7
    }
    with open(batch_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Task 2.1 批次结束: E0 {summary['e0_completed']}/10, E7 {summary['e7_completed']}/10")
    print(f"批次目录: {batch_dir}")
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
