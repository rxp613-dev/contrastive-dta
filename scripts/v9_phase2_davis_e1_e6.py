#!/usr/bin/env python3
"""
V9 Phase 2 Task 2.3: Davis E1-E6 (α 剂量反应) @1000ep, Config C, 每剂量5种子
日期: 2026-11-09
目标: 在收敛协议 Config C 下完整重建 Davis 标准随机划分的 CL 损失权重 α 剂量-反应
      曲线，复核论文标题中关于"多种损失(multiple losses)"的主张是否在基线收敛后
      仍然成立。严格对应 TODO_v9_phase1_2_3.md Task 2.3.1~2.3.4，不简化、不改目标。
剂量点 (6 个, 按 TODO 原文):
      α = 0.01, 0.02, 0.05, 0.1, 0.2, 0.5
      每个剂量 5 个种子 → 6 × 5 = 30 个 run（α=0.05 也在本批次内统一重跑，
      保证整条剂量曲线来自同一批次、同一代码状态，不与 Task 2.1 批次混用）。
种子: [42, 7, 123, 456, 789]
      全部属于 Task 2.1 的 Config C E0 十种子，剂量点与 α=0 基线的同种子配对
      完整（基线数据直接引用 results/controlled/v9_p2_davis_e0_seed*，不重复跑）。
方法: 顺序调用 scripts/train_controlled.py，参数除 α 外与 Task 2.1 E7 完全一致：
      --use_contrastive --beta 0.0 --gamma 0.0 --temperature 0.1 --augmentation noise
      max_epochs=1000, lr_patience=30, early_stop_patience=150。
      为避免与历史/Task2.1 目录冲突，本批次 experiment_id 统一为
      v9_p2p3_davis_{alpha_tag}_seed{seed}，alpha_tag 映射:
      0.01→a001, 0.02→a002, 0.05→a005, 0.1→a010, 0.2→a020, 0.5→a050。
产出:
  - 每 run: results/controlled/v9_p2p3_davis_{tag}_seed{seed}/results.json
  - 批次目录: results/controlled/v9_phase2_davis_e1_e6_{timestamp}/
      config.json / progress.json / summary.json
运行: GPU（自动 CUDA），顺序不并发；预计 ~33 GPU小时（30 run）。
原则: 真实实验出数；执行前须独立审核；不编造任何结果。
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime


# (数值 α, 目录安全标签) —— 顺序即剂量曲线从低到高
DOSES = [
    (0.01, 'a001'),
    (0.02, 'a002'),
    (0.05, 'a005'),
    (0.1, 'a010'),
    (0.2, 'a020'),
    (0.5, 'a050'),
]
SEEDS = [42, 7, 123, 456, 789]
REPO = '/mnt/data/dta2'


def build_cmd(alpha, tag, seed):
    """构造单个剂量-种子 run 的命令行。"""
    exp_id = f"v9_p2p3_davis_{tag}_seed{seed}"
    return [
        sys.executable, 'scripts/train_controlled.py',
        '--experiment_id', exp_id,
        '--dataset', 'davis',
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
        '--alpha', str(alpha),
        '--beta', '0.0',
        '--gamma', '0.0',
        '--temperature', '0.1',
        '--augmentation', 'noise'
    ]


def run_one(alpha, tag, seed):
    """运行单个 run，返回 (success, returncode)。"""
    cmd = build_cmd(alpha, tag, seed)
    print("=" * 80)
    print(f"α={alpha} ({tag}) seed={seed}  命令: {' '.join(cmd)}")
    print("=" * 80, flush=True)
    try:
        result = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    except Exception as exc:
        print(f"❌ α={alpha} seed={seed} 调用异常: {exc}", flush=True)
        return False, -1

    print('\n'.join(result.stdout.split('\n')[-25:]), flush=True)
    if result.stderr.strip():
        print("--- stderr ---", flush=True)
        print(result.stderr[-2000:], flush=True)

    ok = (result.returncode == 0)
    print(f"{'✅ 完成' if ok else '❌ 失败'}: α={alpha} seed={seed} rc={result.returncode}", flush=True)
    return ok, result.returncode


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    batch_dir = Path(REPO) / 'results/controlled' / f'v9_phase2_davis_e1_e6_{timestamp}'
    batch_dir.mkdir(parents=True, exist_ok=True)

    config_record = {
        'config_version': 'Config C', 'dataset': 'davis', 'architecture': 'GIN',
        'batch_size': 512, 'learning_rate': 5e-4, 'weight_decay': 1e-4,
        'lr_patience': 30, 'early_stop_patience': 150, 'max_epochs': 1000,
        'contrastive': {'beta': 0.0, 'gamma': 0.0, 'temperature': 0.1,
                        'augmentation': 'noise'},
        'doses_alpha': [a for a, _ in DOSES],
        'seeds': SEEDS,
        'baseline_source': 'results/controlled/v9_p2_davis_e0_seed{seed} (Task 2.1, Config C)',
        'task': 'V9 Phase 2 Task 2.3',
        'goal': 'Davis alpha dose-response @1000ep, 6 doses x 5 seeds = 30 runs',
        'timestamp': timestamp
    }
    with open(batch_dir / 'config.json', 'w') as f:
        json.dump(config_record, f, indent=2)

    print("V9 Phase 2 Task 2.3: Davis E1-E6 dose-response @1000ep, Config C", flush=True)
    print(f"批次目录: {batch_dir}", flush=True)

    # 剂量从低到高；每剂量内固定 5 种子（同种子跨剂量配对，便于重复测量分析）
    records = []
    total = len(DOSES) * len(SEEDS)
    count = 0
    for alpha, tag in DOSES:
        for seed in SEEDS:
            count += 1
            print(f"\n>>> 全局进度 {count}/{total} ({count/total*100:.0f}%)  α={alpha}", flush=True)
            ok, rc = run_one(alpha, tag, seed)
            records.append({'alpha': alpha, 'tag': tag, 'seed': seed,
                            'success': ok, 'returncode': rc})
            with open(batch_dir / 'progress.json', 'w') as f:
                json.dump({'completed': count, 'total': total,
                           'records': records,
                           'last_update': datetime.now().isoformat()},
                          f, indent=2)

    n_ok = sum(1 for r in records if r['success'])
    with open(batch_dir / 'summary.json', 'w') as f:
        json.dump({'timestamp': timestamp, 'task': 'V9 Phase 2 Task 2.3',
                   'completed': n_ok, 'total': total, 'records': records},
                  f, indent=2)

    print("\n" + "=" * 80)
    print(f"Task 2.3 批次结束: {n_ok}/{total} runs 成功")
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
