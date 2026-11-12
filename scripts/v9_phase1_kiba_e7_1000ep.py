#!/usr/bin/env python3
"""
V9 Phase 1 Task 1.2: KIBA E7 α=0.05 @1000ep, Config C, 10种子
日期: 2026-11-04
目标: 使用Config C (max_epochs=1000)在KIBA数据集上运行E7 (对比学习 α=0.05)
方法: 使用train_controlled.py脚本，Config C配置
配置:
  - dataset: kiba
  - max_epochs: 1000 (Config C: 从500增加到1000)
  - lr_scheduler.patience: 30 (Config C: 保持不变)
  - early_stopping.patience: 150 (Config C: 保持不变)
  - alpha: 0.05 (E7: 对比学习)
  - temperature: 0.1
  - augmentation: noise
  - 其他参数与Config B完全相同
种子: 42, 7, 123, 456, 789, 101, 202, 303, 404, 505 (10个种子，与E0对称)
预计成本: ~44 GPU小时
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime


def run_experiment(seed, output_dir):
    """运行单个种子的实验"""
    exp_id = f"v9_p1_kiba_e7_seed{seed}"

    print("=" * 80)
    print(f"V9 Phase 1 Task 1.2: KIBA E7 (α=0.05) @1000ep, Config C")
    print(f"种子: {seed}")
    print(f"实验ID: {exp_id}")
    print("=" * 80)

    # Config C配置 + 对比学习
    cmd = [
        sys.executable, 'scripts/train_controlled.py',
        '--experiment_id', exp_id,
        '--dataset', 'kiba',
        '--seed', str(seed),
        '--batch_size', '512',
        '--max_epochs', '1000',  # Config C
        '--lr_patience', '30',   # Config C
        '--early_stop_patience', '150',  # Config C
        '--num_workers', '4',
        '--learning_rate', '0.0005',
        '--weight_decay', '0.0001',
        # E7: 启用对比学习
        '--use_contrastive',
        '--alpha', '0.05',
        '--beta', '0.0',
        '--gamma', '0.0',
        '--temperature', '0.1',
        '--augmentation', 'noise'
    ]

    print(f"\n命令: {' '.join(cmd)}")
    print(f"工作目录: /mnt/data/dta2")

    try:
        result = subprocess.run(cmd, cwd='/mnt/data/dta2', capture_output=True, text=True)

        print("\n--- 训练输出 (最后50行) ---")
        stdout_lines = result.stdout.split('\n')
        print('\n'.join(stdout_lines[-50:]))

        if result.stderr:
            print("\n--- 错误输出 ---")
            print(result.stderr)

        if result.returncode == 0:
            print(f"\n✅ 种子 {seed} 训练完成")
            return True
        else:
            print(f"\n❌ 种子 {seed} 训练失败")
            return False
    except Exception as e:
        print(f"\n❌ 种子 {seed} 训练异常: {e}")
        return False


def main():
    """主函数"""
    # Config C配置
    config = {
        'config_version': 'Config C',
        'batch_size': 512,
        'lr': 5e-4,
        'weight_decay': 1e-4,
        'lr_patience': 30,
        'early_stop_patience': 150,
        'max_epochs': 1000,  # Config C关键变化
        'dataset': 'kiba',
        'device': 'cuda',
        'contrastive': True,
        'alpha': 0.05,
        'beta': 0.0,
        'gamma': 0.0,
        'temperature': 0.1,
        'augmentation': 'noise'
    }

    # 10个种子 (与E0对称)
    seeds = [42, 7, 123, 456, 789, 101, 202, 303, 404, 505]

    # 输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'/mnt/data/dta2/results/controlled/v9_phase1_kiba_e7_1000ep_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"V9 Phase 1 Task 1.2: KIBA E7 (α=0.05) @1000ep, Config C")
    print(f"输出目录: {output_dir}")
    print(f"配置: {json.dumps(config, indent=2)}")
    print(f"种子: {seeds}")
    print(f"预计成本: ~44 GPU小时")
    print()

    # 保存配置
    with open(output_dir / 'config.json', 'w') as f:
        json.dump({
            'config': config,
            'seeds': seeds,
            'task': 'V9 Phase 1 Task 1.2',
            'goal': 'KIBA E7 (α=0.05) @1000ep, Config C, 10 seeds',
            'timestamp': timestamp
        }, f, indent=2)

    # 运行实验
    results = []
    for i, seed in enumerate(seeds, 1):
        print(f"\n{'='*80}")
        print(f"进度: {i}/{len(seeds)} ({i/len(seeds)*100:.1f}%)")
        print(f"{'='*80}\n")

        success = run_experiment(seed, output_dir)
        results.append({'seed': seed, 'success': success})

        # 每个种子完成后保存中间状态
        with open(output_dir / 'progress.json', 'w') as f:
            json.dump({
                'completed': i,
                'total': len(seeds),
                'results': results,
                'last_update': datetime.now().isoformat()
            }, f, indent=2)

    # 汇总结果
    completed = sum(1 for r in results if r['success'])

    summary = {
        'timestamp': timestamp,
        'task': 'V9 Phase 1 Task 1.2: KIBA E7 (α=0.05) @1000ep, Config C',
        'config': config,
        'seeds': seeds,
        'results': results,
        'completed': completed,
        'total': len(seeds),
        'success_rate': completed / len(seeds) * 100
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("V9 Phase 1 Task 1.2 完成")
    print("=" * 80)
    print(f"完成实验数: {completed}/{len(seeds)} ({completed/len(seeds)*100:.1f}%)")
    print(f"结果已保存到: {output_dir}")

    if completed == len(seeds):
        print("\n✅ 所有10个种子训练完成！")
    else:
        print(f"\n⚠️  有{len(seeds)-completed}个种子失败，请检查结果")


if __name__ == '__main__':
    main()
