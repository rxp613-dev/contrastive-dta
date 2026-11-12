#!/usr/bin/env python3
"""
V9 Task T1.5: Davis新协议E0+E7
日期: 2026-10-24
目标: 使用Config B在Davis数据集上运行E0和E7各10次
配置:
  - dataset: davis
  - batch_size: 512
  - lr_scheduler.patience: 30 (Config B)
  - early_stopping.patience: 150 (Config B)
  - max_epochs: 500
  - E0: alpha=0 (无对比学习)
  - E7: alpha=0.05 (对比学习)
种子: 42, 7, 123, 456, 789, 101, 202, 303, 404, 505
预计成本: 5-10 GPU小时 (Davis比KIBA快4倍)
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime


def run_experiment(seed, alpha, output_dir):
    """运行单个种子的实验"""
    exp_id = f"v9_t1p5_davis_{'e7' if alpha > 0 else 'e0'}_seed{seed}"

    print("=" * 80)
    print(f"V9 Task T1.5: Davis {'E7 (α=0.05)' if alpha > 0 else 'E0'} 新协议实验")
    print(f"种子: {seed}, α: {alpha}")
    print(f"实验ID: {exp_id}")
    print("=" * 80)

    # 构建命令（使用Config B: lr_patience=30, early_stop=150）
    cmd = [
        sys.executable, 'scripts/train_controlled.py',
        '--experiment_id', exp_id,
        '--dataset', 'davis',
        '--seed', str(seed),
        '--batch_size', '512',
        '--max_epochs', '500',
        '--lr_patience', '30',  # Config B
        '--early_stop_patience', '150',  # Config B
        '--num_workers', '4',
        '--learning_rate', '0.0005',
        '--weight_decay', '0.0001',
    ]

    if alpha > 0:
        cmd.extend([
            '--use_contrastive',
            '--alpha', str(alpha),
            '--beta', '0.0',
            '--gamma', '0.0',
            '--temperature', '0.1',
            '--augmentation', 'noise'
        ])

    print(f"\n命令: {' '.join(cmd)}")
    print(f"工作目录: /mnt/data/dta2")

    try:
        result = subprocess.run(cmd, cwd='/mnt/data/dta2', capture_output=True, text=True)

        # 打印输出
        print("\n--- 训练输出 ---")
        print(result.stdout)
        if result.stderr:
            print("\n--- 错误输出 ---")
            print(result.stderr)

        if result.returncode == 0:
            print(f"\n✅ 种子 {seed} (α={alpha}) 训练完成")
            return True
        else:
            print(f"\n❌ 种子 {seed} (α={alpha}) 训练失败")
            return False
    except Exception as e:
        print(f"\n❌ 种子 {seed} (α={alpha}) 训练异常: {e}")
        return False


def main():
    """主函数"""
    # Config B配置
    config = {
        'dataset': 'davis',
        'batch_size': 512,
        'lr': 5e-4,
        'weight_decay': 1e-4,
        'lr_patience': 30,  # Config B
        'early_stop_patience': 150,  # Config B
        'max_epochs': 500,
        'device': 'cuda'
    }

    # 10个种子
    seeds = [42, 7, 123, 456, 789, 101, 202, 303, 404, 505]

    # E0和E7配置
    alphas = [0.0, 0.05]  # E0和E7

    # 输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'/mnt/data/dta2/results/controlled/v9_t1p5_davis_e0_e7_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"V9 Task T1.5: Davis新协议E0+E7实验")
    print(f"输出目录: {output_dir}")
    print(f"配置: {json.dumps(config, indent=2)}")
    print(f"种子: {seeds}")
    print(f"α值: {alphas}")
    print()

    # 保存配置
    with open(output_dir / 'config.json', 'w') as f:
        json.dump({'config': config, 'seeds': seeds, 'alphas': alphas}, f, indent=2)

    # 运行实验
    results = []
    for alpha in alphas:
        for seed in seeds:
            success = run_experiment(seed, alpha, output_dir)
            results.append({'seed': seed, 'alpha': alpha, 'success': success})
            print("\n" + "=" * 80 + "\n")

    # 汇总结果
    completed = sum(1 for r in results if r['success'])

    summary = {
        'timestamp': timestamp,
        'task': 'V9 T1.5: Davis新协议E0+E7验证',
        'config': config,
        'seeds': seeds,
        'alphas': alphas,
        'results': results,
        'completed': completed,
        'total': len(seeds) * len(alphas)
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("V9 Task T1.5 完成")
    print("=" * 80)
    print(f"完成实验数: {completed}/{len(seeds) * len(alphas)}")
    print(f"结果已保存到: {output_dir}")


if __name__ == '__main__':
    main()
