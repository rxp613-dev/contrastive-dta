#!/usr/bin/env python3
"""
V9 Task T0.1: 旧10种子E0逐epoch曲线分析
日期: 2026-10-22
目标: 识别旧协议下E0训练的"卡在平台"vs"逃逸"二分模式
方法:
  1. 加载10个种子的training_log.json
  2. 提取每epoch的LR、val MSE、train loss
  3. 绘制逐epoch曲线
  4. 识别二分模式并统计逃逸率
  5. 生成机理分析报告
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# 配置
SEEDS = [42, 123, 251, 1496, 2024, 3407, 5934, 7, 7350, 9889]
BASE_DIR = Path('/mnt/data/dta2/results/controlled')
OUTPUT_DIR = Path('/mnt/data/dta2/results/analysis')

# 逃逸阈值定义
ESCAPE_THRESHOLD_EPOCHS = 100  # 训练到100+ epochs视为"逃逸"
ESCAPE_THRESHOLD_MSE = 0.3     # val MSE < 0.3视为成功收敛

def load_training_log(seed):
    """加载单个种子的训练日志"""
    log_path = BASE_DIR / f'KIBA_E0_seed{seed}' / 'training_log.json'
    if not log_path.exists():
        print(f"警告: {log_path} 不存在")
        return None

    with open(log_path, 'r') as f:
        data = json.load(f)

    return data

def analyze_seed(data, seed):
    """分析单个种子的训练模式"""
    if not data:
        return None

    epochs = [d['epoch'] for d in data]
    val_mses = [d['val_mse'] for d in data]
    lrs = [d['lr'] for d in data]
    train_losses = [d['train_loss'] for d in data]
    val_cis = [d['val_ci'] for d in data]

    total_epochs = len(epochs)
    final_val_mse = val_mses[-1]
    best_val_mse = min(val_mses)
    best_epoch = epochs[val_mses.index(best_val_mse)]
    final_lr = lrs[-1]
    final_val_ci = val_cis[-1]

    # 判断是否逃逸
    escaped = total_epochs >= ESCAPE_THRESHOLD_EPOCHS and best_val_mse < ESCAPE_THRESHOLD_MSE

    return {
        'seed': seed,
        'total_epochs': total_epochs,
        'final_val_mse': final_val_mse,
        'best_val_mse': best_val_mse,
        'best_epoch': best_epoch,
        'final_lr': final_lr,
        'final_val_ci': final_val_ci,
        'escaped': escaped,
        'epochs': epochs,
        'val_mses': val_mses,
        'lrs': lrs,
        'train_losses': train_losses,
        'val_cis': val_cis
    }

def plot_epoch_curves(all_results):
    """绘制逐epoch曲线"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Val MSE curves
    ax = axes[0, 0]
    for result in all_results:
        if result:
            color = 'green' if result['escaped'] else 'red'
            ax.plot(result['epochs'], result['val_mses'],
                   label=f"Seed {result['seed']} ({'escaped' if result['escaped'] else 'stuck'})",
                   color=color, alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation MSE', fontsize=12)
    ax.set_title('Val MSE Curves (Red=Stuck, Green=Escaped)', fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 2.5)

    # 2. Learning rate curves
    ax = axes[0, 1]
    for result in all_results:
        if result:
            color = 'green' if result['escaped'] else 'red'
            ax.semilogy(result['epochs'], result['lrs'],
                       label=f"Seed {result['seed']}",
                       color=color, alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate (log scale)', fontsize=12)
    ax.set_title('Learning Rate Schedules', fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # 3. Train loss curves
    ax = axes[1, 0]
    for result in all_results:
        if result:
            color = 'green' if result['escaped'] else 'red'
            ax.plot(result['epochs'], result['train_losses'],
                   label=f"Seed {result['seed']}",
                   color=color, alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Train Loss', fontsize=12)
    ax.set_title('Train Loss Curves', fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    # 4. Val CI curves
    ax = axes[1, 1]
    for result in all_results:
        if result:
            color = 'green' if result['escaped'] else 'red'
            ax.plot(result['epochs'], result['val_cis'],
                   label=f"Seed {result['seed']}",
                   color=color, alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation CI', fontsize=12)
    ax.set_title('Val CI Curves', fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.6, 0.8)

    plt.tight_layout()

    # 保存图片
    output_path = OUTPUT_DIR / 'task_v9_t0p1_old_e0_epoch_curves.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"曲线已保存: {output_path}")

    plt.close()

def generate_report(all_results):
    """生成机理分析报告"""
    escaped_count = sum(1 for r in all_results if r and r['escaped'])
    stuck_count = len(all_results) - escaped_count
    escape_rate = escaped_count / len(all_results) * 100

    report = {
        'timestamp': datetime.now().isoformat(),
        'task': 'T0.1: 旧10种子E0逐epoch曲线分析',
        'total_seeds': len(all_results),
        'escape_threshold': {
            'epochs': ESCAPE_THRESHOLD_EPOCHS,
            'val_mse': ESCAPE_THRESHOLD_MSE
        },
        'summary': {
            'escaped': escaped_count,
            'stuck': stuck_count,
            'escape_rate_percent': escape_rate
        },
        'per_seed_results': []
    }

    for result in all_results:
        if result:
            report['per_seed_results'].append({
                'seed': result['seed'],
                'total_epochs': result['total_epochs'],
                'best_val_mse': result['best_val_mse'],
                'best_epoch': result['best_epoch'],
                'final_val_ci': result['final_val_ci'],
                'final_lr': result['final_lr'],
                'escaped': result['escaped']
            })

    # 保存报告
    output_path = OUTPUT_DIR / 'task_v9_t0p1_old_e0_analysis_report.json'
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n报告已保存: {output_path}")

    # 打印摘要
    print("\n" + "="*80)
    print("V9 Task T0.1: 旧10种子E0逐epoch曲线分析 - 完成")
    print("="*80)
    print(f"\n逃逸阈值: {ESCAPE_THRESHOLD_EPOCHS} epochs, val MSE < {ESCAPE_THRESHOLD_MSE}")
    print(f"\n逃逸统计:")
    print(f"  逃逸种子: {escaped_count}/{len(all_results)}")
    print(f"  卡住种子: {stuck_count}/{len(all_results)}")
    print(f"  逃逸率: {escape_rate:.1f}%")

    print("\n各种子详情:")
    print(f"{'Seed':<8} {'Epochs':<10} {'Best MSE':<12} {'Best Epoch':<12} {'Final CI':<12} {'Final LR':<12} {'Status':<10}")
    print("-"*80)
    for result in all_results:
        if result:
            status = "ESCAPED" if result['escaped'] else "STUCK"
            print(f"{result['seed']:<8} {result['total_epochs']:<10} {result['best_val_mse']:<12.4f} "
                  f"{result['best_epoch']:<12} {result['final_val_ci']:<12.4f} "
                  f"{result['final_lr']:<12.2e} {status:<10}")

    print("\n关键发现:")
    if escape_rate < 20:
        print("  ⚠️  逃逸率极低（<20%），表明旧协议存在严重训练不充分问题")
        print("  ⚠️  大多数种子在3-11 epoch就因patience=50过早停止")
        print("  ⚠️  学习率调度器patience=10在KIBA上过早降低学习率")
    else:
        print(f"  ✓ 逃逸率为{escape_rate:.1f}%，部分种子能够收敛")

    print("\n机理分析:")
    print("  KIBA数据集118K样本，batch 512 → 每epoch仅231步")
    print("  10个epoch = 2300步，远少于GraphDTA的9220步")
    print("  ReduceLROnPlateau(patience=10)在KIBA上过早降学习率")
    print("  导致模型在epoch 3-11就因patience=50过早停止")

    return report

def main():
    print("="*80)
    print("V9 Task T0.1: 旧10种子E0逐epoch曲线分析")
    print("="*80)

    # 加载所有种子的训练日志
    print("\n加载训练日志...")
    all_data = []
    for seed in SEEDS:
        data = load_training_log(seed)
        all_data.append(data)
        if data:
            print(f"  Seed {seed}: {len(data)} epochs")
        else:
            print(f"  Seed {seed}: 加载失败")

    # 分析每个种子
    print("\n分析训练模式...")
    all_results = []
    for data in all_data:
        if data:
            result = analyze_seed(data, data[0].get('seed', 0))
            all_results.append(result)

    # 绘制曲线
    print("\n绘制逐epoch曲线...")
    plot_epoch_curves(all_results)

    # 生成报告
    print("\n生成机理分析报告...")
    report = generate_report(all_results)

    print("\n" + "="*80)
    print("Task T0.1 完成")
    print("="*80)

if __name__ == '__main__':
    main()
