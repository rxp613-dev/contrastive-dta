#!/usr/bin/env python3
"""
V9 Task T0.2: 旧α-scan训练曲线分析
日期: 2026-10-22
目标: 对比CL组vs E0组的逃逸率，验证"CL=训练稳定器"机理
方法:
  1. 加载旧协议下α-scan（α=0.05, 0.1, 0.5等）的训练日志
  2. 提取每epoch的LR、val MSE
  3. 对比CL组vs E0组的逃逸率
  4. 若CL组逃逸率更高，则"CL=训练稳定器"机理被坐实
  5. 生成对比报告
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# 配置
BASE_DIR = Path('/mnt/data/dta2/results/controlled')
OUTPUT_DIR = Path('/mnt/data/dta2/results/analysis')

# Alpha values to analyze
ALPHA_VALUES = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
SEEDS = [42, 7, 123, 2024, 3407]

# 逃逸阈值定义
ESCAPE_THRESHOLD_EPOCHS = 100  # 训练到100+ epochs视为"逃逸"
ESCAPE_THRESHOLD_MSE = 0.3     # val MSE < 0.3视为成功收敛

def load_training_log(alpha, seed):
    """加载单个种子的训练日志"""
    # 尝试不同的目录命名模式
    patterns = [
        f'GIN_KIBA_alpha{int(alpha*1000)}_seed{seed}',  # alpha001, alpha005等
        f'GIN_KIBA_alpha{int(alpha)}_seed{seed}',       # alpha0, alpha1等
    ]

    for pattern in patterns:
        log_path = BASE_DIR / pattern / 'training_log.json'
        if log_path.exists():
            with open(log_path, 'r') as f:
                data = json.load(f)
            return data

    return None

def analyze_seed(data, alpha, seed):
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
        'alpha': alpha,
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

def plot_alpha_comparison(all_results):
    """绘制α对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Escape rate by alpha
    ax = axes[0, 0]
    alpha_escape_rates = {}
    for alpha in ALPHA_VALUES:
        results = [r for r in all_results if r and r['alpha'] == alpha]
        if results:
            escaped = sum(1 for r in results if r['escaped'])
            rate = escaped / len(results) * 100
            alpha_escape_rates[alpha] = rate

    if alpha_escape_rates:
        alphas = list(alpha_escape_rates.keys())
        rates = list(alpha_escape_rates.values())
        colors = ['red' if r == 0 else 'orange' if r < 50 else 'green' for r in rates]
        ax.bar(range(len(alphas)), rates, color=colors, alpha=0.7)
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f'{a}' for a in alphas], rotation=45)
        ax.set_xlabel('Alpha Value', fontsize=12)
        ax.set_ylabel('Escape Rate (%)', fontsize=12)
        ax.set_title('Escape Rate by Alpha (Red=0%, Orange<50%, Green≥50%)', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 100)

    # 2. Average epochs by alpha
    ax = axes[0, 1]
    alpha_epochs = {}
    for alpha in ALPHA_VALUES:
        results = [r for r in all_results if r and r['alpha'] == alpha]
        if results:
            avg_epochs = np.mean([r['total_epochs'] for r in results])
            alpha_epochs[alpha] = avg_epochs

    if alpha_epochs:
        alphas = list(alpha_epochs.keys())
        epochs_vals = list(alpha_epochs.values())
        ax.bar(range(len(alphas)), epochs_vals, color='steelblue', alpha=0.7)
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f'{a}' for a in alphas], rotation=45)
        ax.set_xlabel('Alpha Value', fontsize=12)
        ax.set_ylabel('Average Epochs', fontsize=12)
        ax.set_title('Average Training Epochs by Alpha', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')

    # 3. Best MSE by alpha
    ax = axes[1, 0]
    alpha_mses = {}
    for alpha in ALPHA_VALUES:
        results = [r for r in all_results if r and r['alpha'] == alpha]
        if results:
            avg_mse = np.mean([r['best_val_mse'] for r in results])
            alpha_mses[alpha] = avg_mse

    if alpha_mses:
        alphas = list(alpha_mses.keys())
        mse_vals = list(alpha_mses.values())
        colors = ['green' if m < 0.3 else 'orange' if m < 0.5 else 'red' for m in mse_vals]
        ax.bar(range(len(alphas)), mse_vals, color=colors, alpha=0.7)
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f'{a}' for a in alphas], rotation=45)
        ax.set_xlabel('Alpha Value', fontsize=12)
        ax.set_ylabel('Best Val MSE', fontsize=12)
        ax.set_title('Best Val MSE by Alpha (Green<0.3, Orange<0.5, Red≥0.5)', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='Escape threshold')
        ax.legend()

    # 4. Final CI by alpha
    ax = axes[1, 1]
    alpha_cis = {}
    for alpha in ALPHA_VALUES:
        results = [r for r in all_results if r and r['alpha'] == alpha]
        if results:
            avg_ci = np.mean([r['final_val_ci'] for r in results])
            alpha_cis[alpha] = avg_ci

    if alpha_cis:
        alphas = list(alpha_cis.keys())
        ci_vals = list(alpha_cis.values())
        colors = ['green' if c > 0.85 else 'orange' if c > 0.75 else 'red' for c in ci_vals]
        ax.bar(range(len(alphas)), ci_vals, color=colors, alpha=0.7)
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f'{a}' for a in alphas], rotation=45)
        ax.set_xlabel('Alpha Value', fontsize=12)
        ax.set_ylabel('Final Val CI', fontsize=12)
        ax.set_title('Final Val CI by Alpha (Green>0.85, Orange>0.75, Red≤0.75)', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        ax.axhline(y=0.882, color='green', linestyle='--', alpha=0.5, label='GraphDTA-GIN CI')
        ax.legend()

    plt.tight_layout()

    # 保存图片
    output_path = OUTPUT_DIR / 'task_v9_t0p2_alpha_scan_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"对比图已保存: {output_path}")

    plt.close()

def generate_report(all_results):
    """生成对比报告"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'task': 'T0.2: 旧α-scan训练曲线分析',
        'escape_threshold': {
            'epochs': ESCAPE_THRESHOLD_EPOCHS,
            'val_mse': ESCAPE_THRESHOLD_MSE
        },
        'alpha_analysis': {}
    }

    # 按alpha分组统计
    for alpha in ALPHA_VALUES:
        results = [r for r in all_results if r and r['alpha'] == alpha]
        if results:
            escaped_count = sum(1 for r in results if r['escaped'])
            total_count = len(results)
            escape_rate = escaped_count / total_count * 100

            avg_epochs = np.mean([r['total_epochs'] for r in results])
            avg_best_mse = np.mean([r['best_val_mse'] for r in results])
            avg_final_ci = np.mean([r['final_val_ci'] for r in results])

            report['alpha_analysis'][str(alpha)] = {
                'total_runs': total_count,
                'escaped': escaped_count,
                'escape_rate_percent': escape_rate,
                'avg_epochs': avg_epochs,
                'avg_best_mse': avg_best_mse,
                'avg_final_ci': avg_final_ci,
                'per_seed_results': [
                    {
                        'seed': r['seed'],
                        'total_epochs': r['total_epochs'],
                        'best_val_mse': r['best_val_mse'],
                        'best_epoch': r['best_epoch'],
                        'final_val_ci': r['final_val_ci'],
                        'escaped': r['escaped']
                    }
                    for r in results
                ]
            }

    # 保存报告
    output_path = OUTPUT_DIR / 'task_v9_t0p2_alpha_scan_report.json'
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n报告已保存: {output_path}")

    # 打印摘要
    print("\n" + "="*80)
    print("V9 Task T0.2: 旧α-scan训练曲线分析 - 完成")
    print("="*80)
    print(f"\n逃逸阈值: {ESCAPE_THRESHOLD_EPOCHS} epochs, val MSE < {ESCAPE_THRESHOLD_MSE}")

    print("\n各Alpha值统计:")
    print(f"{'Alpha':<10} {'Runs':<8} {'Escaped':<10} {'Escape%':<10} {'Avg Epochs':<12} {'Avg MSE':<12} {'Avg CI':<12}")
    print("-"*80)

    for alpha in ALPHA_VALUES:
        if str(alpha) in report['alpha_analysis']:
            data = report['alpha_analysis'][str(alpha)]
            print(f"{alpha:<10.3f} {data['total_runs']:<8} {data['escaped']:<10} "
                  f"{data['escape_rate_percent']:<10.1f} {data['avg_epochs']:<12.1f} "
                  f"{data['avg_best_mse']:<12.4f} {data['avg_final_ci']:<12.4f}")

    # 关键发现
    print("\n关键发现:")

    # 对比E0 (alpha=0) vs CL组 (alpha>0)
    e0_data = report['alpha_analysis'].get('0', None)
    cl_alphas = [a for a in ALPHA_VALUES if a > 0 and str(a) in report['alpha_analysis']]

    if e0_data and cl_alphas:
        e0_escape_rate = e0_data['escape_rate_percent']
        cl_escape_rates = [report['alpha_analysis'][str(a)]['escape_rate_percent'] for a in cl_alphas]
        avg_cl_escape_rate = np.mean(cl_escape_rates) if cl_escape_rates else 0

        print(f"\n  E0 (α=0) 逃逸率: {e0_escape_rate:.1f}%")
        print(f"  CL组 (α>0) 平均逃逸率: {avg_cl_escape_rate:.1f}%")

        if avg_cl_escape_rate > e0_escape_rate:
            print(f"  ✅ CL组逃逸率高于E0组，支持'CL=训练稳定器'机理")
        else:
            print(f"  ⚠️  CL组逃逸率不高于E0组，'CL=训练稳定器'机理未获支持")

        # 检查是否有alpha值达到0%逃逸率
        zero_escape_alphas = [a for a in cl_alphas if report['alpha_analysis'][str(a)]['escape_rate_percent'] == 0]
        if zero_escape_alphas:
            print(f"  ⚠️  以下alpha值逃逸率为0%: {zero_escape_alphas}")
            print(f"  ⚠️  表明即使CL也无法解决旧协议的训练不充分问题")

    return report

def main():
    print("="*80)
    print("V9 Task T0.2: 旧α-scan训练曲线分析")
    print("="*80)

    # 加载所有alpha-scan的训练日志
    print("\n加载训练日志...")
    all_results = []

    for alpha in ALPHA_VALUES:
        for seed in SEEDS:
            data = load_training_log(alpha, seed)
            if data:
                result = analyze_seed(data, alpha, seed)
                all_results.append(result)
                print(f"  α={alpha}, Seed {seed}: {len(data)} epochs")
            else:
                print(f"  α={alpha}, Seed {seed}: 加载失败")

    # 绘制对比图
    print("\n绘制α对比图...")
    plot_alpha_comparison(all_results)

    # 生成报告
    print("\n生成对比报告...")
    report = generate_report(all_results)

    print("\n" + "="*80)
    print("Task T0.2 完成")
    print("="*80)

if __name__ == '__main__':
    main()
