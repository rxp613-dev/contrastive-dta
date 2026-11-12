#!/usr/bin/env python3
"""
V9 Task T0.4: 0.4847 vs 0.5220对账
日期: 2026-10-22
目标: 确认两个E0 MSE值的来源并分析差异原因
方法:
  1. 搜索Table 5中0.5220±0.0414的来源
  2. 搜索诊断报告中0.4847±0.1369的来源
  3. 分析7%差异的可能原因
  4. 生成对账报告
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

BASE_DIR = Path('/mnt/data/dta2/results/controlled')
OUTPUT_DIR = Path('/mnt/data/dta2/results/analysis')

def search_for_0_5220():
    """搜索0.5220的来源"""
    # 从旧E0的10个种子计算
    seeds = [42, 123, 251, 1496, 2024, 3407, 5934, 7, 7350, 9889]
    mses = []

    for seed in seeds:
        result_path = BASE_DIR / f'KIBA_E0_seed{seed}' / 'test_results.json'
        if result_path.exists():
            with open(result_path, 'r') as f:
                data = json.load(f)
            if 'mse' in data:
                mses.append(data['mse'])

    if mses:
        return {
            'source': '旧E0 10种子',
            'count': len(mses),
            'mean': np.mean(mses),
            'std': np.std(mses),
            'values': mses
        }

    # 尝试从GIN_KIBA_alpha0_seed目录加载
    mses = []
    for seed in seeds:
        result_path = BASE_DIR / f'GIN_KIBA_alpha0_seed{seed}' / 'test_results.json'
        if result_path.exists():
            with open(result_path, 'r') as f:
                data = json.load(f)
            if 'mse' in data:
                mses.append(data['mse'])

    if mses:
        return {
            'source': '旧E0 10种子 (GIN_KIBA_alpha0)',
            'count': len(mses),
            'mean': np.mean(mses),
            'std': np.std(mses),
            'values': mses
        }

    return None

def search_for_0_4847():
    """搜索0.4847的来源"""
    # 从诊断报告中的5个种子计算
    seeds = [42, 7, 123, 2024, 3407]
    mses = []

    for seed in seeds:
        result_path = BASE_DIR / f'GIN_KIBA_alpha0_seed{seed}' / 'test_results.json'
        if result_path.exists():
            with open(result_path, 'r') as f:
                data = json.load(f)
            if 'mse' in data:
                mses.append(data['mse'])

    if mses:
        return {
            'source': '诊断报告 5种子',
            'count': len(mses),
            'mean': np.mean(mses),
            'std': np.std(mses),
            'values': mses
        }

    return None

def generate_report(val_0_5220, val_0_4847):
    """生成对账报告"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'task': 'T0.4: 0.4847 vs 0.5220对账',
        'target_values': {
            'table_5': {
                'value': 0.5220,
                'std': 0.0414,
                'description': 'Table 5中的E0 MSE值'
            },
            'diagnostic_report': {
                'value': 0.4847,
                'std': 0.1369,
                'description': '诊断报告中的E0 MSE值'
            }
        },
        'actual_values': {},
        'analysis': {}
    }

    if val_0_5220:
        report['actual_values']['table_5_source'] = val_0_5220
        report['analysis']['table_5_match'] = abs(val_0_5220['mean'] - 0.5220) < 0.01

    if val_0_4847:
        report['actual_values']['diagnostic_source'] = val_0_4847
        report['analysis']['diagnostic_match'] = abs(val_0_4847['mean'] - 0.4847) < 0.01

    if val_0_5220 and val_0_4847:
        diff = val_0_5220['mean'] - val_0_4847['mean']
        diff_pct = diff / val_0_4847['mean'] * 100
        report['analysis']['difference'] = {
            'absolute': diff,
            'percentage': diff_pct,
            'possible_causes': [
                '样本量差异（10种子 vs 5种子）',
                '种子选择差异（不同种子集合）',
                '训练轮次差异（不同epoch数）',
                '数据划分差异（不同随机种子）'
            ]
        }

    # 保存报告
    output_path = OUTPUT_DIR / 'task_v9_t0p4_reconciliation_report.json'
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n报告已保存: {output_path}")

    # 打印摘要
    print("\n" + "="*80)
    print("V9 Task T0.4: 0.4847 vs 0.5220对账 - 完成")
    print("="*80)

    print("\n目标值:")
    print(f"  Table 5: 0.5220 ± 0.0414")
    print(f"  诊断报告: 0.4847 ± 0.1369")
    print(f"  差异: {0.5220 - 0.4847:.4f} ({(0.5220 - 0.4847) / 0.4847 * 100:.1f}%)")

    print("\n实际计算值:")
    if val_0_5220:
        print(f"  Table 5来源 ({val_0_5220['source']}):")
        print(f"    MSE = {val_0_5220['mean']:.4f} ± {val_0_5220['std']:.4f}")
        print(f"    匹配: {'✅' if report['analysis']['table_5_match'] else '❌'}")
    else:
        print(f"  Table 5来源: ❌ 未找到")

    if val_0_4847:
        print(f"  诊断报告来源 ({val_0_4847['source']}):")
        print(f"    MSE = {val_0_4847['mean']:.4f} ± {val_0_4847['std']:.4f}")
        print(f"    匹配: {'✅' if report['analysis']['diagnostic_match'] else '❌'}")
    else:
        print(f"  诊断报告来源: ❌ 未找到")

    if val_0_5220 and val_0_4847:
        print("\n差异分析:")
        print(f"  绝对差异: {report['analysis']['difference']['absolute']:.4f}")
        print(f"  相对差异: {report['analysis']['difference']['percentage']:.1f}%")
        print(f"  可能原因:")
        for cause in report['analysis']['difference']['possible_causes']:
            print(f"    - {cause}")

    return report

def main():
    print("="*80)
    print("V9 Task T0.4: 0.4847 vs 0.5220对账")
    print("="*80)

    print("\n搜索0.5220的来源...")
    val_0_5220 = search_for_0_5220()

    print("\n搜索0.4847的来源...")
    val_0_4847 = search_for_0_4847()

    print("\n生成对账报告...")
    report = generate_report(val_0_5220, val_0_4847)

    print("\n" + "="*80)
    print("Task T0.4 完成")
    print("="*80)

if __name__ == '__main__':
    main()
