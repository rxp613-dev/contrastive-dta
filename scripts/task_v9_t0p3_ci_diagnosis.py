#!/usr/bin/env python3
"""
V9 Task T0.3: CI指标核查
日期: 2026-10-22
目标: 对比旧E0 CI（0.7114）vs 修正E0 CI vs 文献GIN CI（0.882）
方法:
  1. 从修正E0结果中提取CI（Seed 42和7）
  2. 从旧E0结果中提取CI（所有种子）
  3. 对比分析
  4. 生成CI诊断报告
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

BASE_DIR = Path('/mnt/data/dta2/results/controlled')
OUTPUT_DIR = Path('/mnt/data/dta2/results/analysis')

def load_test_results_from_corrected_e0():
    """从修正E0结果中提取CI"""
    results = {}
    # 修正E0目录
    corrected_dir = BASE_DIR / 'v9_e3_corrected_e0_20261021_185458'

    for seed in [42, 7]:
        result_path = corrected_dir / f'results_seed{seed}.json'
        if result_path.exists():
            with open(result_path, 'r') as f:
                data = json.load(f)
            results[f'corrected_seed{seed}'] = {
                'ci': data.get('ci', None),
                'mse': data.get('mse', None),
                'rmse': data.get('rmse', None)
            }

    return results

def load_old_e0_ci():
    """从旧E0结果中提取CI"""
    seeds = [42, 123, 251, 1496, 2024, 3407, 5934, 7, 7350, 9889]
    results = {}

    for seed in seeds:
        # 尝试从KIBA_E0_seed{seed}目录加载
        result_path = BASE_DIR / f'KIBA_E0_seed{seed}' / 'test_results.json'
        if result_path.exists():
            with open(result_path, 'r') as f:
                data = json.load(f)
            results[f'old_seed{seed}'] = {
                'ci': data.get('ci', None),
                'mse': data.get('mse', None),
                'rmse': data.get('rmse', None)
            }
        else:
            # 尝试从GIN_KIBA_alpha0_seed{seed}目录加载
            result_path = BASE_DIR / f'GIN_KIBA_alpha0_seed{seed}' / 'test_results.json'
            if result_path.exists():
                with open(result_path, 'r') as f:
                    data = json.load(f)
                results[f'old_seed{seed}'] = {
                    'ci': data.get('ci', None),
                    'mse': data.get('mse', None),
                    'rmse': data.get('rmse', None)
                }

    return results

def generate_report(corrected_results, old_results):
    """生成CI诊断报告"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'task': 'T0.3: CI指标核查',
        'literature_baseline': {
            'model': 'GraphDTA-GIN',
            'dataset': 'KIBA',
            'ci': 0.882,
            'mse': 0.147,
            'source': 'GraphDTA Bioinformatics 2021, Table 3'
        },
        'old_e0_results': {},
        'corrected_e0_results': {}
    }

    # 旧E0结果
    old_cis = []
    old_mses = []
    for key, data in old_results.items():
        if data['ci'] is not None:
            old_cis.append(data['ci'])
        if data['mse'] is not None:
            old_mses.append(data['mse'])

    if old_cis:
        report['old_e0_results'] = {
            'count': len(old_cis),
            'mean_ci': np.mean(old_cis),
            'std_ci': np.std(old_cis),
            'min_ci': np.min(old_cis),
            'max_ci': np.max(old_cis),
            'mean_mse': np.mean(old_mses) if old_mses else None,
            'per_seed': old_results
        }

    # 修正E0结果
    corrected_cis = []
    corrected_mses = []
    for key, data in corrected_results.items():
        if data['ci'] is not None:
            corrected_cis.append(data['ci'])
        if data['mse'] is not None:
            corrected_mses.append(data['mse'])

    if corrected_cis:
        report['corrected_e0_results'] = {
            'count': len(corrected_cis),
            'mean_ci': np.mean(corrected_cis),
            'std_ci': np.std(corrected_cis),
            'mean_mse': np.mean(corrected_mses) if corrected_mses else None,
            'per_seed': corrected_results
        }

    # 保存报告
    output_path = OUTPUT_DIR / 'task_v9_t0p3_ci_diagnosis_report.json'
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n报告已保存: {output_path}")

    # 打印摘要
    print("\n" + "="*80)
    print("V9 Task T0.3: CI指标核查 - 完成")
    print("="*80)

    print("\n文献基线 (GraphDTA-GIN, KIBA):")
    print(f"  CI = {report['literature_baseline']['ci']:.4f}")
    print(f"  MSE = {report['literature_baseline']['mse']:.4f}")

    print("\n旧E0结果 (旧协议, 10种子):")
    if report['old_e0_results']:
        print(f"  平均CI = {report['old_e0_results']['mean_ci']:.4f} ± {report['old_e0_results']['std_ci']:.4f}")
        print(f"  CI范围: [{report['old_e0_results']['min_ci']:.4f}, {report['old_e0_results']['max_ci']:.4f}]")
        if report['old_e0_results']['mean_mse']:
            print(f"  平均MSE = {report['old_e0_results']['mean_mse']:.4f}")
        print(f"  与文献CI差距: {report['literature_baseline']['ci'] - report['old_e0_results']['mean_ci']:.4f}")

    print("\n修正E0结果 (新协议, 2种子):")
    if report['corrected_e0_results']:
        print(f"  平均CI = {report['corrected_e0_results']['mean_ci']:.4f} ± {report['corrected_e0_results']['std_ci']:.4f}")
        if report['corrected_e0_results']['mean_mse']:
            print(f"  平均MSE = {report['corrected_e0_results']['mean_mse']:.4f}")
        print(f"  与文献CI差距: {report['literature_baseline']['ci'] - report['corrected_e0_results']['mean_ci']:.4f}")

    print("\n关键发现:")
    if report['old_e0_results'] and report['corrected_e0_results']:
        old_ci = report['old_e0_results']['mean_ci']
        corrected_ci = report['corrected_e0_results']['mean_ci']
        improvement = corrected_ci - old_ci
        print(f"  CI改进: {old_ci:.4f} → {corrected_ci:.4f} (+{improvement:.4f})")

        lit_ci = report['literature_baseline']['ci']
        if corrected_ci < lit_ci - 0.05:
            print(f"  ⚠️  修正E0 CI ({corrected_ci:.4f}) 仍低于文献CI ({lit_ci:.4f})")
            print(f"  ⚠️  可能存在第二个问题（数据管线或CI实现）")
        else:
            print(f"  ✅ 修正E0 CI ({corrected_ci:.4f}) 接近文献CI ({lit_ci:.4f})")

    return report

def main():
    print("="*80)
    print("V9 Task T0.3: CI指标核查")
    print("="*80)

    print("\n提取修正E0结果中的CI...")
    corrected_results = load_test_results_from_corrected_e0()
    for key, data in corrected_results.items():
        print(f"  {key}: CI={data['ci']:.4f}, MSE={data['mse']:.4f}" if data['ci'] else f"  {key}: 无数据")

    print("\n提取旧E0结果中的CI...")
    old_results = load_old_e0_ci()
    for key, data in old_results.items():
        if data['ci']:
            print(f"  {key}: CI={data['ci']:.4f}, MSE={data['mse']:.4f}")
        else:
            print(f"  {key}: 无CI数据")

    print("\n生成CI诊断报告...")
    report = generate_report(corrected_results, old_results)

    print("\n" + "="*80)
    print("Task T0.3 完成")
    print("="*80)

if __name__ == '__main__':
    main()
