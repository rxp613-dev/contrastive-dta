#!/usr/bin/env python3
"""
V9 Phase 0.3: 统一统计检验脚本 (v2)
日期: 2026-11-04
目标: 为所有实验结果提供统一的统计检验，确保今后所有汇总数字只由此脚本产出

统计检验:
1. paired t-test: 同种子对比 (E0 vs E7)
2. Welch t-test: 独立样本对比 (不同配置)
3. Levene检验: 方差齐性检验
4. Hedges' g: 效应量计算

输入: 各实验的results.json文件
输出: 统计报告(JSON + Markdown)
"""

import json
import os
import glob
import numpy as np
from scipy import stats
from pathlib import Path
from datetime import datetime


def convert_numpy_types(obj):
    """递归转换numpy类型为Python原生类型"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def load_experiment_results(pattern, single_dir=False):
    """
    加载匹配pattern的所有实验结果

    Args:
        pattern: 目录pattern，如 "/mnt/data/dta2/results/controlled/v9_t1p1_*"
        single_dir: 如果为True，则pattern指向单个目录，其中包含多个results_seed*.json文件

    Returns:
        list of dicts, 每个dict包含seed, test_mse, test_ci等
    """
    results = []

    if single_dir:
        # 单个目录模式：加载该目录下所有results_seed*.json
        for result_file in glob.glob(os.path.join(pattern, "results_seed*.json")):
            with open(result_file) as f:
                data = json.load(f)
            results.append({
                'seed': data.get('seed'),
                'test_mse': data.get('test_mse'),
                'test_ci': data.get('test_ci'),
                'best_epoch': data.get('best_epoch'),
                'experiment_id': data.get('experiment_id')
            })
    else:
        # 多目录模式：每个子目录包含一个results.json
        for result_file in glob.glob(os.path.join(pattern, "results.json")):
            with open(result_file) as f:
                data = json.load(f)
            results.append({
                'seed': data.get('seed'),
                'test_mse': data.get('test_mse'),
                'test_ci': data.get('test_ci'),
                'best_epoch': data.get('best_epoch'),
                'experiment_id': data.get('experiment_id')
            })

    return sorted(results, key=lambda x: x['seed'] if x['seed'] else 0)


def paired_t_test(e0_results, e7_results):
    """配对t检验（同种子对比）"""
    e0_seeds = {r['seed']: r for r in e0_results}
    e7_seeds = {r['seed']: r for r in e7_results}
    common_seeds = sorted(set(e0_seeds.keys()) & set(e7_seeds.keys()))

    if len(common_seeds) < 2:
        return {'error': f'Not enough common seeds for paired test (only {len(common_seeds)})'}

    e0_mse = [e0_seeds[s]['test_mse'] for s in common_seeds]
    e7_mse = [e7_seeds[s]['test_mse'] for s in common_seeds]
    e0_ci = [e0_seeds[s]['test_ci'] for s in common_seeds]
    e7_ci = [e7_seeds[s]['test_ci'] for s in common_seeds]

    t_mse, p_mse = stats.ttest_rel(e0_mse, e7_mse)
    t_ci, p_ci = stats.ttest_rel(e0_ci, e7_ci)

    def hedges_g(group1, group2):
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        s_p = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        d = (mean1 - mean2) / s_p
        g = d * (1 - 3 / (4 * (n1 + n2) - 9))
        return g

    g_mse = hedges_g(e0_mse, e7_mse)
    g_ci = hedges_g(e0_ci, e7_ci)

    mse_improvement = (np.mean(e0_mse) - np.mean(e7_mse)) / np.mean(e0_mse) * 100
    ci_improvement = (np.mean(e7_ci) - np.mean(e0_ci)) / np.mean(e0_ci) * 100

    return {
        'n_pairs': len(common_seeds),
        'seeds': common_seeds,
        'e0_mse_mean': np.mean(e0_mse),
        'e0_mse_std': np.std(e0_mse, ddof=1),
        'e7_mse_mean': np.mean(e7_mse),
        'e7_mse_std': np.std(e7_mse, ddof=1),
        'mse_improvement_pct': mse_improvement,
        't_mse': t_mse,
        'p_mse': p_mse,
        'hedges_g_mse': g_mse,
        'e0_ci_mean': np.mean(e0_ci),
        'e0_ci_std': np.std(e0_ci, ddof=1),
        'e7_ci_mean': np.mean(e7_ci),
        'e7_ci_std': np.std(e7_ci, ddof=1),
        'ci_improvement_pct': ci_improvement,
        't_ci': t_ci,
        'p_ci': p_ci,
        'hedges_g_ci': g_ci
    }


def welch_t_test(group1, group2, label1="Group1", label2="Group2"):
    """Welch t检验（独立样本）"""
    t, p = stats.ttest_ind(group1, group2, equal_var=False)

    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    s_p = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    d = (mean1 - mean2) / s_p
    g = d * (1 - 3 / (4 * (n1 + n2) - 9))

    return {
        'n1': n1,
        'n2': n2,
        f'{label1}_mean': mean1,
        f'{label1}_std': std1,
        f'{label2}_mean': mean2,
        f'{label2}_std': std2,
        'mean_difference': mean1 - mean2,
        't_statistic': t,
        'p_value': p,
        'hedges_g': g
    }


def levene_test(group1, group2, label1="Group1", label2="Group2"):
    """Levene检验（方差齐性检验）"""
    stat, p = stats.levene(group1, group2)

    return {
        'n1': len(group1),
        'n2': len(group2),
        f'{label1}_variance': np.var(group1, ddof=1),
        f'{label2}_variance': np.var(group2, ddof=1),
        'variance_ratio': np.var(group1, ddof=1) / np.var(group2, ddof=1),
        'levene_statistic': stat,
        'p_value': p,
        'equal_variance': p > 0.05
    }


def analyze_experiment_pair(e0_pattern, e7_pattern, experiment_name, e0_single_dir=False, e7_single_dir=False):
    """分析一对E0/E7实验"""
    print(f"\n{'='*80}")
    print(f"分析: {experiment_name}")
    print(f"{'='*80}")

    e0_results = load_experiment_results(e0_pattern, single_dir=e0_single_dir)
    e7_results = load_experiment_results(e7_pattern, single_dir=e7_single_dir)

    print(f"E0种子数: {len(e0_results)}")
    print(f"E7种子数: {len(e7_results)}")

    if len(e0_results) == 0 or len(e7_results) == 0:
        print("警告: 数据不足，跳过分析")
        return {'error': 'Insufficient data'}

    e0_mse = [r['test_mse'] for r in e0_results]
    e7_mse = [r['test_mse'] for r in e7_results]
    e0_ci = [r['test_ci'] for r in e0_results]
    e7_ci = [r['test_ci'] for r in e7_results]

    paired_result = paired_t_test(e0_results, e7_results)
    welch_mse = welch_t_test(e0_mse, e7_mse, "E0", "E7")
    welch_ci = welch_t_test(e0_ci, e7_ci, "E0", "E7")
    levene_mse = levene_test(e0_mse, e7_mse, "E0", "E7")
    levene_ci = levene_test(e0_ci, e7_ci, "E0", "E7")

    result = {
        'experiment_name': experiment_name,
        'e0_pattern': e0_pattern,
        'e7_pattern': e7_pattern,
        'n_e0': len(e0_results),
        'n_e7': len(e7_results),
        'paired_test': paired_result,
        'welch_test_mse': welch_mse,
        'welch_test_ci': welch_ci,
        'levene_test_mse': levene_mse,
        'levene_test_ci': levene_ci,
        'timestamp': datetime.now().isoformat()
    }

    if 'error' not in paired_result:
        print(f"\n配对t检验 (MSE):")
        print(f"  E0: {paired_result['e0_mse_mean']:.4f}±{paired_result['e0_mse_std']:.4f}")
        print(f"  E7: {paired_result['e7_mse_mean']:.4f}±{paired_result['e7_mse_std']:.4f}")
        print(f"  改进: {paired_result['mse_improvement_pct']:.2f}%")
        print(f"  p值: {paired_result['p_mse']:.4f}")
        print(f"  Hedges' g: {paired_result['hedges_g_mse']:.2f}")
        print(f"  显著性: {'✅ 显著' if paired_result['p_mse'] < 0.05 else '❌ 不显著'}")

        print(f"\nLevene方差齐性检验 (MSE):")
        print(f"  E0方差: {levene_mse['E0_variance']:.6f}")
        print(f"  E7方差: {levene_mse['E7_variance']:.6f}")
        print(f"  p值: {levene_mse['p_value']:.4f}")
        print(f"  结论: {'方差齐性' if levene_mse['equal_variance'] else '方差不齐'}")

    return result


def generate_report(all_results, output_file):
    """生成统计报告"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'task': 'V9 Phase 0.3: 统一统计检验',
        'experiments': all_results
    }

    report = convert_numpy_types(report)

    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    md_file = output_file.replace('.json', '.md')
    with open(md_file, 'w') as f:
        f.write("# V9 统计检验报告\n\n")
        f.write(f"**生成时间**: {report['timestamp']}\n\n")
        f.write("---\n\n")

        for exp_name, exp_data in all_results.items():
            f.write(f"## {exp_name}\n\n")

            if 'error' in exp_data:
                f.write(f"**错误**: {exp_data['error']}\n\n")
                continue

            f.write(f"**样本量**: E0={exp_data['n_e0']}, E7={exp_data['n_e7']}\n\n")

            if 'error' not in exp_data['paired_test']:
                pt = exp_data['paired_test']
                f.write("### 配对t检验（同种子对比）\n\n")
                f.write(f"- E0 MSE: {pt['e0_mse_mean']:.4f}±{pt['e0_mse_std']:.4f}\n")
                f.write(f"- E7 MSE: {pt['e7_mse_mean']:.4f}±{pt['e7_mse_std']:.4f}\n")
                f.write(f"- MSE改进: {pt['mse_improvement_pct']:.2f}%\n")
                f.write(f"- p值: {pt['p_mse']:.4f}\n")
                f.write(f"- Hedges' g: {pt['hedges_g_mse']:.2f}\n")
                f.write(f"- 结论: {'✅ 显著 (p<0.05)' if pt['p_mse'] < 0.05 else '❌ 不显著'}\n\n")

                f.write(f"- E0 CI: {pt['e0_ci_mean']:.4f}±{pt['e0_ci_std']:.4f}\n")
                f.write(f"- E7 CI: {pt['e7_ci_mean']:.4f}±{pt['e7_ci_std']:.4f}\n")
                f.write(f"- CI改进: {pt['ci_improvement_pct']:.2f}%\n")
                f.write(f"- p值: {pt['p_ci']:.4f}\n")
                f.write(f"- Hedges' g: {pt['hedges_g_ci']:.2f}\n")
                f.write(f"- 结论: {'✅ 显著 (p<0.05)' if pt['p_ci'] < 0.05 else '❌ 不显著'}\n\n")

            lv_mse = exp_data['levene_test_mse']
            f.write("### Levene方差齐性检验\n\n")
            f.write(f"- E0 MSE方差: {lv_mse['E0_variance']:.6f}\n")
            f.write(f"- E7 MSE方差: {lv_mse['E7_variance']:.6f}\n")
            f.write(f"- 方差比: {lv_mse['variance_ratio']:.2f}\n")
            f.write(f"- p值: {lv_mse['p_value']:.4f}\n")
            f.write(f"- 结论: {'方差齐性' if lv_mse['equal_variance'] else '❌ 方差不齐'}\n\n")

            f.write("---\n\n")

    print(f"\n报告已保存:")
    print(f"  JSON: {output_file}")
    print(f"  Markdown: {md_file}")


def main():
    """主函数"""
    print("="*80)
    print("V9 Phase 0.3: 统一统计检验")
    print("="*80)

    experiments = {
        'T1.1_vs_T1.3_KIBA': {
            'e0_pattern': '/mnt/data/dta2/results/controlled/v9_t1p1_e0_10seeds_20261022_072628',
            'e7_pattern': '/mnt/data/dta2/results/controlled/v9_t1p3_e7_alpha005_seed*',
            'e0_single_dir': True,
            'e7_single_dir': False,
            'description': 'KIBA E0 vs E7 (α=0.05), Config B'
        },
        'T1.5_Davis': {
            'e0_pattern': '/mnt/data/dta2/results/controlled/v9_t1p5_davis_e0_seed*',
            'e7_pattern': '/mnt/data/dta2/results/controlled/v9_t1p5_davis_e7_seed*',
            'e0_single_dir': False,
            'e7_single_dir': False,
            'description': 'Davis E0 vs E7 (α=0.05), Config B'
        },
        'T1.6_Davis_ColdStart': {
            'e0_pattern': '/mnt/data/dta2/results/cold_start/v9_t1p6_e0_seed*',
            'e7_pattern': '/mnt/data/dta2/results/cold_start/v9_t1p6_e7_seed*',
            'e0_single_dir': False,
            'e7_single_dir': False,
            'description': 'Davis冷启动 E0 vs E7 (α=0.05), Config B'
        }
    }

    all_results = {}
    for exp_name, exp_config in experiments.items():
        result = analyze_experiment_pair(
            exp_config['e0_pattern'],
            exp_config['e7_pattern'],
            exp_config['description'],
            exp_config.get('e0_single_dir', False),
            exp_config.get('e7_single_dir', False)
        )
        all_results[exp_name] = result

    output_file = '/mnt/data/dta2/results/analysis/v9_phase0p3_statistical_tests.json'
    generate_report(all_results, output_file)

    print("\n" + "="*80)
    print("统计检验完成")
    print("="*80)


if __name__ == '__main__':
    main()
