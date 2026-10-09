#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结果聚合脚本
日期: 2026-08-04
目标: 聚合多个种子的实验结果，计算统计量
输入: 多个种子的results.json
输出: summary.md（Markdown格式的汇总报告）

功能:
  1. 读取所有种子的结果
  2. 计算均值、标准差、置信区间
  3. 进行配对t检验
  4. 计算Cohen's d效应量
  5. 生成Markdown报告
"""

import json
import numpy as np
import scipy.stats as stats
from pathlib import Path
from datetime import datetime


def load_results(experiment_dir):
    """
    加载实验的所有种子结果

    Args:
        experiment_dir: 实验目录路径

    Returns:
        list of dict: 所有种子的结果列表
    """
    results = []
    for seed_dir in sorted(experiment_dir.glob('seed_*')):
        results_file = seed_dir / 'results.json'
        if results_file.exists():
            with open(results_file, 'r') as f:
                result = json.load(f)
                result['seed'] = int(seed_dir.name.split('_')[1])
                results.append(result)
    return results


def compute_statistics(values):
    """
    计算统计量

    Args:
        values: 数值列表

    Returns:
        dict: 统计量字典
    """
    values = np.array(values)
    n = len(values)

    mean = np.mean(values)
    std = np.std(values, ddof=1) if n > 1 else 0.0
    sem = std / np.sqrt(n) if n > 0 else 0.0

    # 95%置信区间
    if n > 1:
        ci = stats.t.interval(0.95, n-1, loc=mean, scale=sem)
    else:
        ci = (mean, mean)

    return {
        'mean': float(mean),
        'std': float(std),
        'sem': float(sem),
        'ci_95': [float(ci[0]), float(ci[1])],
        'n': n,
        'min': float(np.min(values)),
        'max': float(np.max(values))
    }


def paired_t_test(values1, values2):
    """
    配对t检验

    Args:
        values1: 第一组数值
        values2: 第二组数值

    Returns:
        dict: 检验结果
    """
    values1 = np.array(values1)
    values2 = np.array(values2)

    # 确保长度相同
    n = min(len(values1), len(values2))
    values1 = values1[:n]
    values2 = values2[:n]

    if n < 2:
        return {
            't_statistic': None,
            'p_value': None,
            'significant': False
        }

    t_stat, p_value = stats.ttest_rel(values1, values2)

    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'significant': bool(p_value < 0.05)
    }


def cohens_d(values1, values2):
    """
    计算Cohen's d效应量

    Args:
        values1: 第一组数值
        values2: 第二组数值

    Returns:
        float: Cohen's d值
    """
    values1 = np.array(values1)
    values2 = np.array(values2)

    n1, n2 = len(values1), len(values2)
    mean1, mean2 = np.mean(values1), np.mean(values2)
    var1, var2 = np.var(values1, ddof=1), np.var(values2, ddof=1)

    # 合并标准差
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    d = (mean1 - mean2) / pooled_std
    return float(d)


def generate_summary(results_dir, output_file):
    """
    生成汇总报告

    Args:
        results_dir: 结果根目录
        output_file: 输出文件路径
    """
    print("=" * 60)
    print("结果聚合脚本")
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 收集所有实验结果
    experiments = {}
    for exp_dir in sorted(results_dir.glob('E*')):
        if exp_dir.is_dir() and exp_dir.name != 'E0_test':
            exp_name = exp_dir.name
            results = load_results(exp_dir)

            if results:
                # 按种子排序
                results.sort(key=lambda x: x['seed'])

                # 提取指标
                mse_values = [r['test_mse'] for r in results]
                rmse_values = [r['test_rmse'] for r in results]
                ci_values = [r['test_ci'] for r in results]

                experiments[exp_name] = {
                    'results': results,
                    'mse_stats': compute_statistics(mse_values),
                    'rmse_stats': compute_statistics(rmse_values),
                    'ci_stats': compute_statistics(ci_values)
                }

                print(f"✅ {exp_name}: {len(results)} 个种子")
            else:
                print(f"⚠️  {exp_name}: 无结果")

    # 以E0为基准进行对比
    if 'E0' in experiments:
        e0_mse = [r['test_mse'] for r in experiments['E0']['results']]

        for exp_name in experiments:
            if exp_name != 'E0':
                exp_mse = [r['test_mse'] for r in experiments[exp_name]['results']]

                # 配对t检验
                t_test = paired_t_test(e0_mse, exp_mse)

                # Cohen's d
                d = cohens_d(e0_mse, exp_mse)

                # 相对提升
                e0_mean = experiments['E0']['mse_stats']['mean']
                exp_mean = experiments[exp_name]['mse_stats']['mean']
                relative_change = (exp_mean - e0_mean) / e0_mean * 100

                experiments[exp_name]['comparison'] = {
                    't_test': t_test,
                    'cohens_d': d,
                    'relative_change': relative_change
                }

                print(f"\n{exp_name} vs E0:")
                print(f"  相对变化: {relative_change:+.2f}%")
                print(f"  p值: {t_test['p_value']:.4f}")
                print(f"  Cohen's d: {d:.2f}")
                print(f"  显著性: {'✅' if t_test['significant'] else '❌'}")

    # 生成Markdown报告
    with open(output_file, 'w') as f:
        f.write("# 受控实验结果汇总\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 主表
        f.write("## 主要结果\n\n")
        f.write("| 实验 | 种子数 | Test MSE (mean±std) | Test CI (mean±std) | vs E0 | p值 | Cohen's d | 显著性 |\n")
        f.write("|------|--------|---------------------|-------------------|-------|-----|-----------|--------|\n")

        for exp_name in sorted(experiments.keys()):
            exp = experiments[exp_name]
            n = exp['mse_stats']['n']
            mse_str = f"{exp['mse_stats']['mean']:.4f}±{exp['mse_stats']['std']:.4f}"
            ci_str = f"{exp['ci_stats']['mean']:.4f}±{exp['ci_stats']['std']:.4f}"

            if exp_name == 'E0':
                f.write(f"| **{exp_name}** | {n} | **{mse_str}** | **{ci_str}** | — | — | — | 基线 |\n")
            else:
                comp = exp['comparison']
                rel_change = f"{comp['relative_change']:+.1f}%"
                p_val = f"{comp['t_test']['p_value']:.4f}"
                d_val = f"{comp['cohens_d']:.2f}"
                sig = "✅" if comp['t_test']['significant'] else "❌"
                f.write(f"| {exp_name} | {n} | {mse_str} | {ci_str} | {rel_change} | {p_val} | {d_val} | {sig} |\n")

        # 详细统计
        f.write("\n## 详细统计\n\n")

        for exp_name in sorted(experiments.keys()):
            exp = experiments[exp_name]
            f.write(f"### {exp_name}\n\n")

            f.write(f"**样本数**: {exp['mse_stats']['n']}\n\n")

            f.write("**Test MSE**:\n")
            f.write(f"- 均值: {exp['mse_stats']['mean']:.4f}\n")
            f.write(f"- 标准差: {exp['mse_stats']['std']:.4f}\n")
            f.write(f"- 标准误: {exp['mse_stats']['sem']:.4f}\n")
            f.write(f"- 95% CI: [{exp['mse_stats']['ci_95'][0]:.4f}, {exp['mse_stats']['ci_95'][1]:.4f}]\n")
            f.write(f"- 范围: [{exp['mse_stats']['min']:.4f}, {exp['mse_stats']['max']:.4f}]\n\n")

            f.write("**Test CI**:\n")
            f.write(f"- 均值: {exp['ci_stats']['mean']:.4f}\n")
            f.write(f"- 标准差: {exp['ci_stats']['std']:.4f}\n")
            f.write(f"- 标准误: {exp['ci_stats']['sem']:.4f}\n")
            f.write(f"- 95% CI: [{exp['ci_stats']['ci_95'][0]:.4f}, {exp['ci_stats']['ci_95'][1]:.4f}]\n")
            f.write(f"- 范围: [{exp['ci_stats']['min']:.4f}, {exp['ci_stats']['max']:.4f}]\n\n")

            if exp_name != 'E0' and 'comparison' in exp:
                comp = exp['comparison']
                f.write("**与E0对比**:\n")
                f.write(f"- 相对变化: {comp['relative_change']:+.2f}%\n")
                f.write(f"- t统计量: {comp['t_test']['t_statistic']:.4f}\n")
                f.write(f"- p值: {comp['t_test']['p_value']:.4f}\n")
                f.write(f"- Cohen's d: {comp['cohens_d']:.2f}\n")
                f.write(f"- 显著性 (p<0.05): {'是 ✅' if comp['t_test']['significant'] else '否 ❌'}\n\n")

            # 各种子结果
            f.write("**各种子结果**:\n\n")
            f.write("| Seed | Test MSE | Test RMSE | Test CI |\n")
            f.write("|------|----------|-----------|--------|\n")

            for result in exp['results']:
                f.write(f"| {result['seed']} | {result['test_mse']:.4f} | {result['test_rmse']:.4f} | {result['test_ci']:.4f} |\n")

            f.write("\n")

    print(f"\n✅ 汇总报告已保存到: {output_file}")


if __name__ == '__main__':
    results_dir = Path('/home/rxp/dta2/results/controlled')
    output_file = results_dir / 'summary.md'

    generate_summary(results_dir, output_file)
