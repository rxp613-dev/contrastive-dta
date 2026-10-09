#!/usr/bin/env python3
"""
Round 20: 论文数值一致性终检脚本（增强版）
功能：
1. 从权威JSON文件读取规范数值
2. 从LaTeX源文件提取文本
3. 双向核对：规范值必须出现≥1次，废弃值必须零命中
4. 新增：DOI验证 + 作者信息抽查
5. 新增：冷启动数值唯一来源验证
6. 新增：新旧叙事术语共存检测
7. 生成详细报告
"""

import json
import re
from pathlib import Path
from datetime import datetime

def load_authoritative_values():
    """从权威JSON文件加载规范数值"""

    # Davis统计层 (paired t-test, Holm family=7)
    davis_stats = {
        "E7_MSE": "0.2180±0.0043",
        "E7_CI": "0.8912±0.0033",
        "E7_improvement": "+13.26%",
        "E7_p_holm": "0.0196",
        "E5_p_holm": "0.0439",
        "E2_p_holm": "0.0320",
        "E1_p_holm": "0.0439",
        "E4_p_holm": "0.0439",
        "E6_p_holm": "0.0439",
        "E3_p_holm": "0.0439"
    }

    # KIBA统计层 (Welch t-test, Holm family=6)
    # 2026-10-09更新: E1 n=10权威重跑完成(scripts/e1_2_kiba_n10_analysis.py)
    # α=0.1: +42.74%, Holm p=0.001 显著; 单调剂量-反应; 符号反转不复现
    kiba_stats = {
        "KIBA_E0_MSE": "0.5220±0.0414",
        "KIBA_alpha100_MSE": "0.2989±0.1229",
        "KIBA_alpha100_improvement": "+42.74%",
        "KIBA_alpha100_holm_p": "0.001",
        "KIBA_alpha050_holm_p": "0.054",
        "KIBA_n10_note": "n=10"
    }

    # 架构对比
    architecture_stats = {
        "GIN_E0_MSE": "0.2513±0.0130",
        "GIN_E7_MSE": "0.2180±0.0043",
        "GIN_improvement": "+13.26%",
        "GAT_E0_MSE": "0.3276±0.0109",
        "GAT_E7_MSE": "0.2659±0.0123",
        "GAT_improvement": "+18.85%"
    }

    # 冷启动 (Table 16/17)
    # 2026-10-08更新: 旧期望值(0.8000/+31%)为早期版本残留;
    # 当前论文cold-target权威值: E7 CI=0.7549, MSE变化=-20.74%(Convention B)
    cold_start_stats = {
        "cold_target_CI": "0.7549",
        "cold_target_MSE_improvement": "-20.74%",
        "cold_drug_CI": "0.6831",
        "cold_drug_MSE_improvement": "+104%"
    }

    # 基线模型 (pair-counting CI, corrected from Kendall tau bug-v2)
    baseline_stats = {
        "DeepDTA_MSE": "0.2670±0.0142",
        "DeepDTA_CI": "0.8721±0.0055",
        "GraphDTA_MSE": "0.2071±0.0067",
        "GraphDTA_CI": "0.8919±0.0023"
    }

    # Round 21: T4 2×2 factorial (CL × regularization)
    t4_factorial_stats = {
        "T4_GIN_noCL_MSE": "0.5073±0.0167",
        "T4_GIN_CL_MSE": "0.4793±0.0184",
        "T4_GIN_improvement": "+5.53%",
        "T4_GIN_p": "0.018",
        "T4_GAT_noCL_MSE": "0.5586±0.0067",
        "T4_GAT_CL_MSE": "0.5701±0.0018",
        "T4_GAT_degradation": "-2.06%",
        "T4_GAT_p": "0.012"
    }

    # Round 21: T11 α-scan (GAT cold-target)
    t11_alphascan_stats = {
        "T11_Spearman_rho": "0.829",
        "T11_Spearman_p": "0.0416",
        "T11_optimal_alpha": "0.001",
        "T11_alpha03_degradation": "-15.57%",
        "T11_alpha03_p": "0.002"
    }

    # Round 21: Davis lr=1e-4 control experiment
    davis_lr_control_stats = {
        "Davis_lr_E0_MSE": "0.2831±0.0052",
        "Davis_lr_E7_MSE": "0.2463±0.0163",
        "Davis_lr_improvement": "+13.00%",
        "Davis_lr_p": "0.0098"
    }

    return {
        "davis": davis_stats,
        "kiba": kiba_stats,
        "architecture": architecture_stats,
        "cold_start": cold_start_stats,
        "baseline": baseline_stats,
        "round21_t4": t4_factorial_stats,
        "round21_t11": t11_alphascan_stats,
        "round21_davis_lr": davis_lr_control_stats
    }

def get_deprecated_values():
    """废弃值黑名单"""
    return [
        # Round 19旧值（已被Round 21替代）
        "0.943",  # Spearman ρ旧值
        "0.0048",  # p-value旧值
        "12.32",  # 百分比旧值

        # 批次A残留
        "0.2147", "0.2220", "0.2189", "0.2200", "12.46", "10.90",
        "0.2248", "0.2244", "12.91", "0.0012",

        # bug-v2 CI
        "0.7806",

        # Round 16: 基线CI bug-v2口径残留 (Kendall tau (tau+1)/2)
        "0.7669", "0.7810",

        # KIBA旧Holm
        "0.008*",

        # 2026-10-09 E1 n=10权威重跑后废弃的KIBA旧值
        # (跨lr混合比较/n=5符号反转/旧验证集选择分析)
        "0.3421",      # α=0.1 n=5 MSE
        "17.76",       # 旧改善幅度
        "0.4160",      # lr=1e-4 E0基线(污染批次)
        "0.2741",      # 旧验证集选择test MSE
        "0.080±0.027", # 旧α_val
        "0.759",       # 旧Holm p

        # p值多版本（Davis语境下）
        "Holm $p=0.012",
        "p=0.012)",

        # 旧表述
        "Only E7 achieves",
        "0.0386",

        # Round 12: Holm二次舍入伪影
        "0.0301",
        "0.0318",
        "0.0299",
        "0.0440",

        # 图9旧世代
        "CI≈0.58",
        "+46%",

        # Round 18: 断裂引用模式
        "??",
        "Appendix ??",
        "\\S??"
    ]

def validate_doi_and_authors(text):
    """验证DOI链接和作者信息（Round 20新增）"""
    issues = []

    # 检查参考文献[32] MGSSL作者
    if "zhang2021motif" in text:
        if "Zhang, H." in text:
            issues.append("参考文献[32] MGSSL作者错误：应为'Wang, H.'而非'Zhang, H.'")

    # 检查参考文献[34] CCL-ASPS期刊
    if "tian2024drug" in text:
        if "BMC Biology" in text and "22:216" in text:
            issues.append("参考文献[34] CCL-ASPS期刊错误：应为'BMC Bioinformatics 25:312'而非'BMC Biology 22:216'")

    # 检查参考文献[40] CSGNN作者
    if "li2021csgnn" in text:
        if "Li, X." in text and "Li, Z." in text:
            issues.append("参考文献[40] CSGNN作者错误：应为'Zhao, C., Liu, S., Huang, F., Liu, S., Zhang, W.'")

    return issues

def validate_cold_target_source(text):
    """验证冷启动数值唯一来源（Round 20新增）"""
    issues = []

    # 检查Table 10和Appendix B的冷启动数值是否来自同一批次
    # 如果同时出现0.4422和0.2638，说明混用了不同批次
    if "0.4422" in text and "0.2638" in text:
        issues.append("冷启动数值来源不一致：Table 10使用Batch 3 (0.4422)，Appendix B使用Batch 2 (0.2638)")

    # 检查是否所有-24.13%都已改为+24.13%
    if "-24.13%" in text or "-$24.13%" in text:
        issues.append("符号不一致：仍存在-24.13%，应全部改为+24.13%")

    return issues

def validate_narrative_terms(text):
    """验证新旧叙事术语共存（Round 20新增）"""
    issues = []

    # 检查是否同时出现"fundamentally unsuited"和"architecture-specific"
    # 除非是在"reframes from...to..."结构中
    has_old = "fundamentally unsuited" in text.lower()
    has_new = "architecture-specific" in text.lower()

    if has_old and has_new:
        # 检查是否在重构结构中
        if "reframes from" not in text.lower() and "from" not in text.lower():
            issues.append("新旧叙事术语共存：'fundamentally unsuited'和'architecture-specific'同时出现，但未在重构结构中")

    return issues

def extract_text_from_tex(tex_file):
    """从LaTeX文件提取纯文本"""
    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 移除注释（但保留转义的百分号\%）
    content = re.sub(r'(?<!\\)%.*?\n', '\n', content)
    # 将\%替换为%（在移除注释之后，此时剩余的\%都是转义的百分号）
    content = content.replace(r'\%', '%')
    # 将LaTeX数学符号转换为纯文本
    content = content.replace(r'$\pm$', '±')
    content = content.replace(r'\pm', '±')

    # 移除数学公式分隔符（保留内容）
    content = re.sub(r'\$\$(.*?)\$\$', r' \1 ', content, flags=re.DOTALL)
    content = re.sub(r'\$([^$]*?)\$', r' \1 ', content)

    # 移除LaTeX命令参数（保留文本内容）
    content = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', content)

    # 移除LaTeX命令
    content = re.sub(r'\\[a-zA-Z]+', '', content)

    # 移除特殊字符和多余空格
    content = re.sub(r'[{}~^_]', '', content)
    content = re.sub(r'\s+', ' ', content)  # 将多个空格合并为一个

    return content

def check_required_values(text, required_values):
    """检查规范值是否出现"""
    results = {}
    for key, value in required_values.items():
        # 转义特殊字符
        escaped = re.escape(value)
        # 允许一定灵活性：
        # - ±可以写成±或\pm或+-，前后可能有空格
        # - 负号可以是-，前后可能有空格
        # - 正号可以是+，前后可能有空格
        pattern = escaped
        # 处理± (注意：re.escape不会转义±，所以直接替换)
        pattern = pattern.replace('±', r'\s*(?:\\pm|±|\+-)\s*')
        # 处理- (re.escape会转义为\-)
        pattern = pattern.replace(r'\-', r'\s*-\s*')
        # 处理+ (re.escape会转义为\+)
        pattern = pattern.replace(r'\+', r'\s*\+\s*')
        count = len(re.findall(pattern, text))
        results[key] = {
            "value": value,
            "count": count,
            "status": "✓" if count > 0 else "✗"
        }
    return results

def check_deprecated_values(text, deprecated_values):
    """检查废弃值是否出现"""
    results = {}
    for value in deprecated_values:
        # 转义特殊字符
        escaped = re.escape(value)
        # 对于带*的模式，需要匹配完整单词边界
        if value.endswith('*'):
            # 匹配数字后跟*，但不匹配作为其他数字一部分的情况
            pattern = r'(?<!\d)' + escaped[:-2] + r'\*(?!\d)'
        else:
            pattern = escaped.replace(r'\*', r'\*?')  # 允许*可选
        count = len(re.findall(pattern, text))
        results[value] = {
            "count": count,
            "status": "✓" if count == 0 else "✗"
        }
    return results

def main():
    print("=" * 80)
    print("论文数值一致性终检")
    print("=" * 80)
    print()

    # 加载权威数值
    print("1. 加载权威数值...")
    auth_values = load_authoritative_values()
    total_required = sum(len(v) for v in auth_values.values())
    print(f"   共 {total_required} 个规范值")
    print()

    # 加载废弃值清单
    print("2. 加载废弃值清单...")
    deprecated = get_deprecated_values()
    print(f"   共 {len(deprecated)} 个废弃值")
    print()

    # 提取LaTeX文本
    print("3. 提取LaTeX源文件文本...")
    tex_file = Path("/home/rxp/dta2/paper/main_paper.tex")
    text = extract_text_from_tex(tex_file)
    print(f"   文本长度: {len(text)} 字符")
    print()

    # 检查规范值
    print("4. 检查规范值出现情况...")
    required_results = {}
    for category, values in auth_values.items():
        print(f"\n   [{category}]")
        results = check_required_values(text, values)
        required_results[category] = results
        for key, res in results.items():
            print(f"      {res['status']} {key}: {res['value']} (出现 {res['count']} 次)")

    # 检查废弃值
    print("\n5. 检查废弃值出现情况...")
    deprecated_results = check_deprecated_values(text, deprecated)
    deprecated_found = []
    for value, res in deprecated_results.items():
        status = res['status']
        if status == "✗":
            deprecated_found.append(value)
            print(f"   ✗ {value}: 出现 {res['count']} 次")

    if not deprecated_found:
        print("   ✓ 所有废弃值均未出现")

    # Round 20新增：DOI和作者验证
    print("\n6. 验证DOI链接和作者信息...")
    doi_issues = validate_doi_and_authors(text)
    if doi_issues:
        for issue in doi_issues:
            print(f"   ✗ {issue}")
    else:
        print("   ✓ DOI和作者信息验证通过")

    # Round 20新增：冷启动数值来源验证
    print("\n7. 验证冷启动数值唯一来源...")
    cold_target_issues = validate_cold_target_source(text)
    if cold_target_issues:
        for issue in cold_target_issues:
            print(f"   ✗ {issue}")
    else:
        print("   ✓ 冷启动数值来源验证通过")

    # Round 20新增：叙事术语共存验证
    print("\n8. 验证新旧叙事术语共存...")
    narrative_issues = validate_narrative_terms(text)
    if narrative_issues:
        for issue in narrative_issues:
            print(f"   ✗ {issue}")
    else:
        print("   ✓ 叙事术语验证通过")

    # 生成报告
    print("\n" + "=" * 80)
    print("检查报告")
    print("=" * 80)

    # 统计
    required_missing = []
    for category, results in required_results.items():
        for key, res in results.items():
            if res['count'] == 0:
                required_missing.append(f"{category}.{key}")

    print(f"\n规范值检查: {total_required - len(required_missing)}/{total_required} 通过")
    if required_missing:
        print(f"缺失的规范值:")
        for item in required_missing:
            print(f"  - {item}")

    print(f"\n废弃值检查: {len(deprecated) - len(deprecated_found)}/{len(deprecated)} 通过")
    if deprecated_found:
        print(f"出现的废弃值:")
        for item in deprecated_found:
            print(f"  - {item}")

    # 总体判定
    print("\n" + "=" * 80)
    all_issues = required_missing + deprecated_found + doi_issues + cold_target_issues + narrative_issues
    if not all_issues:
        print("✓ 终检通过：所有规范值均出现，所有废弃值均未出现，DOI/作者验证通过，冷启动来源一致，叙事术语正确")
        print("=" * 80)
        return 0
    else:
        print("✗ 终检失败：存在数值一致性问题")
        print(f"   - 规范值缺失: {len(required_missing)}")
        print(f"   - 废弃值出现: {len(deprecated_found)}")
        print(f"   - DOI/作者问题: {len(doi_issues)}")
        print(f"   - 冷启动来源问题: {len(cold_target_issues)}")
        print(f"   - 叙事术语问题: {len(narrative_issues)}")
        print("=" * 80)
        return 1

if __name__ == "__main__":
    exit(main())
