#!/usr/bin/env python3
"""
P1.2: Final Verification Script (v2)
时间: 2026-09-08
目标:
  - P1.2.1: 终检脚本加入KIBA新表格权威值
  - P1.2.2: 终检脚本加入废弃值列表
  - P1.2.3: 全文扫描验证零残留
方法:
  - 扫描deprecated values（排除历史上下文中的mention）
  - 验证authoritative values存在
  - 生成verification report
"""

import re
from pathlib import Path
from datetime import datetime

def check_deprecated_values(paper_path):
    """Check for deprecated values that should have been removed"""

    with open(paper_path, 'r', encoding='utf-8') as f:
        content = f.read()

    violations = {}

    # Pattern 1: n=10 references (excluding future work and historical documentation)
    n10_pattern = r'n\s*=\s*10|n=10'
    n10_matches = list(re.finditer(n10_pattern, content, re.IGNORECASE))
    n10_violations = []
    for match in n10_matches:
        start = max(0, match.start() - 100)
        end = min(len(content), match.end() + 100)
        context = content[start:end]

        # Exclude if it's in future work section or historical documentation
        if 'future work' in context.lower() or 'post-hoc audit' in context.lower():
            continue

        line_num = content[:match.start()].count('\n') + 1
        n10_violations.append({
            'line': line_num,
            'match': match.group(),
            'context': context.replace('\n', ' ')
        })

    if n10_violations:
        violations['n=10 references (excluding future/historical)'] = n10_violations

    # Pattern 2: Old p-values (0.0165, 0.0124)
    old_p_pattern = r'p\s*=\s*0\.0165|p=0\.0165|p\s*=\s*0\.0124|p=0\.0124'
    old_p_matches = list(re.finditer(old_p_pattern, content, re.IGNORECASE))
    if old_p_matches:
        violations['Old p-values (0.0165, 0.0124)'] = []
        for match in old_p_matches:
            line_num = content[:match.start()].count('\n') + 1
            start = max(0, match.start() - 50)
            end = min(len(content), match.end() + 50)
            context = content[start:end].replace('\n', ' ')
            violations['Old p-values (0.0165, 0.0124)'].append({
                'line': line_num,
                'match': match.group(),
                'context': context
            })

    # Pattern 3: α=0.50 references (excluding error documentation)
    alpha_pattern = r'α\s*=\s*0\.50|alpha\s*=\s*0\.50|α=0\.50'
    alpha_matches = list(re.finditer(alpha_pattern, content, re.IGNORECASE))
    alpha_violations = []
    for match in alpha_matches:
        start = max(0, match.start() - 100)
        end = min(len(content), match.end() + 100)
        context = content[start:end]

        # Exclude if it's documenting the error
        if 'previous version' in context.lower() or 'was an error' in context.lower() or 'no' in context.lower():
            continue

        line_num = content[:match.start()].count('\n') + 1
        alpha_violations.append({
            'line': line_num,
            'match': match.group(),
            'context': context.replace('\n', ' ')
        })

    if alpha_violations:
        violations['α=0.50 references (excluding error docs)'] = alpha_violations

    # Pattern 4: +13.97% references (excluding error documentation)
    improvement_pattern = r'\+13\.97\s*\\%|\+13\.97\s*%'
    improvement_matches = list(re.finditer(improvement_pattern, content, re.IGNORECASE))
    improvement_violations = []
    for match in improvement_matches:
        start = max(0, match.start() - 100)
        end = min(len(content), match.end() + 100)
        context = content[start:end]

        # Exclude if it's documenting the error
        if 'previous version' in context.lower() or 'was an error' in context.lower():
            continue

        line_num = content[:match.start()].count('\n') + 1
        improvement_violations.append({
            'line': line_num,
            'match': match.group(),
            'context': context.replace('\n', ' ')
        })

    if improvement_violations:
        violations['+13.97% references (excluding error docs)'] = improvement_violations

    # Pattern 5: Batch 1 references (excluding historical documentation)
    batch_pattern = r'Batch\s*1|batch\s*1|Batch-1'
    batch_matches = list(re.finditer(batch_pattern, content, re.IGNORECASE))
    batch_violations = []
    for match in batch_matches:
        start = max(0, match.start() - 100)
        end = min(len(content), match.end() + 100)
        context = content[start:end]

        # Exclude if it's in appendix or historical documentation
        if 'appendix' in context.lower() or 'previous version' in context.lower():
            continue

        line_num = content[:match.start()].count('\n') + 1
        batch_violations.append({
            'line': line_num,
            'match': match.group(),
            'context': context.replace('\n', ' ')
        })

    if batch_violations:
        violations['Batch 1 references (excluding appendix/history)'] = batch_violations

    # Pattern 6: 13.5% inflation claims
    inflation_pattern = r'13\.5%.*inflation|inflation.*13\.5%'
    inflation_matches = list(re.finditer(inflation_pattern, content, re.IGNORECASE))
    if inflation_matches:
        violations['13.5% inflation claims'] = []
        for match in inflation_matches:
            line_num = content[:match.start()].count('\n') + 1
            start = max(0, match.start() - 50)
            end = min(len(content), match.end() + 50)
            context = content[start:end].replace('\n', ' ')
            violations['13.5% inflation claims'].append({
                'line': line_num,
                'match': match.group(),
                'context': context
            })

    # Pattern 7: Sequential testing (Pocock, adaptive sample size)
    sequential_pattern = r'Pocock|sequential.*testing|adaptive.*sample.*size'
    sequential_matches = list(re.finditer(sequential_pattern, content, re.IGNORECASE))
    if sequential_matches:
        violations['Sequential testing (Pocock, adaptive)'] = []
        for match in sequential_matches:
            line_num = content[:match.start()].count('\n') + 1
            start = max(0, match.start() - 50)
            end = min(len(content), match.end() + 50)
            context = content[start:end].replace('\n', ' ')
            violations['Sequential testing (Pocock, adaptive)'].append({
                'line': line_num,
                'match': match.group(),
                'context': context
            })

    # Pattern 8: Wilcoxon min p=0.002
    wilcoxon_pattern = r'min\s*p\s*=\s*0\.002|min p=0.002'
    wilcoxon_matches = list(re.finditer(wilcoxon_pattern, content, re.IGNORECASE))
    if wilcoxon_matches:
        violations['Wilcoxon min p=0.002'] = []
        for match in wilcoxon_matches:
            line_num = content[:match.start()].count('\n') + 1
            start = max(0, match.start() - 50)
            end = min(len(content), match.end() + 50)
            context = content[start:end].replace('\n', ' ')
            violations['Wilcoxon min p=0.002'].append({
                'line': line_num,
                'match': match.group(),
                'context': context
            })

    # Pattern 9: Deprecated significance range claims (α ∈ [0.001, 0.05])
    # After Holm(6) correction, only α ≤ 0.005 is significant (p=0.033, 0.011)
    # α ∈ [0.01, 0.05] is directional but non-significant (p=0.281-0.759)
    # Only flag if claiming significance for the ENTIRE [0.001, 0.05] range
    range_pattern = r'significant.*degrad.*α\s*∈\s*\[0\.001,\s*0\.05\]|significant.*harm.*α\s*∈\s*\[0\.001,\s*0\.05\]'
    range_matches = list(re.finditer(range_pattern, content, re.IGNORECASE))
    if range_matches:
        violations['Deprecated significance range [0.001, 0.05]'] = []
        for match in range_matches:
            line_num = content[:match.start()].count('\n') + 1
            start = max(0, match.start() - 80)
            end = min(len(content), match.end() + 80)
            context = content[start:end].replace('\n', ' ')
            violations['Deprecated significance range [0.001, 0.05]'].append({
                'line': line_num,
                'match': match.group(),
                'context': context
            })

    # Pattern 10: GitHub dead link
    github_pattern = r'github\.com/contrastive-dta'
    github_matches = list(re.finditer(github_pattern, content, re.IGNORECASE))
    if github_matches:
        violations['GitHub dead link (contrastive-dta)'] = []
        for match in github_matches:
            line_num = content[:match.start()].count('\n') + 1
            start = max(0, match.start() - 50)
            end = min(len(content), match.end() + 50)
            context = content[start:end].replace('\n', ' ')
            violations['GitHub dead link (contrastive-dta)'].append({
                'line': line_num,
                'match': match.group(),
                'context': context
            })

    return violations


def check_authoritative_values(paper_path):
    """Verify that authoritative KIBA values are present"""

    with open(paper_path, 'r', encoding='utf-8') as f:
        content = f.read()

    authoritative_checks = {
        'KIBA α=0.1 improvement': r'\+17\.76\\%|\$\+\$17\.76\\%',
        'KIBA p=0.759 (Holm family=6)': r'p\s*=\s*0\.759',
        'KIBA p=0.033 (α=0.001)': r'p\s*=\s*0\.033',
        'KIBA p=0.011 (α=0.005)': r'p\s*=\s*0\.011',
        'KIBA n=5': r'KIBA.*n\s*=\s*5',
        'Dataset-dependent sign reversal': r'sign.*reversal|reversal.*sign',
        'KIBA small α degradation': r'small.*\$\\alpha\$.*degrad',
        'β=0.0 consistency': r'β\s*=\s*0\.0|beta\s*=\s*0\.0',
        'InfoNCE alone sufficient': r'InfoNCE.*alone.*sufficient',
        'Holm family size = 6': r'family\s*size\s*=\s*6',
        'tang2014kiba citation': r'tang2014kiba',
        'α ≤ 0.005 significance': r'\\alpha\s*\\leq?\s*0\.005',
    }

    found = {}
    missing = {}

    for check_name, pattern in authoritative_checks.items():
        matches = list(re.finditer(pattern, content, re.IGNORECASE))
        if matches:
            found[check_name] = len(matches)
        else:
            missing[check_name] = pattern

    return found, missing


def main():
    print("=" * 80)
    print("P1.2: Final Verification Script (v2)")
    print("=" * 80)
    print()

    paper_path = Path("/home/rxp/dta2/paper/main_paper.tex")

    if not paper_path.exists():
        print(f"ERROR: Paper not found at {paper_path}")
        return

    # Check 1: Deprecated values
    print("Check 1: Scanning for deprecated values...")
    print("-" * 80)
    violations = check_deprecated_values(paper_path)

    total_violations = sum(len(v) for v in violations.values())

    if total_violations == 0:
        print("✅ PASS: No deprecated values found")
    else:
        print(f"❌ FAIL: Found {total_violations} deprecated value(s)")
        for category, items in violations.items():
            if items:
                print(f"\n{category} ({len(items)} violation(s)):")
                for item in items:
                    print(f"  Line {item['line']}: '{item['match']}'")
                    print(f"    Context: ...{item['context']}...")

    # Check 2: Authoritative values
    print("\n" + "=" * 80)
    print("Check 2: Verifying authoritative values...")
    print("-" * 80)
    found, missing = check_authoritative_values(paper_path)

    print(f"✅ Found {len(found)} authoritative value(s):")
    for check_name, count in found.items():
        print(f"  {check_name}: {count} occurrence(s)")

    if missing:
        print(f"\n⚠️  WARNING: Missing {len(missing)} authoritative value(s):")
        for check_name, pattern in missing.items():
            print(f"  {check_name}: pattern '{pattern}' not found")
    else:
        print("\n✅ PASS: All authoritative values present")

    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    if total_violations == 0 and not missing:
        print("✅ ALL CHECKS PASSED")
        print("   - No deprecated values found")
        print("   - All authoritative values present")
        print("   - Paper is ready for compilation")
        status = "PASS"
    else:
        print("❌ VERIFICATION FAILED")
        if total_violations > 0:
            print(f"   - {total_violations} deprecated value(s) found")
        if missing:
            print(f"   - {len(missing)} authoritative value(s) missing")
        print("   - Please fix issues before compilation")
        status = "FAIL"

    # Generate report
    report_path = Path("/home/rxp/dta2/results/analysis/p1_2_verification_report.md")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Final Verification Report (v2)\n\n")
        f.write(f"**Time**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Status**: {status}\n\n")

        f.write("## Deprecated Values Check\n\n")
        if total_violations == 0:
            f.write("✅ No deprecated values found\n\n")
        else:
            f.write(f"❌ Found {total_violations} deprecated value(s)\n\n")
            for category, items in violations.items():
                if items:
                    f.write(f"### {category} ({len(items)} violation(s))\n\n")
                    for item in items:
                        f.write(f"- Line {item['line']}: `{item['match']}`\n")
                        f.write(f"  - Context: `{item['context']}`\n")
                    f.write("\n")

        f.write("## Authoritative Values Check\n\n")
        f.write(f"✅ Found {len(found)} authoritative value(s):\n\n")
        for check_name, count in found.items():
            f.write(f"- {check_name}: {count} occurrence(s)\n")

        if missing:
            f.write(f"\n⚠️ Missing {len(missing)} authoritative value(s):\n\n")
            for check_name, pattern in missing.items():
                f.write(f"- {check_name}: pattern `{pattern}` not found\n")
        else:
            f.write("\n✅ All authoritative values present\n")

        f.write("\n## Conclusion\n\n")
        if status == "PASS":
            f.write("✅ Paper is ready for compilation and submission\n")
        else:
            f.write("❌ Paper requires fixes before compilation\n")

    print(f"\nDetailed report saved to: {report_path}")

    return status == "PASS"


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
