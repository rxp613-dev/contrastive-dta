#!/usr/bin/env python3
"""
P0.1.1-KIBA审计脚本: 列出所有KIBA实验目录并读取真实alpha值
时间: 2026-09-08
目标:
  1. 列出所有KIBA实验目录
  2. 从results.json读取config.alpha的真实值（不信目录名）
  3. 统计不同批次的种子数（n=5 vs n=10）
  4. 产出KIBA审计表
方法:
  - 扫描results/controlled下所有*KIBA*目录
  - 读取每个目录的results.json中的config.alpha和test_mse
  - 按(alpha, batch)分组统计
  - 输出审计表到results/analysis/p0_1_1_kiba_audit.json和.md
"""

import json
import os
import re
from pathlib import Path
from collections import defaultdict
import datetime

def parse_dir_name(dirname):
    """解析目录名,提取alpha(目录名值)、lr、seed、batch标识"""
    info = {
        'dirname': dirname,
        'dir_alpha': None,
        'lr': None,
        'seed': None,
        'batch': None,
        'model': None,
    }

    # 识别模型
    if dirname.startswith('GIN_KIBA'):
        info['model'] = 'GIN'
    elif dirname.startswith('KIBA'):
        info['model'] = 'GAT'  # 默认KIBA前缀为GAT

    # 提取alpha (目录名)
    m = re.search(r'alpha(\d+)', dirname)
    if m:
        info['dir_alpha'] = int(m.group(1))

    # 提取lr
    m = re.search(r'lr([0-9.]+)', dirname)
    if m:
        info['lr'] = m.group(1)

    # 提取seed
    m = re.search(r'seed(\d+)', dirname)
    if m:
        info['seed'] = int(m.group(1))

    # 识别batch标识
    if 'old' in dirname:
        info['batch'] = 'old'
    elif 'fixed100' in dirname:
        info['batch'] = 'fixed100'
    elif 'lr1e4' in dirname:
        info['batch'] = 'lr1e4'
    elif 'best' in dirname:
        info['batch'] = 'best'
    elif 'E0' in dirname or 'E2' in dirname or 'E3' in dirname or 'E5' in dirname or 'E7' in dirname:
        info['batch'] = 'ablation'
    elif 'lr0.0001' in dirname:
        info['batch'] = 'lr0.0001'  # 可能是alpha扫描
    else:
        info['batch'] = 'unknown'

    return info


def read_results(exp_dir):
    """读取results.json"""
    results_file = exp_dir / 'results.json'
    if not results_file.exists():
        return None

    try:
        with open(results_file) as f:
            data = json.load(f)
        return {
            'config_alpha': data.get('config', {}).get('alpha'),
            'config_lr': data.get('config', {}).get('lr'),
            'config_weight_decay': data.get('config', {}).get('weight_decay'),
            'config_beta': data.get('config', {}).get('beta'),
            'config_gamma': data.get('config', {}).get('gamma'),
            'config_augmentation': data.get('config', {}).get('augmentation'),
            'config_use_contrastive': data.get('config', {}).get('use_contrastive'),
            'test_mse': data.get('test_mse'),
            'test_ci': data.get('test_ci'),
            'best_val_mse': data.get('best_val_mse'),
            'best_epoch': data.get('best_epoch'),
            'experiment_id': data.get('experiment_id'),
        }
    except Exception as e:
        return {'error': str(e)}


def main():
    print("=" * 80)
    print("P0.1.1: KIBA实验全量审计")
    print("=" * 80)
    print()

    base_dir = Path("/home/rxp/dta2/results/controlled")

    # 收集所有KIBA目录
    all_dirs = []
    for d in sorted(base_dir.glob("*KIBA*")):
        if not d.is_dir():
            continue
        # 跳过old子目录内的内容
        if "old" in d.parts[-1] and d.parts[-1].endswith("_old"):
            # 仍纳入审计,但标记为old批次
            pass
        all_dirs.append(d)

    print(f"找到 {len(all_dirs)} 个KIBA目录")
    print()

    # 按模型和batch分组
    by_model_batch = defaultdict(list)

    audit_records = []

    for d in all_dirs:
        dirname = d.name
        dir_info = parse_dir_name(dirname)
        results = read_results(d)

        if results is None:
            print(f"  WARNING: {dirname} 无results.json")
            continue
        if 'error' in results:
            print(f"  ERROR: {dirname} 读取失败: {results['error']}")
            continue

        record = {
            'dirname': dirname,
            'path': str(d),
            **dir_info,
            **results,
        }
        audit_records.append(record)

        # 按model + batch分组
        key = (dir_info['model'], dir_info['batch'])
        by_model_batch[key].append(record)

    # 输出分组统计
    print("=" * 80)
    print("按(model, batch)分组统计:")
    print("=" * 80)

    for key in sorted(by_model_batch.keys()):
        model, batch = key
        records = by_model_batch[key]
        alphas = sorted(set(r['config_alpha'] for r in records if r['config_alpha'] is not None))
        seeds = sorted(set(r['seed'] for r in records if r['seed'] is not None))

        print(f"\n[{model}, {batch}]")
        print(f"  目录数: {len(records)}")
        print(f"  真实alpha值(config.alpha): {alphas}")
        print(f"  种子列表(n={len(seeds)}): {seeds}")

        # 检查目录名alpha vs 真实alpha是否一致
        for r in records:
            if r['dir_alpha'] is not None and r['config_alpha'] is not None:
                dir_val = r['dir_alpha'] / 1000.0  # 目录名是整数表示
                # 目录命名惯例: alpha001 = 0.001, alpha010 = 0.01, alpha100 = 0.1
                # 但KIBA_alpha050可能指0.05或0.50 - 需要看真实值
                if abs(dir_val - r['config_alpha']) > 1e-6:
                    print(f"    ⚠️  不一致: {r['dirname']} 目录名alpha={dir_val}, 真实alpha={r['config_alpha']}")

    # 输出完整审计表
    print("\n" + "=" * 80)
    print("完整审计表(按dirname排序):")
    print("=" * 80)

    for r in sorted(audit_records, key=lambda x: x['dirname']):
        print(f"{r['dirname']}")
        print(f"  真实alpha={r['config_alpha']}, lr={r['config_lr']}, seed={r['seed']}")
        print(f"  test_mse={r['test_mse']:.4f}, test_ci={r['test_ci']:.4f}")
        print(f"  protocol: wd={r['config_weight_decay']}, beta={r['config_beta']}, aug={r['config_augmentation']}")

    # 保存JSON
    output_dir = Path("/home/rxp/dta2/results/analysis")
    output_dir.mkdir(exist_ok=True)

    json_out = output_dir / "p0_1_1_kiba_audit.json"
    with open(json_out, 'w') as f:
        json.dump({
            'timestamp': datetime.datetime.now().isoformat(),
            'total_dirs': len(all_dirs),
            'total_records': len(audit_records),
            'records': audit_records,
            'summary_by_model_batch': {
                f"{m}|{b}": {
                    'count': len(recs),
                    'alphas': sorted(set(r['config_alpha'] for r in recs if r['config_alpha'] is not None)),
                    'seeds': sorted(set(r['seed'] for r in recs if r['seed'] is not None)),
                    'n_seeds': len(set(r['seed'] for r in recs if r['seed'] is not None)),
                }
                for (m, b), recs in by_model_batch.items()
            }
        }, f, indent=2, default=str)
    print(f"\nJSON已保存: {json_out}")

    # 保存Markdown
    md_out = output_dir / "p0_1_1_kiba_audit.md"
    with open(md_out, 'w') as f:
        f.write("# KIBA实验全量审计报告\n\n")
        f.write(f"**时间**: {datetime.datetime.now().isoformat()}\n\n")
        f.write(f"**总目录数**: {len(all_dirs)}\n\n")
        f.write(f"**有效记录数**: {len(audit_records)}\n\n")

        f.write("## 按(model, batch)分组统计\n\n")
        f.write("| Model | Batch | 目录数 | n(种子) | 真实alpha值 |\n")
        f.write("|-------|-------|--------|---------|-------------|\n")
        for (m, b), recs in sorted(by_model_batch.items()):
            alphas = sorted(set(r['config_alpha'] for r in recs if r['config_alpha'] is not None))
            seeds = set(r['seed'] for r in recs if r['seed'] is not None)
            f.write(f"| {m} | {b} | {len(recs)} | {len(seeds)} | {alphas} |\n")

        f.write("\n## 完整审计表\n\n")
        f.write("| 目录名 | Model | 真实α | lr | seed | test_mse | test_ci | weight_decay | beta | augmentation |\n")
        f.write("|--------|-------|-------|----|------|----------|---------|--------------|------|--------------|\n")
        for r in sorted(audit_records, key=lambda x: x['dirname']):
            f.write(f"| {r['dirname']} | {r['model']} | {r['config_alpha']} | {r['config_lr']} | {r['seed']} | {r['test_mse']:.4f} | {r['test_ci']:.4f} | {r['config_weight_decay']} | {r['config_beta']} | {r['config_augmentation']} |\n")

        f.write("\n## 关键发现\n\n")

        # 识别n=10批次
        f.write("### n=10批次识别\n\n")
        for (m, b), recs in sorted(by_model_batch.items()):
            seeds = set(r['seed'] for r in recs if r['seed'] is not None)
            if len(seeds) >= 10:
                alphas = sorted(set(r['config_alpha'] for r in recs if r['config_alpha'] is not None))
                f.write(f"- **[{m}, {b}]**: n={len(seeds)}, alpha={alphas}\n")
                f.write(f"  - 种子列表: {sorted(seeds)}\n")

        # 识别alpha命名不一致
        f.write("\n### 目录名alpha vs 真实alpha不一致\n\n")
        mismatches = []
        for r in audit_records:
            if r['dir_alpha'] is not None and r['config_alpha'] is not None:
                # 猜测目录名的含义
                dir_val_raw = r['dir_alpha']
                config_val = r['config_alpha']
                # 常见的目录命名: alpha001=0.001, alpha010=0.01, alpha100=0.1
                # 但也可能alpha050=0.50(错误)或0.05
                if dir_val_raw == 1: dir_val = 0.001
                elif dir_val_raw == 3: dir_val = 0.003
                elif dir_val_raw == 5: dir_val = 0.005
                elif dir_val_raw == 10: dir_val = 0.01
                elif dir_val_raw == 30: dir_val = 0.03
                elif dir_val_raw == 50: dir_val = 0.05  # 关键:0.05还是0.50?
                elif dir_val_raw == 100: dir_val = 0.1
                elif dir_val_raw == 500: dir_val = 0.5
                else: dir_val = dir_val_raw / 1000.0

                if abs(dir_val - config_val) > 1e-9:
                    mismatches.append((r['dirname'], dir_val, config_val))

        if mismatches:
            for dn, dv, cv in mismatches:
                f.write(f"- ⚠️  `{dn}`: 目录名推断alpha={dv}, 真实alpha={cv}\n")
        else:
            f.write("未发现不一致。\n")

    print(f"Markdown已保存: {md_out}")

    print("\n" + "=" * 80)
    print("审计完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
