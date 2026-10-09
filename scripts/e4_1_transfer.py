#!/usr/bin/env python3
"""
E4.1: 跨数据集迁移实验 Davis↔KIBA 训练驱动脚本
================================================================================
时间: 2026-10-09
任务来源: 《实验复核意见与下一阶段实验方案.md》实验E4
目标:
  检验 InfoNCE 表征的跨数据集迁移是否存在不对称性:
  Davis 预训练表征 → KIBA 微调 (D2K)，与 KIBA 预训练表征 → Davis 微调 (K2D)。
  (方案背景更新: n=10重跑已证伪"符号反转"——其为跨lr基线伪影; 本实验的
   解读框架相应调整为: 检验"KIBA需要更大α"是否源于表征层面的数据集错位)
方法:
  - 预训练权重(均已存在, 协议β=0.0/γ=0.0/noise, 仅InfoNCE):
      D2K: results/controlled/E7_alpha005_seed{S}/best_model.pt (Davis GIN α=0.05)
      K2D: results/controlled/GIN_KIBA_alpha50_seed{S}/best_model.pt (KIBA GIN α=0.05)
  - 微调(方案E4): lr=1e-5, max_epochs=50, 纯监督(不传--use_contrastive),
    aug=none, bs=512, wd=1e-4, patience=50(不触发), joint
  - 通过 train_controlled.py 新增的 --pretrained_weights 参数注入权重
    (该参数2026-10-09添加, 默认None不影响既有行为)
  - 对照(已存在, 不重跑): KIBA_E0 (n=10, 0.5220±0.0414) 与 Davis E0 (n=5,
    0.2513±0.0130) —— 同协议从头训练基线, 重跑不增加信息
  - 种子: {42,3407,2024,7,123}, 与预训练权重种子一一对应(同种子迁移)
输出: results/controlled/e4_D2K_seed{S}/, results/controlled/e4_K2D_seed{S}/
      + results/analysis/e4_1_transfer_summary.json
================================================================================
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = '/home/rxp/dta2'
TRAIN_SCRIPT = f'{PROJECT_ROOT}/scripts/train_controlled.py'
RESULTS_DIR = Path(f'{PROJECT_ROOT}/results/controlled')
SUMMARY_FILE = Path(f'{PROJECT_ROOT}/results/analysis/e4_1_transfer_summary.json')
TIMEOUT_PER_RUN = 3600   # 50 epochs × ~15s + 加载 ≈ 15min, 留足余量

SEEDS = [42, 3407, 2024, 7, 123]

# (方向, experiment_id模板, 目标数据集, 预训练权重模板)
TRANSFERS = [
    ('D2K', 'e4_D2K_seed{}', 'kiba',
     PROJECT_ROOT + '/results/controlled/E7_alpha005_seed{}/best_model.pt'),
    ('K2D', 'e4_K2D_seed{}', 'davis',
     PROJECT_ROOT + '/results/controlled/GIN_KIBA_alpha50_seed{}/best_model.pt'),
]

FINETUNE = dict(learning_rate='0.00001',   # lr=1e-5 (方案E4)
                max_epochs='50',           # 50 epochs (方案E4)
                weight_decay='0.0001',
                batch_size='512',
                early_stop_patience='50',
                training_strategy='joint')


def build_command(exp_id, seed, dataset, pretrained):
    return [
        sys.executable, TRAIN_SCRIPT,
        '--experiment_id', exp_id,
        '--dataset', dataset,
        '--architecture', 'GIN',
        '--seed', str(seed),
        '--learning_rate', FINETUNE['learning_rate'],
        '--weight_decay', FINETUNE['weight_decay'],
        '--batch_size', FINETUNE['batch_size'],
        '--max_epochs', FINETUNE['max_epochs'],
        '--early_stop_patience', FINETUNE['early_stop_patience'],
        '--augmentation', 'none',
        '--training_strategy', FINETUNE['training_strategy'],
        '--pretrained_weights', pretrained,
    ]


def verify(exp_id, seed, dataset, pretrained):
    """完成后回读 results.json 做协议断言"""
    rj = RESULTS_DIR / exp_id / 'results.json'
    if not rj.exists():
        return False, 'results.json不存在'
    try:
        with open(rj) as f:
            r = json.load(f)
    except Exception as e:
        return False, f'解析失败: {e}'
    c = r.get('config', {})
    checks = [
        (c.get('seed') == seed, f"seed {c.get('seed')}"),
        (c.get('dataset') == dataset, f"dataset {c.get('dataset')}"),
        (abs(float(c.get('learning_rate', -1)) - 1e-5) < 1e-10,
         f"lr {c.get('learning_rate')}"),
        (int(c.get('max_epochs', -1)) == 50, f"epochs {c.get('max_epochs')}"),
        (str(c.get('pretrained_weights')) == pretrained,
         f"pretrained {c.get('pretrained_weights')}"),
        (bool(c.get('use_contrastive')) is False, f"uc {c.get('use_contrastive')}"),
        (str(c.get('augmentation')) == 'none', f"aug {c.get('augmentation')}"),
        (str(c.get('architecture')) == 'GIN', f"arch {c.get('architecture')}"),
    ]
    for k in ('test_mse', 'test_ci', 'best_epoch'):
        checks.append((k in r, f'缺字段{k}'))
    bad = [m for ok, m in checks if not ok]
    if bad:
        return False, '; '.join(bad)
    return True, f"OK mse={r['test_mse']:.4f} ci={r['test_ci']:.4f} ep={r['best_epoch']}"


def main():
    print('=' * 70)
    print('E4.1: 跨数据集迁移 Davis↔KIBA (10 runs)')
    print(f'开始: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 70)
    assert Path(TRAIN_SCRIPT).exists()
    import torch
    assert torch.cuda.is_available(), 'GPU不可用'
    print(f'GPU: {torch.cuda.get_device_name(0)} ✓')

    results = []
    for direction, tpl, dataset, wtpl in TRANSFERS:
        for seed in SEEDS:
            exp_id = tpl.format(seed)
            pretrained = wtpl.format(seed)
            print(f"\n{'─'*70}\n实验: {exp_id} ({direction}, 微调数据集={dataset})\n{'─'*70}")
            assert Path(pretrained).exists(), f'预训练权重缺失: {pretrained}'
            ok, msg = verify(exp_id, seed, dataset, pretrained)
            if ok:
                print(f'⚠️  已存在且合规，跳过: {msg}')
                results.append(dict(exp_id=exp_id, direction=direction,
                                    status='skipped', detail=msg))
                continue
            cmd = build_command(exp_id, seed, dataset, pretrained)
            print(f'命令: {" ".join(cmd)}')
            t0 = datetime.now()
            try:
                proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True,
                                      text=True, timeout=TIMEOUT_PER_RUN)
            except subprocess.TimeoutExpired:
                print('⏱️ 超时')
                results.append(dict(exp_id=exp_id, direction=direction,
                                    status='timeout'))
                continue
            el = (datetime.now() - t0).total_seconds()
            if proc.returncode != 0:
                print(f'❌ 失败: {proc.stderr[-500:]}')
                results.append(dict(exp_id=exp_id, direction=direction,
                                    status='failed', detail=proc.stderr[-500:]))
                continue
            ok, msg = verify(exp_id, seed, dataset, pretrained)
            print(f'{"✓" if ok else "❌协议断言"}: {msg} ({el/60:.1f}min)')
            results.append(dict(exp_id=exp_id, direction=direction,
                                status='success' if ok else 'protocol_violation',
                                detail=msg, elapsed_min=round(el / 60, 1)))

    summary = dict(script='scripts/e4_1_transfer.py',
                   timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                   finetune_protocol=FINETUNE, seeds=SEEDS, total=len(results),
                   success=sum(1 for r in results if r['status'] in ('success', 'skipped')),
                   failed=sum(1 for r in results if r['status'] not in ('success', 'skipped')),
                   results=results)
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_FILE, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n{'='*70}\n完成 {summary['success']}/{summary['total']}, "
          f"失败 {summary['failed']}\n汇总: {SUMMARY_FILE}\n{'='*70}")
    sys.exit(0 if summary['failed'] == 0 else 1)


if __name__ == '__main__':
    main()
