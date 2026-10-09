#!/usr/bin/env python3
"""
E1.1: KIBA n=10 扩展实验——缺失实验补跑脚本
================================================================================
时间: 2026-10-08
任务来源: 《实验复核意见与下一阶段实验方案.md》实验E1 (R1/R2/R3)
目标:
  补齐 KIBA n=10 权威重跑中缺失的 8 个实验，使 α 扫描(β=0.0)与 E0 基线
  均达到 n=10（原5种子 {42,3407,2024,7,123} + 新5种子 {251,1496,5934,7350,9889}）。

缺失实验清单（由 2026-10-08 全量目录审计确定，审计依据为 results.json 的
config 字段而非目录名——Round 22 教训）:
  处理组(对比学习, --use_contrastive, β=0.0, γ=0.0):
    1. GIN_KIBA_alpha30_seed251   (α=0.03)
    2. GIN_KIBA_alpha50_seed251   (α=0.05)
    3. GIN_KIBA_alpha50_seed7350  (α=0.05)
  基线组(E0 纯监督, 不加 --use_contrastive, augmentation=none):
    4-8. KIBA_E0_seed{251,1496,5934,7350,9889}

协议（与权威批次 GIN_KIBA_alpha*/KIBA_E0 的 config 实证完全一致）:
  dataset=kiba, architecture=GIN(参数量498,177), lr=5e-4, weight_decay=1e-4,
  batch_size=512, max_epochs=500, early_stop_patience=50, temperature=0.1,
  training_strategy=joint, mask_ratio=0.2, seed=见上
  处理组: augmentation=noise, alpha=见上, beta=0.0, gamma=0.0
  基线组: augmentation=none, 不使用对比学习

注意(偏差说明):
  方案文档硬约束写 lr=1e-4，但权威 n=5 批次(GIN_KIBA_alpha*, KIBA_E0)的
  results.json config 实证为 lr=5e-4；lr=1e-4 的 KIBA_alpha*_lr0.0001 批次
  全部 β=0.05 污染已于 Round 22 被排除。为与既有 32 个有效实验(27新+5旧可合并)
  保持协议一致，本脚本使用 lr=5e-4。详见 TODO_2026-10-08.md 阶段0偏差说明。

方法:
  - 通过 subprocess 调用 scripts/train_controlled.py（与 p2_1 相同机制）
  - 每个实验完成后立即回读 results.json 做协议断言（β/γ/lr/aug/seed/α），
    断言失败立即标记并继续，最后汇总
  - 已存在且 config 合规的实验自动跳过（幂等，可重复执行）
  - GPU 训练（train_controlled.py 内部自动检测 CUDA）

预期产出:
  - results/controlled/GIN_KIBA_alpha30_seed251/
  - results/controlled/GIN_KIBA_alpha50_seed{251,7350}/
  - results/controlled/KIBA_E0_seed{251,1496,5934,7350,9889}/
  - results/analysis/e1_1_kiba_n10_completion_summary.json
================================================================================
"""

import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# 协议常量（与权威批次实证一致，写死，不接受外部覆盖）
# ---------------------------------------------------------------------------
PROJECT_ROOT = '/home/rxp/dta2'
TRAIN_SCRIPT = f'{PROJECT_ROOT}/scripts/train_controlled.py'
RESULTS_DIR = Path(f'{PROJECT_ROOT}/results/controlled')
SUMMARY_FILE = Path(f'{PROJECT_ROOT}/results/analysis/e1_1_kiba_n10_completion_summary.json')
TIMEOUT_PER_RUN = 10800  # 单实验3小时超时(500ep×~14s+数据加载+评估≈2h边际,留余量)
                         # 2026-10-09: alpha50_seed251首轮2h超时(未触发早停的全500轮训练)

PROTOCOL = dict(
    dataset='kiba',
    architecture='GIN',
    learning_rate='0.0005',
    weight_decay='0.0001',
    batch_size='512',
    max_epochs='500',
    early_stop_patience='50',
    temperature='0.1',
    training_strategy='joint',
    pretrain_epochs='200',
    finetune_epochs='300',
)

NEW_SEEDS = [251, 1496, 5934, 7350, 9889]

# 缺失实验清单: (experiment_id, alpha, use_contrastive, augmentation)
# alpha=None 表示 E0 基线（不启用对比学习，alpha 参数无效）
MISSING_RUNS = [
    ('GIN_KIBA_alpha30_seed251',   0.03, True,  'noise'),
    ('GIN_KIBA_alpha50_seed251',   0.05, True,  'noise'),
    ('GIN_KIBA_alpha50_seed7350',  0.05, True,  'noise'),
    ('KIBA_E0_seed251',            None, False, 'none'),
    ('KIBA_E0_seed1496',           None, False, 'none'),
    ('KIBA_E0_seed5934',           None, False, 'none'),
    ('KIBA_E0_seed7350',           None, False, 'none'),
    ('KIBA_E0_seed9889',           None, False, 'none'),
]


def build_command(exp_id, alpha, use_contrastive, augmentation, seed):
    """构建 train_controlled.py 调用命令（协议参数全部显式传参）"""
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        '--experiment_id', exp_id,
        '--dataset', PROTOCOL['dataset'],
        '--architecture', PROTOCOL['architecture'],
        '--seed', str(seed),
        '--learning_rate', PROTOCOL['learning_rate'],
        '--weight_decay', PROTOCOL['weight_decay'],
        '--batch_size', PROTOCOL['batch_size'],
        '--max_epochs', PROTOCOL['max_epochs'],
        '--early_stop_patience', PROTOCOL['early_stop_patience'],
        '--temperature', PROTOCOL['temperature'],
        '--augmentation', augmentation,
        '--mask_ratio', '0.2',
        '--training_strategy', PROTOCOL['training_strategy'],
        '--pretrain_epochs', PROTOCOL['pretrain_epochs'],
        '--finetune_epochs', PROTOCOL['finetune_epochs'],
    ]
    if use_contrastive:
        cmd += ['--use_contrastive',
                '--alpha', str(alpha),
                '--beta', '0.0',
                '--gamma', '0.0']
    return cmd


def verify_results_json(exp_id, alpha, use_contrastive, augmentation, seed):
    """实验完成后回读 results.json 做协议断言（以config为准，不信目录名）。

    返回 (ok: bool, message: str)
    """
    rj = RESULTS_DIR / exp_id / 'results.json'
    if not rj.exists():
        return False, f'results.json 不存在: {rj}'
    try:
        with open(rj) as f:
            r = json.load(f)
    except Exception as e:
        return False, f'results.json 解析失败: {e}'
    c = r.get('config', {})

    # 通用协议断言（覆盖全部关键超参，防默认值静默漂移）
    checks = [
        (c.get('seed') == seed, f"seed: 期望{seed} 实际{c.get('seed')}"),
        (c.get('dataset') == 'kiba', f"dataset: {c.get('dataset')}"),
        (abs(float(c.get('learning_rate', -1)) - 5e-4) < 1e-9,
         f"lr: 期望5e-4 实际{c.get('learning_rate')}"),
        (str(c.get('augmentation')) == augmentation,
         f"augmentation: 期望{augmentation} 实际{c.get('augmentation')}"),
        (int(c.get('batch_size', -1)) == 512, f"batch_size: {c.get('batch_size')}"),
        (bool(c.get('use_contrastive')) == use_contrastive,
         f"use_contrastive: 期望{use_contrastive} 实际{c.get('use_contrastive')}"),
        (abs(float(c.get('weight_decay', -1)) - 1e-4) < 1e-9,
         f"weight_decay: 期望1e-4 实际{c.get('weight_decay')}"),
        (int(c.get('max_epochs', -1)) == 500, f"max_epochs: {c.get('max_epochs')}"),
        (int(c.get('early_stop_patience', -1)) == 50,
         f"early_stop_patience: {c.get('early_stop_patience')}"),
        (abs(float(c.get('temperature', -1)) - 0.1) < 1e-9,
         f"temperature: 期望0.1 实际{c.get('temperature')}"),
        (abs(float(c.get('mask_ratio', -1)) - 0.2) < 1e-9,
         f"mask_ratio: 期望0.2 实际{c.get('mask_ratio')}"),
        (str(c.get('training_strategy')) == 'joint',
         f"training_strategy: {c.get('training_strategy')}"),
        (c.get('split_type') in (None, 'none', 'None'),
         f"split_type: {c.get('split_type')}"),
    ]
    if use_contrastive:
        checks += [
            (abs(float(c.get('alpha', -1)) - alpha) < 1e-9,
             f"alpha: 期望{alpha} 实际{c.get('alpha')}"),
            (float(c.get('beta', -1)) == 0.0, f"beta: 期望0.0 实际{c.get('beta')}"),
            (float(c.get('gamma', -1)) == 0.0, f"gamma: 期望0.0 实际{c.get('gamma')}"),
        ]
    # 关键结果字段存在性
    for k in ('test_mse', 'test_ci', 'best_epoch'):
        checks.append((k in r, f'缺少结果字段: {k}'))

    failed = [msg for ok, msg in checks if not ok]
    if failed:
        return False, '; '.join(failed)
    return True, f"OK (test_mse={r['test_mse']:.4f}, test_ci={r['test_ci']:.4f}, best_epoch={r['best_epoch']})"


def already_done(exp_id, alpha, use_contrastive, augmentation, seed):
    """幂等检查：目录存在且results.json通过协议断言则跳过"""
    if (RESULTS_DIR / exp_id / 'results.json').exists():
        ok, msg = verify_results_json(exp_id, alpha, use_contrastive, augmentation, seed)
        return ok, msg
    return False, ''


def main():
    print('=' * 70)
    print('E1.1: KIBA n=10 缺失实验补跑')
    print(f'开始时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'计划实验数: {len(MISSING_RUNS)}')
    for exp_id, alpha, uc, aug in MISSING_RUNS:
        print(f'  - {exp_id} (α={alpha}, contrastive={uc}, aug={aug})')
    print('=' * 70)

    # 前置检查: 训练脚本与GPU
    assert Path(TRAIN_SCRIPT).exists(), f'训练脚本不存在: {TRAIN_SCRIPT}'
    import torch
    assert torch.cuda.is_available(), 'GPU不可用，拒绝运行(原则5: 尽量GPU运行)'
    print(f'GPU: {torch.cuda.get_device_name(0)} ✓\n')

    results = []
    for exp_id, alpha, uc, aug in MISSING_RUNS:
        seed = int(exp_id.split('seed')[1])
        print(f"\n{'─' * 70}\n实验: {exp_id}\n{'─' * 70}")

        skip, msg = already_done(exp_id, alpha, uc, aug, seed)
        if skip:
            print(f'⚠️  已存在且协议合规，跳过: {msg}')
            results.append(dict(exp_id=exp_id, status='skipped', detail=msg))
            continue

        cmd = build_command(exp_id, alpha, uc, aug, seed)
        print(f'命令: {" ".join(cmd)}')
        t0 = datetime.now()
        try:
            proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True,
                                  text=True, timeout=TIMEOUT_PER_RUN)
        except subprocess.TimeoutExpired:
            print('⏱️  超时(2h)')
            results.append(dict(exp_id=exp_id, status='timeout'))
            continue
        elapsed = (datetime.now() - t0).total_seconds()

        if proc.returncode != 0:
            print(f'❌ 训练失败 (rc={proc.returncode})')
            print(f'STDERR尾部: {proc.stderr[-800:]}')
            results.append(dict(exp_id=exp_id, status='failed',
                                detail=proc.stderr[-500:]))
            continue

        # 完成后协议断言
        ok, msg = verify_results_json(exp_id, alpha, uc, aug, seed)
        status = 'success' if ok else 'protocol_violation'
        print(f'{"✓" if ok else "❌ 协议断言失败"}: {msg}  (耗时 {elapsed/60:.1f} min)')
        results.append(dict(exp_id=exp_id, status=status, detail=msg,
                            elapsed_min=round(elapsed / 60, 1)))

    # 汇总
    summary = dict(
        script='scripts/e1_1_kiba_n10_completion.py',
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        protocol=PROTOCOL,
        new_seeds=NEW_SEEDS,
        total=len(MISSING_RUNS),
        success=sum(1 for r in results if r['status'] in ('success', 'skipped')),
        failed=sum(1 for r in results if r['status'] not in ('success', 'skipped')),
        results=results,
    )
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_FILE, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f'完成: {summary["success"]}/{summary["total"]}  失败: {summary["failed"]}')
    print(f'汇总: {SUMMARY_FILE}')
    print(f'结束时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 70)

    # 任一失败则以非零码退出，提醒人工核查
    sys.exit(0 if summary['failed'] == 0 else 1)


if __name__ == '__main__':
    main()
