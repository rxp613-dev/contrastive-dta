#!/usr/bin/env python3
"""
V9 Task T1.1: E0补至10种子（KIBA，新协议）
日期: 2026-10-22
目标: 使用Config B（lr_patience=30, early_stop=150）运行10个种子
方法:
  1. 设计实验：Config B + 10种子
  2. 代码审核：确保与E3.4脚本一致
  3. 运行10种子实验（预计20-30 GPU小时）
  4. 计算均值、标准差、CI
  5. 对比文献GIN（MSE=0.147, CI=0.882）
  6. 生成结果报告
"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.gin_model import GINModel
from data.dta_dataset import DTADataset
from utils.train_utils import train_epoch, validate, compute_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 7, 123, 456, 789, 101, 202, 303, 404, 505])
    parser.add_argument('--output_dir', type=str, default='results/controlled/v9_t1p1_e0_10seeds')
    args = parser.parse_args()

    # 配置
    config = {
        'dataset': 'kiba',
        'batch_size': 512,
        'lr': 5e-4,
        'weight_decay': 1e-4,
        'lr_patience': 30,
        'lr_factor': 0.5,
        'early_stop_patience': 150,
        'max_epochs': 500,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'num_workers': 4,
        'use_contrastive': False,
        'alpha': 0.0,
        'beta': 0.0,
        'gamma': 0.0,
        'temperature': 0.1,
        'augmentation': 'none',
        'mask_ratio': 0.2,
        'training_strategy': 'joint'
    }

    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'v9_t1p1_e0_10seeds_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存配置
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print("="*80)
    print("V9 Task T1.1: E0补至10种子（KIBA，新协议）")
    print("="*80)
    print(f"\n配置: {json.dumps(config, indent=2)}")
    print(f"种子: {args.seeds}")
    print(f"输出目录: {output_dir}")

    # 加载数据集
    print("\n加载KIBA数据集...")
    train_dataset = DTADataset(
        dataset_name='kiba',
        split='train',
        augmentation=config['augmentation'],
        mask_ratio=config['mask_ratio']
    )
    val_dataset = DTADataset(
        dataset_name='kiba',
        split='val',
        augmentation='none',
        mask_ratio=0.0
    )
    test_dataset = DTADataset(
        dataset_name='kiba',
        split='test',
        augmentation='none',
        mask_ratio=0.0
    )

    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")

    # 运行所有种子
    all_results = []

    for seed_idx, seed in enumerate(args.seeds):
        print(f"\n{'='*80}")
        print(f"种子 {seed_idx+1}/{len(args.seeds)}: Seed {seed}")
        print(f"{'='*80}")

        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # 创建数据加载器
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )

        # 创建模型
        model = GINModel(
            node_dim=train_dataset.node_dim,
            edge_dim=train_dataset.edge_dim,
            protein_dim=train_dataset.protein_dim,
            hidden_dim=128,
            num_layers=3,
            dropout=0.2
        ).to(config['device'])

        # 创建优化器
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config['weight_decay']
        )

        # 创建学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['lr_factor'],
            patience=config['lr_patience'],
            verbose=True
        )

        # 训练循环
        best_val_mse = float('inf')
        best_epoch = 0
        patience_counter = 0
        training_history = []

        for epoch in range(1, config['max_epochs'] + 1):
            # 训练
            train_loss = train_epoch(
                model, train_loader, optimizer, config['device'],
                use_contrastive=config['use_contrastive'],
                alpha=config['alpha'],
                beta=config['beta'],
                gamma=config['gamma'],
                temperature=config['temperature']
            )

            # 验证
            val_metrics = validate(model, val_loader, config['device'])
            val_mse = val_metrics['mse']
            val_ci = val_metrics['ci']

            # 记录历史
            training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_mse': val_mse,
                'val_ci': val_ci,
                'lr': optimizer.param_groups[0]['lr']
            })

            # 学习率调度
            scheduler.step(val_mse)

            # 早停检查
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_epoch = epoch
                patience_counter = 0
                # 保存最佳模型
                torch.save(model.state_dict(), output_dir / f'best_model_seed{seed}.pt')
            else:
                patience_counter += 1

            # 打印进度
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_mse={val_mse:.4f}, val_ci={val_ci:.4f}, lr={optimizer.param_groups[0]['lr']:.2e}")

            # 早停
            if patience_counter >= config['early_stop_patience']:
                print(f"早停: epoch {epoch}, best_epoch={best_epoch}, best_val_mse={best_val_mse:.4f}")
                break

        # 测试评估
        print(f"\n加载最佳模型 (epoch {best_epoch})...")
        model.load_state_dict(torch.load(output_dir / f'best_model_seed{seed}.pt'))
        test_metrics = validate(model, test_loader, config['device'])

        print(f"测试结果: MSE={test_metrics['mse']:.4f}, CI={test_metrics['ci']:.4f}")

        # 保存结果
        result = {
            'seed': seed,
            'config': config,
            'best_epoch': best_epoch,
            'total_epochs': len(training_history),
            'best_val_mse': best_val_mse,
            'test_mse': test_metrics['mse'],
            'test_ci': test_metrics['ci'],
            'training_history': training_history
        }

        all_results.append(result)

        # 保存单个种子结果
        with open(output_dir / f'results_seed{seed}.json', 'w') as f:
            json.dump(result, f, indent=2)

    # 汇总统计
    print(f"\n{'='*80}")
    print("汇总统计")
    print(f"{'='*80}")

    test_mses = [r['test_mse'] for r in all_results]
    test_cis = [r['test_ci'] for r in all_results]
    best_epochs = [r['best_epoch'] for r in all_results]
    total_epochs_list = [r['total_epochs'] for r in all_results]

    summary = {
        'timestamp': timestamp,
        'task': 'V9 T1.1: E0补至10种子（KIBA，新协议）',
        'config': config,
        'seeds': args.seeds,
        'results': all_results,
        'statistics': {
            'test_mse_mean': np.mean(test_mses),
            'test_mse_std': np.std(test_mses),
            'test_ci_mean': np.mean(test_cis),
            'test_ci_std': np.std(test_cis),
            'best_epoch_mean': np.mean(best_epochs),
            'best_epoch_std': np.std(best_epochs),
            'total_epochs_mean': np.mean(total_epochs_list),
            'total_epochs_std': np.std(total_epochs_list)
        },
        'literature_comparison': {
            'graphdta_gin_mse': 0.147,
            'graphdta_gin_ci': 0.882,
            'our_mse_mean': np.mean(test_mses),
            'our_ci_mean': np.mean(test_cis),
            'mse_difference_pct': (np.mean(test_mses) - 0.147) / 0.147 * 100,
            'ci_difference': np.mean(test_cis) - 0.882
        }
    }

    # 保存汇总
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # 打印汇总
    print(f"\n测试MSE: {np.mean(test_mses):.4f} ± {np.std(test_mses):.4f}")
    print(f"测试CI: {np.mean(test_cis):.4f} ± {np.std(test_cis):.4f}")
    print(f"最佳epoch: {np.mean(best_epochs):.1f} ± {np.std(best_epochs):.1f}")
    print(f"总epochs: {np.mean(total_epochs_list):.1f} ± {np.std(total_epochs_list):.1f}")

    print(f"\n文献对比 (GraphDTA-GIN):")
    print(f"  文献MSE: 0.147")
    print(f"  本文MSE: {np.mean(test_mses):.4f}")
    print(f"  差异: {summary['literature_comparison']['mse_difference_pct']:.1f}%")
    print(f"\n  文献CI: 0.882")
    print(f"  本文CI: {np.mean(test_cis):.4f}")
    print(f"  差异: {summary['literature_comparison']['ci_difference']:.4f}")

    print(f"\n{'='*80}")
    print(f"Task T1.1 完成")
    print(f"结果保存在: {output_dir}")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
