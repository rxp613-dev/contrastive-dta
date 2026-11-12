#!/usr/bin/env python3
"""
V9 Task T1.2: max_epochs提升至1000验证
日期: 2026-10-23
目标: 验证max_epochs=500是否已充分收敛，对比500 vs 1000的结果
方法: 基于T1.1脚本，将max_epochs从500提升至1000
配置:
  - batch_size: 512
  - lr_scheduler.patience: 30 (Config B)
  - early_stopping.patience: 150 (Config B)
  - max_epochs: 1000 (从500提升)
种子: 42, 7 (选择T1.1中表现最好和最差的各一个)
预计成本: 10-15 GPU小时
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, '/home/rxp/dta2')

from src.data.dta_dataset import get_all_loaders
from src.models.controlled_dta import GINEncoder, ProteinCNN, ControlledDTA
from scipy.stats import kendalltau


def set_seed(seed):
    """设置随机种子确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_ci(preds, targets):
    """计算Concordance Index"""
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    tau, _ = kendalltau(preds, targets)
    return (tau + 1) / 2


def evaluate(model, loader, criterion, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            protein_seq = batch.protein_seq.view(batch.num_graphs, -1)
            outputs = model(batch, protein_seq)
            loss = criterion(outputs.squeeze(), batch.y)
            total_loss += loss.item() * batch.num_graphs
            all_preds.append(outputs.squeeze())
            all_targets.append(batch.y)

    avg_loss = total_loss / len(loader.dataset)
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    mse = nn.functional.mse_loss(all_preds, all_targets).item()
    ci = compute_ci(all_preds, all_targets)

    return {'loss': avg_loss, 'mse': mse, 'ci': ci}


def train_e0_corrected(seed, config, output_dir):
    """训练修正配置的E0模型"""
    print("=" * 80)
    print(f"V9 Task E3.4: KIBA E0 修正配置实验")
    print(f"种子: {seed}")
    print(f"配置: lr_patience={config['lr_patience']}, early_stop_patience={config['early_stop_patience']}")
    print(f"设备: {config['device']}")
    print("=" * 80)
    
    set_seed(seed)
    
    # 加载KIBA数据
    print("\n加载KIBA数据...")
    loaders = get_all_loaders(
        dataset_name='kiba',
        batch_size=config['batch_size'],
        num_workers=4,
        split_type=None
    )
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']
    print(f"训练集: {len(train_loader.dataset)} 样本")
    print(f"验证集: {len(val_loader.dataset)} 样本")
    print(f"测试集: {len(test_loader.dataset)} 样本")
    
    # 创建模型
    print("\n创建模型...")
    model = ControlledDTA().to(config['device'])
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器和调度器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config['lr_patience'])
    
    # 训练循环
    best_val_mse = float('inf')
    best_epoch = 0
    patience_counter = 0
    training_history = []
    
    print(f"\n开始训练（修正配置）...")
    print(f"最大epochs: {config['max_epochs']}")
    print(f"早停patience: {config['early_stop_patience']}")
    print(f"LR调度patience: {config['lr_patience']}")
    print()
    
    for epoch in range(config['max_epochs']):
        model.train()
        train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['max_epochs']}", leave=False):
            batch = batch.to(config['device'])
            protein_seq = batch.protein_seq.view(batch.num_graphs, -1)
            
            optimizer.zero_grad()
            outputs = model(batch, protein_seq)
            loss = criterion(outputs.squeeze(), batch.y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
        
        train_loss /= len(train_loader.dataset)
        
        # 验证
        val_metrics = evaluate(model, val_loader, criterion, config['device'])
        val_mse = val_metrics['mse']
        
        # 学习率调度
        scheduler.step(val_mse)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录历史
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_mse': val_mse,
            'val_ci': val_metrics['ci'],
            'lr': current_lr
        })
        
        # 早停检查
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch + 1
            patience_counter = 0
            # 保存最佳模型
            torch.save(model.state_dict(), output_dir / f'best_model_seed{seed}.pt')
        else:
            patience_counter += 1
        
        # 打印进度（每10个epoch）
        if (epoch + 1) % 10 == 0 or patience_counter == config['early_stop_patience']:
            print(f"Epoch {epoch+1:3d}: train_loss={train_loss:.4f}, val_mse={val_mse:.4f}, val_ci={val_metrics['ci']:.4f}, lr={current_lr:.2e}, patience={patience_counter}/{config['early_stop_patience']}")
        
        # 早停
        if patience_counter >= config['early_stop_patience']:
            print(f"\n早停触发：验证集MSE在{config['early_stop_patience']}个epochs内未改善")
            print(f"最佳epoch: {best_epoch}, 最佳val_mse: {best_val_mse:.4f}")
            break
    
    # 测试评估
    print(f"\n加载最佳模型（epoch {best_epoch}）进行测试评估...")
    model.load_state_dict(torch.load(output_dir / f'best_model_seed{seed}.pt'))
    test_metrics = evaluate(model, test_loader, criterion, config['device'])
    
    print(f"\n测试结果:")
    print(f"  Test MSE: {test_metrics['mse']:.4f}")
    print(f"  Test CI:  {test_metrics['ci']:.4f}")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Total Epochs: {epoch + 1}")
    
    # 保存结果
    result = {
        'seed': seed,
        'config': config,
        'best_epoch': best_epoch,
        'total_epochs': epoch + 1,
        'best_val_mse': best_val_mse,
        'test_mse': test_metrics['mse'],
        'test_ci': test_metrics['ci'],
        'training_history': training_history
    }
    
    with open(output_dir / f'results_seed{seed}.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\n结果已保存到: {output_dir}")
    return result


def main():
    """主函数"""
    # 修正配置（Config B：适度调整）
    config = {
        'batch_size': 512,
        'lr': 5e-4,
        'weight_decay': 1e-4,
        'lr_patience': 30,  # 从10增加到30
        'early_stop_patience': 150,  # 从50增加到150
        'max_epochs': 1000,  # T1.2: 从500提升到1000验证收敛性
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    # 选择2个种子进行验证（T1.1中表现最好和最差的各一个）
    seeds = [42, 7]  # seed 42: best_epoch=491, seed 7: best_epoch=500
    
    # 输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'/mnt/data/dta2/results/controlled/v9_t1p2_max_epochs_1000_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"V9 Task T1.2: KIBA E0 max_epochs=1000验证实验")
    print(f"输出目录: {output_dir}")
    print(f"配置: {json.dumps(config, indent=2)}")
    print(f"种子: {seeds}")
    print()
    
    # 保存配置
    with open(output_dir / 'config.json', 'w') as f:
        json.dump({'config': config, 'seeds': seeds}, f, indent=2)
    
    # 运行实验
    all_results = []
    for seed in seeds:
        result = train_e0_corrected(seed, config, output_dir)
        all_results.append(result)
        print("\n" + "=" * 80 + "\n")
    
    # 汇总结果
    test_mses = [r['test_mse'] for r in all_results]
    test_cis = [r['test_ci'] for r in all_results]
    best_epochs = [r['best_epoch'] for r in all_results]
    total_epochs_list = [r['total_epochs'] for r in all_results]

    summary = {
        'timestamp': timestamp,
        'task': 'V9 T1.1: KIBA E0 新协议10种子验证',
        'config': config,
        'seeds': seeds,
        'results': all_results,
        'statistics': {
            'test_mse_mean': float(np.mean(test_mses)),
            'test_mse_std': float(np.std(test_mses)),
            'test_ci_mean': float(np.mean(test_cis)),
            'test_ci_std': float(np.std(test_cis)),
            'best_epoch_mean': float(np.mean(best_epochs)),
            'best_epoch_std': float(np.std(best_epochs)),
            'total_epochs_mean': float(np.mean(total_epochs_list)),
            'total_epochs_std': float(np.std(total_epochs_list))
        },
        'literature_comparison': {
            'graphdta_gin_mse': 0.147,
            'graphdta_gin_ci': 0.882,
            'our_mse_mean': float(np.mean(test_mses)),
            'our_ci_mean': float(np.mean(test_cis)),
            'mse_difference_pct': float((np.mean(test_mses) - 0.147) / 0.147 * 100),
            'ci_difference': float(np.mean(test_cis) - 0.882)
        },
        'comparison': {
            'original_e0_mean': 0.4847,  # 来自E2诊断
            'original_e0_std': 0.1369,
            'corrected_mean': float(np.mean(test_mses)),
            'corrected_std': float(np.std(test_mses)),
            'improvement': 0.4847 - float(np.mean(test_mses))
        }
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("=" * 80)
    print("V9 Task T1.1 完成")
    print("=" * 80)
    print(f"\n测试MSE: {np.mean(test_mses):.4f} ± {np.std(test_mses):.4f}")
    print(f"测试CI:  {np.mean(test_cis):.4f} ± {np.std(test_cis):.4f}")
    print(f"最佳epoch: {np.mean(best_epochs):.1f} ± {np.std(best_epochs):.1f}")
    print(f"总epochs: {np.mean(total_epochs_list):.1f} ± {np.std(total_epochs_list):.1f}")

    print(f"\n文献对比 (GraphDTA-GIN):")
    print(f"  文献MSE: 0.147")
    print(f"  本文MSE: {np.mean(test_mses):.4f}")
    print(f"  差异: {summary['literature_comparison']['mse_difference_pct']:.1f}%")
    print(f"\n  文献CI: 0.882")
    print(f"  本文CI: {np.mean(test_cis):.4f}")
    print(f"  差异: {summary['literature_comparison']['ci_difference']:.4f}")

    print(f"\n原始E0 MSE: {summary['comparison']['original_e0_mean']:.4f} ± {summary['comparison']['original_e0_std']:.4f}")
    print(f"修正E0 MSE: {summary['comparison']['corrected_mean']:.4f} ± {summary['comparison']['corrected_std']:.4f}")
    print(f"改进: {summary['comparison']['improvement']:.4f}")
    print(f"\n结果已保存到: {output_dir}")


if __name__ == '__main__':
    main()
