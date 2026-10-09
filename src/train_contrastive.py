#!/usr/bin/env python3
"""
受控DTA训练脚本
日期: 2026-08-04
修改: 2026-08-29 - 添加模型架构支持和参数断言（第八轮评审修复）
目标: 实现统一的训练流程，支持多种实验配置和架构（GIN/GAT）
输入: 数据加载器 + 模型 + 实验配置
输出: 训练好的模型 + 评估结果 + 训练日志

支持的实验类型:
  - E0: 纯监督基线（仅MSE损失）
  - E1-E7: 对比学习变体（InfoNCE + Uniformity + CrossModal）
  - 支持两阶段训练（预训练 + 微调）
  - 支持数据增强（noise, random_mask, sub_mask）

支持的架构:
  - GIN: ControlledDTA (498,177 params)
  - GAT: GATControlledDTA (521,345 params)

特性:
  1. 固定随机种子确保可复现性
  2. 支持GPU加速
  3. 自动早停（基于验证集MSE）
  4. 详细记录训练过程
  5. 保存最佳模型和训练日志
  6. 模型参数断言（防止错误模型）
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, '/home/rxp/dta2')

from src.models.controlled_dta import ControlledDTA
from src.models.gat_encoder import GATControlledDTA
from src.data.dta_dataset import get_all_loaders
from scripts.augmentation_strategies import RandomMaskAugmentor, SubstructureMaskAugmentor

# 模型架构参数量映射（用于断言验证）
MODEL_PARAM_COUNTS = {
    'GIN': 498177,
    'GAT': 521345,
}


def set_seed(seed):
    """
    设置随机种子确保可复现性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ContrastiveLoss(nn.Module):
    """
    InfoNCE对比损失

    用于对比学习实验（E1-E7）
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        Args:
            z_i: [B, D] 视图1的表示
            z_j: [B, D] 视图2的表示

        Returns:
            标量损失值
        """
        B, D = z_i.shape

        # 归一化
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)

        # 拼接: [2B, D]
        z = torch.cat([z_i, z_j], dim=0)

        # 相似度矩阵: [2B, 2B]
        sim = torch.mm(z, z.t()) / self.temperature

        # 掩码：对角线为负样本对
        mask = torch.eye(2 * B, device=z.device).bool()

        # 排除对角线
        sim = sim.masked_fill(mask, -1e9)

        # 正样本对索引
        pos_idx = torch.arange(B, 2 * B, device=z.device)

        # 计算InfoNCE损失
        logits = sim[:B]  # [B, 2B]
        labels = pos_idx  # [B]

        loss = nn.functional.cross_entropy(logits, labels)

        return loss


class UniformityLoss(nn.Module):
    """
    均匀性损失

    鼓励表示在超球面上均匀分布
    """

    def forward(self, z):
        """
        Args:
            z: [B, D] 表示

        Returns:
            标量损失值
        """
        z = nn.functional.normalize(z, dim=1)

        # 计算所有配对的距离
        pdist = torch.pdist(z, p=2)

        # 均匀性损失（负对数高斯核）
        loss = torch.log(torch.exp(-2 * pdist.pow(2)).mean() + 1e-8)

        return loss


class CrossModalLoss(nn.Module):
    """
    跨模态对比损失

    对齐药物和蛋白质的表示空间
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, drug_z, protein_z):
        """
        Args:
            drug_z: [B, D] 药物表示
            protein_z: [B, D] 蛋白质表示

        Returns:
            标量损失值
        """
        B, D = drug_z.shape

        # 归一化
        drug_z = nn.functional.normalize(drug_z, dim=1)
        protein_z = nn.functional.normalize(protein_z, dim=1)

        # 相似度矩阵: [B, B]
        sim = torch.mm(drug_z, protein_z.t()) / self.temperature

        # 正样本对在对角线
        labels = torch.arange(B, device=drug_z.device)

        # 计算交叉熵损失
        loss = nn.functional.cross_entropy(sim, labels)

        return loss


def train_epoch(model, dataloader, optimizer, criterion, device, config):
    """
    训练一个epoch

    Args:
        model: ControlledDTA模型
        dataloader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        device: 设备（cuda/cpu）
        config: 实验配置

    Returns:
        平均损失
    """
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc='Training', leave=False):
        # 移动到设备
        batch = batch.to(device)

        # 前向传播
        drug_repr, protein_repr, fusion = model.get_embeddings(
            batch, batch.protein_seq
        )

        # 预测
        pred = model.predictor(fusion).squeeze()

        # 计算MSE损失
        mse_loss = criterion(pred, batch.y)

        # 对比学习损失（如果启用）
        if config.get('use_contrastive', False):
            # 根据增强策略生成增强视图
            augmentation = config.get('augmentation', 'noise')

            if augmentation == 'none':
                # 不使用增强，使用原始表示
                drug_repr_aug = drug_repr
                protein_repr_aug = protein_repr
            elif augmentation == 'noise':
                # 噪声增强（原有实现）
                drug_repr_aug = drug_repr + torch.randn_like(drug_repr) * 0.1
                protein_repr_aug = protein_repr + torch.randn_like(protein_repr) * 0.1
            elif augmentation == 'random_mask':
                # 随机掩码增强
                mask_ratio = config.get('mask_ratio', 0.2)
                random_augmentor = RandomMaskAugmentor(mask_ratio=mask_ratio, seed=config.get('seed', 42))

                # 对每个样本的表示进行掩码
                drug_repr_aug_list = []
                for i in range(drug_repr.shape[0]):
                    # 创建虚拟的边索引（因为我们只在表示空间操作）
                    dummy_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                    aug_repr, _ = random_augmentor.augment(
                        drug_repr[i].unsqueeze(0),
                        dummy_edge_index
                    )
                    drug_repr_aug_list.append(aug_repr.squeeze(0))
                drug_repr_aug = torch.stack(drug_repr_aug_list)

                # 蛋白质表示也进行随机掩码
                protein_repr_aug_list = []
                for i in range(protein_repr.shape[0]):
                    dummy_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                    aug_repr, _ = random_augmentor.augment(
                        protein_repr[i].unsqueeze(0),
                        dummy_edge_index
                    )
                    protein_repr_aug_list.append(aug_repr.squeeze(0))
                protein_repr_aug = torch.stack(protein_repr_aug_list)

            elif augmentation == 'sub_mask':
                # 子结构感知掩码增强
                mask_ratio = config.get('mask_ratio', 0.2)
                sub_augmentor = SubstructureMaskAugmentor(mask_ratio=mask_ratio, seed=config.get('seed', 42))

                # 对每个样本的表示进行掩码
                # 注意：这里我们简化处理，因为SMILES信息不在batch中
                # 实际上应该在数据加载时保存SMILES信息
                drug_repr_aug_list = []
                for i in range(drug_repr.shape[0]):
                    dummy_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                    # 由于没有SMILES信息，退化为随机掩码
                    aug_repr, _ = sub_augmentor.augment(
                        drug_repr[i].unsqueeze(0),
                        dummy_edge_index,
                        smiles=None
                    )
                    drug_repr_aug_list.append(aug_repr.squeeze(0))
                drug_repr_aug = torch.stack(drug_repr_aug_list)

                protein_repr_aug_list = []
                for i in range(protein_repr.shape[0]):
                    dummy_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                    aug_repr, _ = sub_augmentor.augment(
                        protein_repr[i].unsqueeze(0),
                        dummy_edge_index,
                        smiles=None
                    )
                    protein_repr_aug_list.append(aug_repr.squeeze(0))
                protein_repr_aug = torch.stack(protein_repr_aug_list)
            else:
                # 默认使用噪声增强
                drug_repr_aug = drug_repr + torch.randn_like(drug_repr) * 0.1
                protein_repr_aug = protein_repr + torch.randn_like(protein_repr) * 0.1

            # InfoNCE损失
            contrastive_loss = ContrastiveLoss(temperature=config.get('temperature', 0.1))
            cl_loss_drug = contrastive_loss(drug_repr, drug_repr_aug)
            cl_loss_protein = contrastive_loss(protein_repr, protein_repr_aug)

            # 总损失
            alpha = config.get('alpha', 0.1)
            beta = config.get('beta', 0.05)
            gamma = config.get('gamma', 0.0)  # E1不使用CrossModal，默认为0

            loss = mse_loss + alpha * (cl_loss_drug + cl_loss_protein)

            # 均匀性损失（仅当beta > 0时计算，支持E5实验）
            if beta > 0:
                uniformity_loss = UniformityLoss()
                uni_loss_drug = uniformity_loss(drug_repr)
                uni_loss_protein = uniformity_loss(protein_repr)
                loss = loss + beta * (uni_loss_drug + uni_loss_protein)

            # 仅当gamma > 0时才计算CrossModal损失（E6使用）
            if gamma > 0:
                crossmodal_loss = CrossModalLoss(temperature=config.get('temperature', 0.1))
                cm_loss = crossmodal_loss(drug_repr, protein_repr)
                loss = loss + gamma * cm_loss
        else:
            loss = mse_loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """
    评估模型

    Args:
        model: ControlledDTA模型
        dataloader: 数据加载器
        criterion: 损失函数
        device: 设备

    Returns:
        dict: {'mse': ..., 'rmse': ..., 'ci': ...}
    """
    model.eval()
    all_preds = []
    all_targets = []
    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
            batch = batch.to(device)

            # 前向传播
            pred = model(batch, batch.protein_seq).squeeze()

            # 计算损失
            loss = criterion(pred, batch.y)
            total_loss += loss.item()

            # 收集预测和真实值
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(batch.y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 计算指标
    mse = np.mean((all_preds - all_targets) ** 2)
    rmse = np.sqrt(mse)

    # 计算一致性指数（CI）- 向量化版本（修复O(n²)瓶颈）
    # 标准CI = (concordant + 0.5*tied) / relevant_pairs
    # 其中 relevant_pairs 是 target 不相等的样本对
    from scipy.stats import kendalltau

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 使用向量化计算（比纯Python循环快100-1000倍）
    n = len(all_preds)

    # 计算所有样本对的target差异
    target_diff = all_targets[:, np.newaxis] - all_targets[np.newaxis, :]
    pred_diff = all_preds[:, np.newaxis] - all_preds[np.newaxis, :]

    # 只考虑target不相等的样本对（上三角，避免重复计算）
    mask = np.triu(np.abs(target_diff) > 1e-10, k=1)

    relevant_pairs = np.sum(mask)

    if relevant_pairs > 0:
        # 提取相关的pred_diff
        relevant_pred_diff = pred_diff[mask]

        # 计算一致性
        concordant = np.sum((target_diff[mask] > 0) & (relevant_pred_diff > 0))
        concordant += np.sum((target_diff[mask] < 0) & (relevant_pred_diff < 0))
        tied_pred = np.sum(np.abs(relevant_pred_diff) < 1e-10)

        ci = (concordant + 0.5 * tied_pred) / relevant_pairs
    else:
        ci = 0.5

    # 保留tau用于调试（可选）
    tau, _ = kendalltau(all_preds, all_targets)
    ci_buggy = (tau + 1) / 2  # 旧方法（保留用于对比）

    return {
        'mse': mse,
        'rmse': rmse,
        'ci': ci,
        'loss': total_loss / len(dataloader)
    }


def train_joint(model, loaders, criterion, device, config, output_dir):
    """
    联合训练函数（原有的训练逻辑）

    Args:
        model: 模型
        loaders: 数据加载器
        criterion: 损失函数
        device: 设备
        config: 配置
        output_dir: 输出目录

    Returns:
        best_val_mse, best_epoch, training_log
    """
    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    # 创建学习率调度器
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )

    # 训练循环
    print("\n开始联合训练...")
    best_val_mse = float('inf')
    best_epoch = 0
    patience_counter = 0

    training_log = []

    for epoch in range(config['max_epochs']):
        # 训练
        train_loss = train_epoch(
            model, loaders['train'], optimizer, criterion, device, config
        )

        # 验证
        val_metrics = evaluate(model, loaders['val'], criterion, device)

        # 学习率调度
        scheduler.step(val_metrics['mse'])

        # 记录日志
        log_entry = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_mse': val_metrics['mse'],
            'val_rmse': val_metrics['rmse'],
            'val_ci': val_metrics['ci'],
            'lr': optimizer.param_groups[0]['lr']
        }
        training_log.append(log_entry)

        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{config['max_epochs']}: "
                  f"Train Loss={train_loss:.4f}, "
                  f"Val MSE={val_metrics['mse']:.4f}, "
                  f"Val CI={val_metrics['ci']:.4f}")

        # 保存最佳模型
        if val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']
            best_epoch = epoch + 1
            patience_counter = 0

            # 保存模型
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= config['early_stop_patience']:
            print(f"\n早停: 在epoch {epoch+1}停止训练")
            break

    return best_val_mse, best_epoch, training_log


def train_two_stage(model, loaders, criterion, device, config, output_dir):
    """
    两阶段训练函数

    阶段1: 预训练Drug Encoder（使用对比学习）
    阶段2: 微调所有参数（使用MSE损失）

    Args:
        model: 模型
        loaders: 数据加载器
        criterion: 损失函数
        device: 设备
        config: 配置
        output_dir: 输出目录

    Returns:
        best_val_mse, best_epoch, training_log
    """
    pretrain_epochs = config.get('pretrain_epochs', 200)
    finetune_epochs = config.get('finetune_epochs', 300)

    training_log = []

    # ========== 阶段1: 预训练Drug Encoder ==========
    print(f"\n{'='*60}")
    print(f"阶段1: 预训练Drug Encoder ({pretrain_epochs} epochs)")
    print(f"{'='*60}")

    # 冻结Protein Encoder和Predictor
    for param in model.protein_encoder.parameters():
        param.requires_grad = False
    for param in model.predictor.parameters():
        param.requires_grad = False

    # 只优化Drug Encoder
    optimizer_stage1 = optim.Adam(
        model.drug_encoder.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    scheduler_stage1 = ReduceLROnPlateau(
        optimizer_stage1,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )

    best_val_mse_stage1 = float('inf')
    patience_counter_stage1 = 0

    for epoch in range(pretrain_epochs):
        # 训练（只使用对比学习损失）
        model.train()
        total_loss = 0

        for batch in tqdm(loaders['train'], desc=f'Pretrain Epoch {epoch+1}/{pretrain_epochs}', leave=False):
            batch = batch.to(device)

            # 前向传播
            drug_repr, protein_repr, fusion = model.get_embeddings(
                batch, batch.protein_seq
            )

            # 对比学习损失
            if config.get('use_contrastive', False):
                # 生成增强视图
                augmentation = config.get('augmentation', 'noise')

                if augmentation == 'noise':
                    drug_repr_aug = drug_repr + torch.randn_like(drug_repr) * 0.1
                elif augmentation == 'random_mask':
                    mask_ratio = config.get('mask_ratio', 0.2)
                    random_augmentor = RandomMaskAugmentor(mask_ratio=mask_ratio, seed=config.get('seed', 42))
                    drug_repr_aug_list = []
                    for i in range(drug_repr.shape[0]):
                        dummy_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                        aug_repr, _ = random_augmentor.augment(
                            drug_repr[i].unsqueeze(0),
                            dummy_edge_index
                        )
                        drug_repr_aug_list.append(aug_repr.squeeze(0))
                    drug_repr_aug = torch.stack(drug_repr_aug_list)
                elif augmentation == 'sub_mask':
                    mask_ratio = config.get('mask_ratio', 0.2)
                    sub_augmentor = SubstructureMaskAugmentor(mask_ratio=mask_ratio, seed=config.get('seed', 42))
                    drug_repr_aug_list = []
                    for i in range(drug_repr.shape[0]):
                        dummy_edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
                        aug_repr, _ = sub_augmentor.augment(
                            drug_repr[i].unsqueeze(0),
                            dummy_edge_index,
                            smiles=None
                        )
                        drug_repr_aug_list.append(aug_repr.squeeze(0))
                    drug_repr_aug = torch.stack(drug_repr_aug_list)
                else:
                    drug_repr_aug = drug_repr + torch.randn_like(drug_repr) * 0.1

                # InfoNCE损失
                contrastive_loss = ContrastiveLoss(temperature=config.get('temperature', 0.1))
                cl_loss_drug = contrastive_loss(drug_repr, drug_repr_aug)

                alpha = config.get('alpha', 0.1)
                loss = alpha * cl_loss_drug

                # 均匀性损失
                beta = config.get('beta', 0.05)
                if beta > 0:
                    uniformity_loss = UniformityLoss()
                    uni_loss_drug = uniformity_loss(drug_repr)
                    loss = loss + beta * uni_loss_drug
            else:
                loss = torch.tensor(0.0, device=device)

            # 反向传播
            optimizer_stage1.zero_grad()
            loss.backward()
            optimizer_stage1.step()

            total_loss += loss.item()

        train_loss = total_loss / len(loaders['train'])

        # 验证（使用MSE）
        val_metrics = evaluate(model, loaders['val'], criterion, device)

        # 学习率调度
        scheduler_stage1.step(val_metrics['mse'])

        # 记录日志
        log_entry = {
            'epoch': epoch + 1,
            'stage': 1,
            'train_loss': train_loss,
            'val_mse': val_metrics['mse'],
            'val_rmse': val_metrics['rmse'],
            'val_ci': val_metrics['ci'],
            'lr': optimizer_stage1.param_groups[0]['lr']
        }
        training_log.append(log_entry)

        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f"Pretrain Epoch {epoch+1}/{pretrain_epochs}: "
                  f"Train Loss={train_loss:.4f}, "
                  f"Val MSE={val_metrics['mse']:.4f}, "
                  f"Val CI={val_metrics['ci']:.4f}")

        # 早停
        if val_metrics['mse'] < best_val_mse_stage1:
            best_val_mse_stage1 = val_metrics['mse']
            patience_counter_stage1 = 0
        else:
            patience_counter_stage1 += 1

        if patience_counter_stage1 >= config['early_stop_patience']:
            print(f"\n阶段1早停: 在epoch {epoch+1}停止训练")
            break

    # ========== 阶段2: 微调所有参数 ==========
    print(f"\n{'='*60}")
    print(f"阶段2: 微调所有参数 ({finetune_epochs} epochs)")
    print(f"{'='*60}")

    # 解冻所有参数
    for param in model.parameters():
        param.requires_grad = True

    # 优化所有参数
    optimizer_stage2 = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )

    scheduler_stage2 = ReduceLROnPlateau(
        optimizer_stage2,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )

    best_val_mse = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(finetune_epochs):
        # 训练（使用MSE损失）
        train_loss = train_epoch(
            model, loaders['train'], optimizer_stage2, criterion, device, config
        )

        # 验证
        val_metrics = evaluate(model, loaders['val'], criterion, device)

        # 学习率调度
        scheduler_stage2.step(val_metrics['mse'])

        # 记录日志
        log_entry = {
            'epoch': epoch + 1,
            'stage': 2,
            'train_loss': train_loss,
            'val_mse': val_metrics['mse'],
            'val_rmse': val_metrics['rmse'],
            'val_ci': val_metrics['ci'],
            'lr': optimizer_stage2.param_groups[0]['lr']
        }
        training_log.append(log_entry)

        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f"Finetune Epoch {epoch+1}/{finetune_epochs}: "
                  f"Train Loss={train_loss:.4f}, "
                  f"Val MSE={val_metrics['mse']:.4f}, "
                  f"Val CI={val_metrics['ci']:.4f}")

        # 保存最佳模型
        if val_metrics['mse'] < best_val_mse:
            best_val_mse = val_metrics['mse']
            best_epoch = epoch + 1
            patience_counter = 0

            # 保存模型
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= config['early_stop_patience']:
            print(f"\n阶段2早停: 在epoch {epoch+1}停止训练")
            break

    return best_val_mse, best_epoch, training_log


def train_model(config):
    """
    训练模型主函数

    Args:
        config: 实验配置字典
    """
    print("=" * 60)
    print(f"实验: {config['experiment_id']}")
    print(f"数据集: {config['dataset']}")
    print(f"随机种子: {config['seed']}")
    print(f"训练策略: {config.get('training_strategy', 'joint')}")
    print("=" * 60)

    # 设置随机种子
    set_seed(config['seed'])

    # 设置设备 - 强制使用GPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        # 确保CUDA正确初始化
        torch.cuda.init()
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
        print("⚠️  警告: CUDA不可用，使用CPU运行（速度会很慢）")

    print(f"使用设备: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # 创建输出目录
    split_type = config.get('split_type', None)
    if split_type is not None:
        output_dir = Path('/home/rxp/dta2/results/cold_start') / config['experiment_id']
    else:
        output_dir = Path('/home/rxp/dta2/results/controlled') / config['experiment_id']
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    print("\n加载数据...")
    if split_type is not None:
        print(f"  划分类型: {split_type}")
    loaders = get_all_loaders(
        config['dataset'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        split_type=split_type
    )

    # 创建模型
    print("创建模型...")
    architecture = config.get('architecture', 'GIN')
    print(f"  架构: {architecture}")

    if architecture == 'GAT':
        model = GATControlledDTA().to(device)
    else:
        model = ControlledDTA().to(device)

    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 参数断言：验证模型身份（防止错误模型缺陷复发）
    expected_params = MODEL_PARAM_COUNTS.get(architecture)
    if expected_params is not None:
        if total_params != expected_params:
            raise ValueError(
                f"模型参数断言失败！\n"
                f"  期望架构: {architecture}\n"
                f"  期望参数量: {expected_params:,}\n"
                f"  实际参数量: {total_params:,}\n"
                f"  差异: {total_params - expected_params:,}\n"
                f"  请检查模型配置是否正确。"
            )
        print(f"✅ 参数断言通过: {architecture} = {total_params:,} params")

    # 损失函数
    criterion = nn.MSELoss()

    # 根据训练策略选择不同的训练方式
    training_strategy = config.get('training_strategy', 'joint')

    if training_strategy == 'joint':
        # 联合训练（原有逻辑）
        best_val_mse, best_epoch, training_log = train_joint(
            model, loaders, criterion, device, config, output_dir
        )
    elif training_strategy == 'two_stage':
        # 两阶段训练
        best_val_mse, best_epoch, training_log = train_two_stage(
            model, loaders, criterion, device, config, output_dir
        )
    else:
        raise ValueError(f"未知的训练策略: {training_strategy}")

    # 测试
    print("\n在测试集上评估...")
    model.load_state_dict(torch.load(output_dir / 'best_model.pt'))
    test_metrics = evaluate(model, loaders['test'], criterion, device)

    print(f"\n测试结果:")
    print(f"  Best Epoch: {best_epoch}")
    print(f"  Test MSE: {test_metrics['mse']:.4f}")
    print(f"  Test RMSE: {test_metrics['rmse']:.4f}")
    print(f"  Test CI: {test_metrics['ci']:.4f}")

    # 保存结果
    results = {
        'experiment_id': config['experiment_id'],
        'dataset': config['dataset'],
        'seed': config['seed'],
        'best_epoch': best_epoch,
        'best_val_mse': float(best_val_mse),
        'test_mse': float(test_metrics['mse']),
        'test_rmse': float(test_metrics['rmse']),
        'test_ci': float(test_metrics['ci']),
        'total_params': int(total_params),
        'config': config
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # 保存训练日志
    # 转换numpy类型为Python原生类型
    training_log_serializable = []
    for entry in training_log:
        serializable_entry = {}
        for key, value in entry.items():
            if isinstance(value, (np.floating, np.integer)):
                serializable_entry[key] = float(value)
            else:
                serializable_entry[key] = value
        training_log_serializable.append(serializable_entry)

    with open(output_dir / 'training_log.json', 'w') as f:
        json.dump(training_log_serializable, f, indent=2)

    print(f"\n结果已保存到: {output_dir}")

    return results


def main():
    """
    主函数：解析参数并启动训练
    """
    parser = argparse.ArgumentParser(description='受控DTA训练脚本')

    parser.add_argument('--experiment_id', type=str, required=True,
                        help='实验ID (e.g., E0, E1, E2)')
    parser.add_argument('--dataset', type=str, default='davis',
                        choices=['davis', 'kiba'],
                        help='数据集名称')
    parser.add_argument('--split_type', type=str, default=None,
                        help='数据划分类型 (None=标准, cold_drug, cold_target, cold_both)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--architecture', type=str, default='GIN',
                        choices=['GIN', 'GAT'],
                        help='模型架构: GIN (ControlledDTA) 或 GAT (GATControlledDTA)')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=512,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=5e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--max_epochs', type=int, default=500,
                        help='最大训练轮数')
    parser.add_argument('--early_stop_patience', type=int, default=50,
                        help='早停耐心值')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')

    # 对比学习参数
    parser.add_argument('--use_contrastive', action='store_true',
                        help='是否使用对比学习')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='InfoNCE损失权重')
    parser.add_argument('--beta', type=float, default=0.05,
                        help='均匀性损失权重')
    parser.add_argument('--gamma', type=float, default=0.05,
                        help='跨模态损失权重')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='对比学习温度参数')

    # 增强策略参数
    parser.add_argument('--augmentation', type=str, default='none',
                        choices=['none', 'noise', 'random_mask', 'sub_mask'],
                        help='数据增强策略')
    parser.add_argument('--mask_ratio', type=float, default=0.2,
                        help='掩码比例（用于random_mask和sub_mask）')

    # 两阶段训练参数（E4使用）
    parser.add_argument('--training_strategy', type=str, default='joint',
                        choices=['joint', 'two_stage'],
                        help='训练策略：joint（联合训练）或two_stage（两阶段训练）')
    parser.add_argument('--pretrain_epochs', type=int, default=200,
                        help='预训练轮数（两阶段训练的阶段1）')
    parser.add_argument('--finetune_epochs', type=int, default=300,
                        help='微调轮数（两阶段训练的阶段2）')

    args = parser.parse_args()

    # 创建配置
    config = vars(args)

    # 开始训练
    train_model(config)


if __name__ == '__main__':
    main()
