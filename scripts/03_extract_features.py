#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征提取脚本
日期: 2026-08-04
目标: 从SMILES和蛋白质序列提取模型输入特征
方法:
  1. 药物特征: 使用RDKit提取原子特征，构建分子图
  2. 蛋白质特征: 氨基酸序列整数编码
输入: data/processed/{davis,kiba}_{train,val,test}.csv
输出: data/features/{dataset}_{split}.pt (PyG Data对象列表)
"""

import os
import sys
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from rdkit import Chem
from rdkit.Chem import rdmolops
from torch_geometric.data import Data

# ============================================================
# 路径配置
# ============================================================
BASE_DIR = Path('/home/rxp/dta2')
DATA_DIR = BASE_DIR / 'data'
PROCESSED_DIR = DATA_DIR / 'processed'
FEATURES_DIR = DATA_DIR / 'features'
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 蛋白质编码配置
# ============================================================
# 20种标准氨基酸 + X(未知) + PAD(填充)
AA_VOCAB = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19,
    'X': 20,  # 未知氨基酸
    '<PAD>': 21  # 填充符
}

MAX_PROTEIN_LENGTH = 1000  # 蛋白质序列最大长度


def encode_protein_sequence(sequence: str, max_length: int = MAX_PROTEIN_LENGTH) -> torch.Tensor:
    """
    将蛋白质序列编码为整数序列

    Args:
        sequence: 蛋白质序列字符串
        max_length: 最大序列长度（超过则截断，不足则填充）

    Returns:
        整数编码tensor, shape: [max_length]
    """
    # 转换为大写
    sequence = sequence.upper()

    # 编码
    encoded = []
    for aa in sequence:
        if aa in AA_VOCAB:
            encoded.append(AA_VOCAB[aa])
        else:
            # 非标准氨基酸编码为X
            encoded.append(AA_VOCAB['X'])

    # 截断或填充
    if len(encoded) > max_length:
        encoded = encoded[:max_length]
    else:
        encoded.extend([AA_VOCAB['<PAD>']] * (max_length - len(encoded)))

    return torch.tensor(encoded, dtype=torch.long)


def get_atom_features(atom) -> np.ndarray:
    """
    提取单个原子的特征向量 (35维)

    特征列表:
    1. 原子序数 (归一化, 1维)
    2. 度 (归一化, 1维)
    3. 形式电荷 (归一化, 1维)
    4. 杂化类型 (one-hot, 5种)
    5. 连接氢原子数 (归一化, 1维)
    6. 是否在环中 (1维)
    7. 是否芳香 (1维)
    8. 原子质量 (归一化, 1维)
    9. 总价电子数 (归一化, 1维)
    10. 自由基电子数 (归一化, 1维)
    11-35. 原子序数分桶 (one-hot, 25个桶)

    Returns:
        35维特征向量
    """
    features = []

    # 1. 原子序数 (归一化, 1-118 -> 0-1)
    atom_num = atom.GetAtomicNum()
    features.append(atom_num / 118.0)

    # 2. 度 (归一化, 0-10 -> 0-1)
    degree = atom.GetDegree()
    features.append(degree / 10.0)

    # 3. 形式电荷 (归一化, -5到+5 -> 0-1)
    formal_charge = atom.GetFormalCharge()
    features.append((formal_charge + 5) / 10.0)

    # 4. 杂化类型 (one-hot, 5种)
    hybridization = atom.GetHybridization()
    hybrid_types = [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D
    ]
    features.extend([int(hybridization == h) for h in hybrid_types])

    # 5. 连接氢原子数 (归一化, 0-8 -> 0-1)
    num_h = atom.GetTotalNumHs()
    features.append(num_h / 8.0)

    # 6. 是否在环中 (1维)
    features.append(int(atom.IsInRing()))

    # 7. 是否芳香 (1维)
    features.append(int(atom.GetIsAromatic()))

    # 8. 原子质量 (归一化, 0-300 -> 0-1)
    mass = atom.GetMass()
    features.append(mass / 300.0)

    # 9. 总价电子数 (归一化, 0-12 -> 0-1)
    valence = atom.GetTotalValence()
    features.append(valence / 12.0)

    # 10. 自由基电子数 (归一化, 0-5 -> 0-1)
    radical = atom.GetNumRadicalElectrons()
    features.append(radical / 5.0)

    # 11-35. 原子序数分桶 (one-hot, 25个桶)
    # 将原子序数分为25个桶
    atom_num_bins = 25
    bin_idx = min(int(atom_num / 5), atom_num_bins - 1)
    atom_onehot = [0] * atom_num_bins
    atom_onehot[bin_idx] = 1
    features.extend(atom_onehot)

    return np.array(features, dtype=np.float32)


def smiles_to_graph(smiles: str, affinity: float, substructure_mask: np.ndarray = None) -> Data:
    """
    将SMILES转换为PyG Data对象

    Args:
        smiles: 分子SMILES字符串
        affinity: 亲和力值
        substructure_mask: 子结构掩码 (可选)

    Returns:
        PyG Data对象
    """
    # 解析SMILES
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"无法解析SMILES: {smiles}")

    # 提取原子特征
    atom_features = []
    for atom in mol.GetAtoms():
        features = get_atom_features(atom)
        atom_features.append(features)

    atom_features = np.array(atom_features, dtype=np.float32)

    # 添加子结构标志位 (第36维)
    if substructure_mask is not None:
        assert len(substructure_mask) == len(atom_features), \
            f"子结构掩码长度不匹配: {len(substructure_mask)} vs {len(atom_features)}"
        atom_features = np.concatenate([
            atom_features,
            substructure_mask.reshape(-1, 1)
        ], axis=1)
    else:
        # 如果没有子结构掩码，使用0填充
        atom_features = np.concatenate([
            atom_features,
            np.zeros((len(atom_features), 1), dtype=np.float32)
        ], axis=1)

    # 提取边（化学键）
    edge_indices = []
    edge_attrs = []

    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        # 无向图，添加双向边
        edge_indices.append([i, j])
        edge_indices.append([j, i])

        # 边特征: 键类型 (one-hot, 5种)
        bond_type = bond.GetBondType()
        bond_features = [
            int(bond_type == Chem.rdchem.BondType.SINGLE),
            int(bond_type == Chem.rdchem.BondType.DOUBLE),
            int(bond_type == Chem.rdchem.BondType.TRIPLE),
            int(bond_type == Chem.rdchem.BondType.AROMATIC),
            int(bond.IsInRing())
        ]
        edge_attrs.append(bond_features)
        edge_attrs.append(bond_features)  # 双向边特征相同

    # 转换为tensor
    x = torch.tensor(atom_features, dtype=torch.float32)

    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        # 孤立原子（无化学键）
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 5), dtype=torch.float32)

    y = torch.tensor([affinity], dtype=torch.float32)

    # 创建PyG Data对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return data


def process_dataset(dataset_name: str, split: str) -> Tuple[List[Data], int]:
    """
    处理单个数据分割

    Args:
        dataset_name: 'davis' 或 'kiba'
        split: 'train', 'val', 或 'test'

    Returns:
        (data_list, num_samples)
    """
    print(f"处理 {dataset_name}_{split}...")

    # 读取CSV
    csv_file = PROCESSED_DIR / f'{dataset_name}_{split}.csv'
    df = pd.read_csv(csv_file)

    data_list = []
    errors = 0

    for idx, row in df.iterrows():
        try:
            smiles = row['SMILES']
            sequence = row['sequence']
            affinity = row['affinity']

            # 转换为图
            data = smiles_to_graph(smiles, affinity)

            # 编码蛋白质序列
            data.protein_seq = encode_protein_sequence(sequence)

            # 添加元数据
            data.drug_id = row['drug_id']
            data.target_id = row['target_id']

            data_list.append(data)

        except Exception as e:
            errors += 1
            if errors <= 5:  # 只打印前5个错误
                print(f"  警告: 处理样本 {idx} 失败: {e}")

    if errors > 0:
        print(f"  共 {errors} 个样本处理失败")

    print(f"  成功处理 {len(data_list)} 个样本")

    return data_list, len(data_list)


def compute_dataset_statistics(data_list: List[Data]) -> Dict:
    """
    计算数据集统计信息

    Args:
        data_list: PyG Data对象列表

    Returns:
        统计信息字典
    """
    # 统计原子特征维度
    atom_dims = [data.x.shape[1] for data in data_list]
    assert len(set(atom_dims)) == 1, "原子特征维度不一致"
    atom_dim = atom_dims[0]

    # 统计边特征维度
    edge_dims = [data.edge_attr.shape[1] for data in data_list if data.edge_attr.shape[0] > 0]
    assert len(set(edge_dims)) == 1, "边特征维度不一致"
    edge_dim = edge_dims[0] if edge_dims else 0

    # 统计蛋白质序列长度
    protein_lens = [data.protein_seq.shape[0] for data in data_list]

    # 统计每个分子的原子数
    atom_counts = [data.x.shape[0] for data in data_list]

    # 统计亲和力分布
    affinities = [data.y.item() for data in data_list]

    stats = {
        'num_samples': len(data_list),
        'atom_feature_dim': atom_dim,
        'edge_feature_dim': edge_dim,
        'protein_seq_length': MAX_PROTEIN_LENGTH,
        'protein_vocab_size': len(AA_VOCAB),
        'avg_atoms_per_mol': np.mean(atom_counts),
        'max_atoms_per_mol': int(np.max(atom_counts)),
        'min_atoms_per_mol': int(np.min(atom_counts)),
        'avg_protein_len': np.mean(protein_lens),
        'affinity_mean': np.mean(affinities),
        'affinity_std': np.std(affinities),
        'affinity_min': np.min(affinities),
        'affinity_max': np.max(affinities),
    }

    return stats


def save_features(data_list: List[Data], dataset_name: str, split: str):
    """
    保存特征到文件

    Args:
        data_list: PyG Data对象列表
        dataset_name: 'davis' 或 'kiba'
        split: 'train', 'val', 或 'test'
    """
    output_file = FEATURES_DIR / f'{dataset_name}_{split}.pt'

    # 保存为PyG格式
    torch.save(data_list, output_file)

    # 计算SHA256
    sha256_hash = hashlib.sha256()
    with open(output_file, 'rb') as f:
        for block in iter(lambda: f.read(4096), b''):
            sha256_hash.update(block)

    file_size_mb = output_file.stat().st_size / (1024 * 1024)

    print(f"  保存到: {output_file}")
    print(f"  文件大小: {file_size_mb:.2f} MB")
    print(f"  SHA256: {sha256_hash.hexdigest()[:32]}...")

    return {
        'file': str(output_file),
        'samples': len(data_list),
        'size_mb': file_size_mb,
        'sha256': sha256_hash.hexdigest()
    }


def main():
    """主函数"""
    print("=" * 60)
    print("DTA特征提取脚本")
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 检查输入文件
    datasets = ['davis', 'kiba']
    splits = ['train', 'val', 'test']

    all_stats = {}
    all_file_info = {}

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"处理 {dataset.upper()} 数据集")
        print(f"{'='*60}")

        all_stats[dataset] = {}
        all_file_info[dataset] = {}

        for split in splits:
            data_list, num_samples = process_dataset(dataset, split)

            # 计算统计信息
            stats = compute_dataset_statistics(data_list)
            all_stats[dataset][split] = stats

            # 保存特征
            file_info = save_features(data_list, dataset, split)
            all_file_info[dataset][split] = file_info

            print()

    # 打印统计摘要
    print("=" * 60)
    print("数据集统计摘要")
    print("=" * 60)

    for dataset in datasets:
        print(f"\n{dataset.upper()}:")
        for split in splits:
            stats = all_stats[dataset][split]
            print(f"  {split}:")
            print(f"    样本数: {stats['num_samples']}")
            print(f"    原子特征维度: {stats['atom_feature_dim']}")
            print(f"    边特征维度: {stats['edge_feature_dim']}")
            print(f"    平均原子数: {stats['avg_atoms_per_mol']:.1f}")
            print(f"    亲和力: {stats['affinity_mean']:.3f} ± {stats['affinity_std']:.3f}")

    # 保存统计信息到manifest
    manifest_file = DATA_DIR / 'raw' / 'manifest.json'
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)

    # 添加特征信息
    manifest['features'] = {
        'atom_feature_dim': all_stats['davis']['train']['atom_feature_dim'],
        'edge_feature_dim': all_stats['davis']['train']['edge_feature_dim'],
        'protein_vocab_size': len(AA_VOCAB),
        'protein_seq_length': MAX_PROTEIN_LENGTH,
        'aa_vocab': AA_VOCAB,
        'file_info': all_file_info,
        'statistics': all_stats
    }

    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n✅ 特征提取完成")
    print(f"统计信息已更新到: {manifest_file}")


if __name__ == '__main__':
    main()
