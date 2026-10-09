#!/usr/bin/env python3
"""
冷启动特征提取脚本

使用与03_extract_features.py完全相同的特征提取逻辑，
处理冷启动数据划分（cold_drug, cold_target, cold_both）。

输入: data/cold_splits/davis_cold_{type}_{split}.csv
输出: data/features/davis_cold_{type}_{split}.pt
"""

import sys
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
from rdkit import Chem
from torch_geometric.data import Data

# Import feature extraction functions from the original script
import importlib.util
spec = importlib.util.spec_from_file_location(
    "extract_features",
    "/home/rxp/dta2/scripts/03_extract_features.py"
)
extract_features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract_features)

encode_protein_sequence = extract_features.encode_protein_sequence
smiles_to_graph = extract_features.smiles_to_graph
compute_dataset_statistics = extract_features.compute_dataset_statistics
AA_VOCAB = extract_features.AA_VOCAB
MAX_PROTEIN_LENGTH = extract_features.MAX_PROTEIN_LENGTH

BASE_DIR = Path('/home/rxp/dta2')
COLD_SPLITS_DIR = BASE_DIR / 'data' / 'cold_splits'
FEATURES_DIR = BASE_DIR / 'data' / 'features'
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_TYPES = ['cold_drug', 'cold_target', 'cold_both']
SPLITS = ['train', 'val', 'test']


def process_cold_dataset(split_type: str, split: str) -> Tuple[List[Data], int]:
    """处理冷启动数据分割"""
    csv_file = COLD_SPLITS_DIR / f'davis_{split_type}_{split}.csv'
    print(f"处理 davis_{split_type}_{split}...")

    df = pd.read_csv(csv_file)
    data_list = []
    errors = 0

    for idx, row in df.iterrows():
        try:
            smiles = row['SMILES']
            sequence = row['sequence']
            affinity = row['affinity']

            data = smiles_to_graph(smiles, affinity)
            data.protein_seq = encode_protein_sequence(sequence)
            data.drug_id = row['drug_id']
            data.target_id = row['target_id']
            data_list.append(data)

        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  警告: 处理样本 {idx} 失败: {e}")

    if errors > 0:
        print(f"  共 {errors} 个样本处理失败")

    print(f"  成功处理 {len(data_list)} 个样本")
    return data_list, len(data_list)


def save_features(data_list: List[Data], split_type: str, split: str):
    """保存特征"""
    output_file = FEATURES_DIR / f'davis_{split_type}_{split}.pt'
    torch.save(data_list, output_file)

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
    print("=" * 60)
    print("冷启动特征提取")
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    all_stats = {}

    for split_type in SPLIT_TYPES:
        print(f"\n{'='*60}")
        print(f"处理 {split_type.upper()} 划分")
        print(f"{'='*60}")
        all_stats[split_type] = {}

        for split in SPLITS:
            data_list, num_samples = process_cold_dataset(split_type, split)
            stats = compute_dataset_statistics(data_list)
            all_stats[split_type][split] = stats
            save_features(data_list, split_type, split)
            print()

    # Save manifest
    manifest = {
        'created': datetime.now().isoformat(),
        'split_type': 'cold_start',
        'statistics': all_stats
    }

    manifest_path = COLD_SPLITS_DIR / 'features_manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print("\n✅ 冷启动特征提取完成")
    for split_type in SPLIT_TYPES:
        for split in SPLITS:
            stats = all_stats[split_type][split]
            print(f"  {split_type}_{split}: {stats['num_samples']} samples, "
                  f"avg_atoms={stats['avg_atoms_per_mol']:.1f}, "
                  f"affinity={stats['affinity_mean']:.3f}±{stats['affinity_std']:.3f}")


if __name__ == '__main__':
    main()
