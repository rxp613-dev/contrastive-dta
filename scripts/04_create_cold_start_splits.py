#!/usr/bin/env python3
"""
冷启动数据划分脚本

创建三种冷启动划分：
1. cold-drug: 测试集药物不出现在训练集
2. cold-target: 测试集靶点不出现在训练集
3. cold-both: 测试集药物和靶点都不出现在训练集

输出：
- data/cold_splits/davis_cold_drug_{train,val,test}.csv
- data/cold_splits/davis_cold_target_{train,val,test}.csv
- data/cold_splits/davis_cold_both_{train,val,test}.csv
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import json
import hashlib
import os

def compute_sha256(filepath):
    """计算文件SHA256"""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def cold_drug_split(df, test_ratio=0.2, val_ratio=0.1, seed=42):
    """
    冷启动药物划分

    测试集中的药物不出现在训练集中
    """
    np.random.seed(seed)

    # 获取所有唯一药物
    all_drugs = df['drug_id'].unique()
    n_drugs = len(all_drugs)

    # 随机选择测试集药物（20%）
    test_drugs = np.random.choice(all_drugs, size=int(n_drugs * test_ratio), replace=False)
    train_val_drugs = np.setdiff1d(all_drugs, test_drugs)

    # 从train_val_drugs中选择验证集药物（10%/90% ≈ 11%）
    val_drugs = np.random.choice(train_val_drugs, size=int(len(train_val_drugs) * val_ratio / (1 - test_ratio)), replace=False)
    train_drugs = np.setdiff1d(train_val_drugs, val_drugs)

    print(f"Cold-drug split:")
    print(f"  Train drugs: {len(train_drugs)}")
    print(f"  Val drugs: {len(val_drugs)}")
    print(f"  Test drugs: {len(test_drugs)}")

    # 按药物划分数据
    test_df = df[df['drug_id'].isin(test_drugs)]
    val_df = df[df['drug_id'].isin(val_drugs)]
    train_df = df[df['drug_id'].isin(train_drugs)]

    # 验证无泄漏
    assert len(set(train_df['drug_id']) & set(test_df['drug_id'])) == 0, "Drug leak: train-test"
    assert len(set(train_df['drug_id']) & set(val_df['drug_id'])) == 0, "Drug leak: train-val"
    assert len(set(val_df['drug_id']) & set(test_df['drug_id'])) == 0, "Drug leak: val-test"

    return train_df, val_df, test_df

def cold_target_split(df, test_ratio=0.2, val_ratio=0.1, seed=42):
    """
    冷启动靶点划分

    测试集中的靶点不出现在训练集中
    """
    np.random.seed(seed)

    # 获取所有唯一靶点
    all_targets = df['target_id'].unique()
    n_targets = len(all_targets)

    # 随机选择测试集靶点（20%）
    test_targets = np.random.choice(all_targets, size=int(n_targets * test_ratio), replace=False)
    train_val_targets = np.setdiff1d(all_targets, test_targets)

    # 从train_val_targets中选择验证集靶点
    val_targets = np.random.choice(train_val_targets, size=int(len(train_val_targets) * val_ratio / (1 - test_ratio)), replace=False)
    train_targets = np.setdiff1d(train_val_targets, val_targets)

    print(f"Cold-target split:")
    print(f"  Train targets: {len(train_targets)}")
    print(f"  Val targets: {len(val_targets)}")
    print(f"  Test targets: {len(test_targets)}")

    # 按靶点划分数据
    test_df = df[df['target_id'].isin(test_targets)]
    val_df = df[df['target_id'].isin(val_targets)]
    train_df = df[df['target_id'].isin(train_targets)]

    # 验证无泄漏
    assert len(set(train_df['target_id']) & set(test_df['target_id'])) == 0, "Target leak: train-test"
    assert len(set(train_df['target_id']) & set(val_df['target_id'])) == 0, "Target leak: train-val"
    assert len(set(val_df['target_id']) & set(test_df['target_id'])) == 0, "Target leak: val-test"

    return train_df, val_df, test_df

def cold_both_split(df, drug_test_ratio=0.2, target_test_ratio=0.2, seed=42):
    """
    冷启动双冷划分

    测试集中的药物和靶点都不出现在训练集中
    """
    np.random.seed(seed)

    # 获取所有唯一药物和靶点
    all_drugs = df['drug_id'].unique()
    all_targets = df['target_id'].unique()

    # 随机选择测试集药物和靶点
    test_drugs = np.random.choice(all_drugs, size=int(len(all_drugs) * drug_test_ratio), replace=False)
    test_targets = np.random.choice(all_targets, size=int(len(all_targets) * target_test_ratio), replace=False)

    train_drugs = np.setdiff1d(all_drugs, test_drugs)
    train_targets = np.setdiff1d(all_targets, test_targets)

    print(f"Cold-both split:")
    print(f"  Train drugs: {len(train_drugs)}, Test drugs: {len(test_drugs)}")
    print(f"  Train targets: {len(train_targets)}, Test targets: {len(test_targets)}")

    # 测试集：药物和靶点都是新的
    test_df = df[df['drug_id'].isin(test_drugs) & df['target_id'].isin(test_targets)]

    # 训练集：药物和靶点都是旧的
    train_df = df[df['drug_id'].isin(train_drugs) & df['target_id'].isin(train_targets)]

    # 验证集：从训练药物和训练靶点中选
    val_drugs = np.random.choice(train_drugs, size=int(len(train_drugs) * 0.1), replace=False)
    val_targets = np.random.choice(train_targets, size=int(len(train_targets) * 0.1), replace=False)
    val_df = df[df['drug_id'].isin(val_drugs) & df['target_id'].isin(val_targets)]

    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples: {len(val_df)}")
    print(f"  Test samples: {len(test_df)}")

    # 验证无泄漏
    assert len(set(train_df['drug_id']) & set(test_df['drug_id'])) == 0, "Drug leak"
    assert len(set(train_df['target_id']) & set(test_df['target_id'])) == 0, "Target leak"

    return train_df, val_df, test_df

def main():
    # 加载数据
    print("Loading Davis dataset...")
    df = pd.read_csv('/home/rxp/dta2/data/processed/davis.csv')
    print(f"Total samples: {len(df)}")
    print(f"Unique drugs: {df['drug_id'].nunique()}")
    print(f"Unique targets: {df['target_id'].nunique()}")

    # 创建输出目录
    os.makedirs('/home/rxp/dta2/data/cold_splits', exist_ok=True)

    # 创建划分清单
    manifest = {
        'seed': 42,
        'test_ratio': 0.2,
        'val_ratio': 0.1,
        'splits': {}
    }

    # 1. Cold-drug split
    print("\n" + "="*60)
    print("Creating cold-drug split...")
    train_df, val_df, test_df = cold_drug_split(df)

    train_path = '/home/rxp/dta2/data/cold_splits/davis_cold_drug_train.csv'
    val_path = '/home/rxp/dta2/data/cold_splits/davis_cold_drug_val.csv'
    test_path = '/home/rxp/dta2/data/cold_splits/davis_cold_drug_test.csv'

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    manifest['splits']['cold_drug'] = {
        'train': {
            'samples': len(train_df),
            'drugs': train_df['drug_id'].nunique(),
            'targets': train_df['target_id'].nunique(),
            'sha256': compute_sha256(train_path)
        },
        'val': {
            'samples': len(val_df),
            'drugs': val_df['drug_id'].nunique(),
            'targets': val_df['target_id'].nunique(),
            'sha256': compute_sha256(val_path)
        },
        'test': {
            'samples': len(test_df),
            'drugs': test_df['drug_id'].nunique(),
            'targets': test_df['target_id'].nunique(),
            'sha256': compute_sha256(test_path)
        }
    }

    # 2. Cold-target split
    print("\n" + "="*60)
    print("Creating cold-target split...")
    train_df, val_df, test_df = cold_target_split(df)

    train_path = '/home/rxp/dta2/data/cold_splits/davis_cold_target_train.csv'
    val_path = '/home/rxp/dta2/data/cold_splits/davis_cold_target_val.csv'
    test_path = '/home/rxp/dta2/data/cold_splits/davis_cold_target_test.csv'

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    manifest['splits']['cold_target'] = {
        'train': {
            'samples': len(train_df),
            'drugs': train_df['drug_id'].nunique(),
            'targets': train_df['target_id'].nunique(),
            'sha256': compute_sha256(train_path)
        },
        'val': {
            'samples': len(val_df),
            'drugs': val_df['drug_id'].nunique(),
            'targets': val_df['target_id'].nunique(),
            'sha256': compute_sha256(val_path)
        },
        'test': {
            'samples': len(test_df),
            'drugs': test_df['drug_id'].nunique(),
            'targets': test_df['target_id'].nunique(),
            'sha256': compute_sha256(test_path)
        }
    }

    # 3. Cold-both split
    print("\n" + "="*60)
    print("Creating cold-both split...")
    train_df, val_df, test_df = cold_both_split(df)

    train_path = '/home/rxp/dta2/data/cold_splits/davis_cold_both_train.csv'
    val_path = '/home/rxp/dta2/data/cold_splits/davis_cold_both_val.csv'
    test_path = '/home/rxp/dta2/data/cold_splits/davis_cold_both_test.csv'

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    manifest['splits']['cold_both'] = {
        'train': {
            'samples': len(train_df),
            'drugs': train_df['drug_id'].nunique(),
            'targets': train_df['target_id'].nunique(),
            'sha256': compute_sha256(train_path)
        },
        'val': {
            'samples': len(val_df),
            'drugs': val_df['drug_id'].nunique(),
            'targets': val_df['target_id'].nunique(),
            'sha256': compute_sha256(val_path)
        },
        'test': {
            'samples': len(test_df),
            'drugs': test_df['drug_id'].nunique(),
            'targets': test_df['target_id'].nunique(),
            'sha256': compute_sha256(test_path)
        }
    }

    # 保存清单
    manifest_path = '/home/rxp/dta2/data/cold_splits/manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "="*60)
    print("✅ Cold start splits created successfully!")
    print(f"Output directory: /home/rxp/dta2/data/cold_splits/")
    print(f"Manifest: {manifest_path}")

    # 打印汇总
    print("\n" + "="*60)
    print("Summary:")
    for split_name, split_data in manifest['splits'].items():
        print(f"\n{split_name}:")
        print(f"  Train: {split_data['train']['samples']} samples, "
              f"{split_data['train']['drugs']} drugs, "
              f"{split_data['train']['targets']} targets")
        print(f"  Val:   {split_data['val']['samples']} samples, "
              f"{split_data['val']['drugs']} drugs, "
              f"{split_data['val']['targets']} targets")
        print(f"  Test:  {split_data['test']['samples']} samples, "
              f"{split_data['test']['drugs']} drugs, "
              f"{split_data['test']['targets']} targets")

if __name__ == '__main__':
    main()
