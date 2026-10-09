#!/usr/bin/env python3
"""
数据划分脚本
日期: 2026-08-04
目标: 将完整数据集按80/10/10比例划分为训练集、验证集、测试集
方法:
  1. 使用固定随机种子42确保可复现性
  2. 按样本随机划分（非按药物或靶点）
  3. 生成三个独立的CSV文件
  4. 记录每个文件的SHA256校验和
输入: data/processed/davis.csv, data/processed/kiba.csv
输出:
  - data/processed/davis_train.csv, davis_val.csv, davis_test.csv
  - data/processed/kiba_train.csv, kiba_val.csv, kiba_test.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import hashlib
from sklearn.model_selection import train_test_split

# 配置
RANDOM_SEED = 42
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

DATA_DIR = Path('data/processed')

def compute_sha256(filepath):
    """计算文件的SHA256校验和"""
    sha256_hash = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for block in iter(lambda: f.read(4096), b''):
            sha256_hash.update(block)
    return sha256_hash.hexdigest()

def split_dataset(dataset_name):
    """
    划分数据集

    Args:
        dataset_name: 'davis' 或 'kiba'
    """
    print(f"\n{'='*60}")
    print(f"处理 {dataset_name.upper()} 数据集")
    print(f"{'='*60}")

    # 读取完整数据集
    input_file = DATA_DIR / f'{dataset_name}.csv'
    df = pd.read_csv(input_file)
    print(f"读取数据: {len(df)} 个样本")

    # 第一次划分: 80% 训练, 20% 临时
    train_df, temp_df = train_test_split(
        df,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        shuffle=True
    )

    # 第二次划分: 从20%中划分出50%验证, 50%测试 (即总体的10%和10%)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=RANDOM_SEED,
        shuffle=True
    )

    print(f"划分结果:")
    print(f"  训练集: {len(train_df)} 样本 ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  验证集: {len(val_df)} 样本 ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  测试集: {len(test_df)} 样本 ({len(test_df)/len(df)*100:.1f}%)")

    # 保存文件
    train_file = DATA_DIR / f'{dataset_name}_train.csv'
    val_file = DATA_DIR / f'{dataset_name}_val.csv'
    test_file = DATA_DIR / f'{dataset_name}_test.csv'

    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    test_df.to_csv(test_file, index=False)

    print(f"保存文件:")
    print(f"  {train_file}")
    print(f"  {val_file}")
    print(f"  {test_file}")

    # 计算校验和
    train_sha256 = compute_sha256(train_file)
    val_sha256 = compute_sha256(val_file)
    test_sha256 = compute_sha256(test_file)

    print(f"SHA256校验:")
    print(f"  train: {train_sha256[:16]}...")
    print(f"  val:   {val_sha256[:16]}...")
    print(f"  test:  {test_sha256[:16]}...")

    return {
        'train': {
            'file': str(train_file),
            'samples': len(train_df),
            'ratio': len(train_df) / len(df),
            'sha256': train_sha256
        },
        'val': {
            'file': str(val_file),
            'samples': len(val_df),
            'ratio': len(val_df) / len(df),
            'sha256': val_sha256
        },
        'test': {
            'file': str(test_file),
            'samples': len(test_df),
            'ratio': len(test_df) / len(df),
            'sha256': test_sha256
        }
    }

def main():
    """主函数"""
    print("="*60)
    print("DTA数据集划分脚本")
    print("="*60)
    print(f"随机种子: {RANDOM_SEED}")
    print(f"划分比例: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")

    # 更新manifest.json
    manifest_file = Path('data/raw/manifest.json')
    with open(manifest_file, 'r') as f:
        manifest = json.load(f)

    # 划分Davis数据集
    davis_splits = split_dataset('davis')
    manifest['datasets']['davis']['splits'] = davis_splits

    # 划分KIBA数据集
    kiba_splits = split_dataset('kiba')
    manifest['datasets']['kiba']['splits'] = kiba_splits

    # 保存更新后的manifest
    manifest['split_config'] = {
        'random_seed': RANDOM_SEED,
        'train_ratio': TRAIN_RATIO,
        'val_ratio': VAL_RATIO,
        'test_ratio': TEST_RATIO,
        'split_method': 'random_sample_split'
    }

    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("更新 manifest.json")
    print(f"{'='*60}")
    print(f"已保存划分配置和校验信息")

    print("\n✅ 数据划分完成")

if __name__ == '__main__':
    main()
