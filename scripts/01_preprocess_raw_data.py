#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理脚本 - 将DeepDTA原始数据转换为CSV格式
日期: 2026-08-04
目标: 解析DeepDTA格式的原始数据，生成标准化的CSV文件
方法:
  1. 使用numpy加载pickled亲和力矩阵
  2. 解析JSON格式的药物SMILES和蛋白质序列字典
  3. 将矩阵形式转换为长格式CSV（每行一个药物-靶点对）
  4. 过滤NaN值（无实验数据的配对）
输入: data/raw/davis/ 和 data/raw/kiba/ 目录下的原始文件
输出: data/processed/davis.csv 和 data/processed/kiba.csv
"""

import os
import sys
import json
import pickle
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ============================================================
# 路径配置
# ============================================================
RAW_DIR = Path("/home/rxp/dta2/data/raw")
PROCESSED_DIR = Path("/home/rxp/dta2/data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 数据加载函数
# ============================================================

def load_affinity_matrix(filepath: Path) -> np.ndarray:
    """
    加载亲和力矩阵（numpy pickle格式）

    DeepDTA的Y文件是numpy数组的pickle格式：
    - Davis: [68, 442] 矩阵，68个药物 × 442个靶点
    - KIBA: [2111, 229] 矩阵，2111个药物 × 229个靶点

    值说明:
    - Davis: Kd (nM)，需要转换为 pKd = -log10(Kd / 1e9)
    - KIBA: KIBA score

    NaN值表示无实验数据

    Args:
        filepath: Y文件路径
    Returns:
        numpy数组，形状为 [num_drugs, num_targets]
    """
    print(f"  加载亲和力矩阵: {filepath}")

    # 尝试多种加载方式（兼容Python 2/3 pickle格式差异）
    try:
        # 方式1: numpy load（兼容pickle格式）
        with open(filepath, 'rb') as f:
            affinity = np.load(f, allow_pickle=True)
        print(f"    ✅ numpy.load 成功")
    except Exception as e1:
        try:
            # 方式2: pickle with latin1 encoding (Python 2兼容)
            with open(filepath, 'rb') as f:
                affinity = pickle.load(f, encoding='latin1')
            print(f"    ✅ pickle.load (latin1) 成功")
        except Exception as e2:
            try:
                # 方式3: pickle with bytes encoding
                with open(filepath, 'rb') as f:
                    affinity = pickle.load(f, encoding='bytes')
                print(f"    ✅ pickle.load (bytes) 成功")
            except Exception as e3:
                print(f"    ❌ 加载失败:")
                print(f"       numpy: {e1}")
                print(f"       pickle(latin1): {e2}")
                print(f"       pickle(bytes): {e3}")
                raise

    print(f"    形状: {affinity.shape}")
    print(f"    数据类型: {affinity.dtype}")
    print(f"    非NaN数量: {np.sum(~np.isnan(affinity))}")
    print(f"    NaN数量: {np.sum(np.isnan(affinity))}")

    if affinity.ndim == 2:
        print(f"    药物数: {affinity.shape[0]}")
        print(f"    靶点数: {affinity.shape[1]}")

    return affinity


def load_json_dict(filepath: Path) -> dict:
    """
    加载JSON格式的字典文件

    DeepDTA的ligands_can.txt和proteins.txt是JSON字典格式：
    - ligands_can.txt: {drug_id: SMILES}
    - proteins.txt: {protein_id: sequence}

    Args:
        filepath: JSON文件路径
    Returns:
        字典 {str_id: value}
    """
    print(f"  加载字典文件: {filepath}")

    with open(filepath, 'r') as f:
        content = f.read()

    data = json.loads(content)

    print(f"    ✅ 成功加载 {len(data)} 个条目")
    print(f"    键类型: {type(list(data.keys())[0])}")
    print(f"    示例键: {list(data.keys())[:3]}")

    return data


def convert_kd_to_pkd(kd_nm: float) -> float:
    """
    将Kd (nM) 转换为 pKd

    公式: pKd = -log10(Kd / 1e9)

    例如:
    - Kd = 1 nM → pKd = -log10(1e-9) = 9.0
    - Kd = 100 nM → pKd = -log10(1e-7) = 7.0

    Args:
        kd_nm: Kd值，单位nM
    Returns:
        pKd值
    """
    return -np.log10(kd_nm / 1e9)


# ============================================================
# 数据转换函数
# ============================================================

def process_davis():
    """
    处理Davis数据集

    步骤:
    1. 加载亲和力矩阵 Y [68, 442]
    2. 加载药物SMILES字典
    3. 加载蛋白质序列字典
    4. 将Kd转换为pKd
    5. 过滤NaN值
    6. 生成长格式CSV

    输出列: drug_id, target_id, SMILES, sequence, affinity (pKd)
    """
    print("\n" + "="*60)
    print("处理Davis数据集")
    print("="*60)

    davis_dir = RAW_DIR / "davis"

    # 加载数据
    affinity_matrix = load_affinity_matrix(davis_dir / "Y")
    drugs_dict = load_json_dict(davis_dir / "ligands_can.txt")
    proteins_dict = load_json_dict(davis_dir / "proteins.txt")

    # 验证维度一致性
    num_drugs = affinity_matrix.shape[0]
    num_targets = affinity_matrix.shape[1]

    assert len(drugs_dict) == num_drugs, \
        f"药物数不匹配: 矩阵{num_drugs} vs 字典{len(drugs_dict)}"
    assert len(proteins_dict) == num_targets, \
        f"靶点数不匹配: 矩阵{num_targets} vs 字典{len(proteins_dict)}"

    print(f"\n  ✅ 维度验证通过")
    print(f"    药物数: {num_drugs}")
    print(f"    靶点数: {num_targets}")

    # 构建索引映射（保持原始顺序）
    drug_ids = list(drugs_dict.keys())
    target_ids = list(proteins_dict.keys())

    # 转换为长格式
    records = []
    for i, drug_id in enumerate(drug_ids):
        for j, target_id in enumerate(target_ids):
            affinity_value = affinity_matrix[i, j]

            # 跳过NaN值
            if np.isnan(affinity_value):
                continue

            # 转换Kd到pKd
            pKd = convert_kd_to_pkd(affinity_value)

            records.append({
                'drug_id': drug_id,
                'target_id': target_id,
                'SMILES': drugs_dict[drug_id],
                'sequence': proteins_dict[target_id],
                'affinity': pKd,
                'affinity_type': 'pKd'
            })

    # 创建DataFrame
    df = pd.DataFrame(records)

    print(f"\n  转换结果:")
    print(f"    总配对数: {len(records)}")
    print(f"    唯一药物数: {df['drug_id'].nunique()}")
    print(f"    唯一靶点数: {df['target_id'].nunique()}")
    print(f"    亲和力范围: [{df['affinity'].min():.3f}, {df['affinity'].max():.3f}]")
    print(f"    亲和力均值: {df['affinity'].mean():.3f} ± {df['affinity'].std():.3f}")

    # 保存CSV
    output_path = PROCESSED_DIR / "davis.csv"
    df.to_csv(output_path, index=False)

    print(f"\n  ✅ 保存到: {output_path}")
    print(f"    文件大小: {output_path.stat().st_size / 1024:.2f} KB")

    return df


def process_kiba():
    """
    处理KIBA数据集

    步骤:
    1. 加载亲和力矩阵 Y [2111, 229]
    2. 加载药物SMILES字典
    3. 加载蛋白质序列字典
    4. 过滤NaN值
    5. 生成长格式CSV

    注意: KIBA数据已经是KIBA score，不需要转换

    输出列: drug_id, target_id, SMILES, sequence, affinity (KIBA score)
    """
    print("\n" + "="*60)
    print("处理KIBA数据集")
    print("="*60)

    kiba_dir = RAW_DIR / "kiba"

    # 加载数据
    affinity_matrix = load_affinity_matrix(kiba_dir / "Y")
    drugs_dict = load_json_dict(kiba_dir / "ligands_can.txt")
    proteins_dict = load_json_dict(kiba_dir / "proteins.txt")

    # 验证维度一致性
    num_drugs = affinity_matrix.shape[0]
    num_targets = affinity_matrix.shape[1]

    assert len(drugs_dict) == num_drugs, \
        f"药物数不匹配: 矩阵{num_drugs} vs 字典{len(drugs_dict)}"
    assert len(proteins_dict) == num_targets, \
        f"靶点数不匹配: 矩阵{num_targets} vs 字典{len(proteins_dict)}"

    print(f"\n  ✅ 维度验证通过")
    print(f"    药物数: {num_drugs}")
    print(f"    靶点数: {num_targets}")

    # 构建索引映射（保持原始顺序）
    drug_ids = list(drugs_dict.keys())
    target_ids = list(proteins_dict.keys())

    # 转换为长格式
    records = []
    for i, drug_id in enumerate(drug_ids):
        for j, target_id in enumerate(target_ids):
            affinity_value = affinity_matrix[i, j]

            # 跳过NaN值
            if np.isnan(affinity_value):
                continue

            records.append({
                'drug_id': drug_id,
                'target_id': target_id,
                'SMILES': drugs_dict[drug_id],
                'sequence': proteins_dict[target_id],
                'affinity': affinity_value,
                'affinity_type': 'KIBA'
            })

    # 创建DataFrame
    df = pd.DataFrame(records)

    print(f"\n  转换结果:")
    print(f"    总配对数: {len(records)}")
    print(f"    唯一药物数: {df['drug_id'].nunique()}")
    print(f"    唯一靶点数: {df['target_id'].nunique()}")
    print(f"    亲和力范围: [{df['affinity'].min():.3f}, {df['affinity'].max():.3f}]")
    print(f"    亲和力均值: {df['affinity'].mean():.3f} ± {df['affinity'].std():.3f}")

    # 保存CSV
    output_path = PROCESSED_DIR / "kiba.csv"
    df.to_csv(output_path, index=False)

    print(f"\n  ✅ 保存到: {output_path}")
    print(f"    文件大小: {output_path.stat().st_size / 1024:.2f} KB")

    return df


# ============================================================
# 数据验证函数
# ============================================================

def validate_dataset(df: pd.DataFrame, dataset_name: str):
    """
    验证数据集的完整性和一致性

    检查项:
    1. 无缺失值
    2. SMILES格式有效
    3. 蛋白质序列仅含标准氨基酸
    4. 亲和力值在合理范围内
    """
    print(f"\n{'='*60}")
    print(f"{dataset_name} 数据集验证")
    print(f"{'='*60}")

    # 1. 缺失值检查
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(f"  ❌ 存在缺失值:")
        print(null_counts[null_counts > 0])
    else:
        print(f"  ✅ 无缺失值")

    # 2. SMILES有效性检查（简单检查）
    valid_smiles = df['SMILES'].str.len() > 0
    print(f"  ✅ SMILES有效性: {valid_smiles.sum()}/{len(df)} ({valid_smiles.mean()*100:.1f}%)")

    # 3. 蛋白质序列检查
    valid_aa = set('ACDEFGHIKLMNPQRSTVWY')
    invalid_seq = []
    for idx, seq in df['sequence'].items():
        if not all(c in valid_aa for c in seq.upper()):
            invalid_seq.append(idx)

    if invalid_seq:
        print(f"  ⚠️  发现 {len(invalid_seq)} 个含非标准氨基酸的序列")
    else:
        print(f"  ✅ 所有序列仅含标准氨基酸")

    # 4. 亲和力范围检查
    if dataset_name == "Davis":
        # pKd通常在3-12之间
        valid_range = (df['affinity'] >= 3) & (df['affinity'] <= 12)
    else:
        # KIBA score通常在3-17之间
        valid_range = (df['affinity'] >= 3) & (df['affinity'] <= 17)

    print(f"  ✅ 亲和力范围合理: {valid_range.sum()}/{len(df)} ({valid_range.mean()*100:.1f}%)")

    # 5. 统计摘要
    print(f"\n  统计摘要:")
    print(f"    样本数: {len(df)}")
    print(f"    唯一药物: {df['drug_id'].nunique()}")
    print(f"    唯一靶点: {df['target_id'].nunique()}")
    print(f"    亲和力均值: {df['affinity'].mean():.3f} ± {df['affinity'].std():.3f}")
    print(f"    亲和力中位数: {df['affinity'].median():.3f}")
    print(f"    亲和力范围: [{df['affinity'].min():.3f}, {df['affinity'].max():.3f}]")


def compute_file_hash(filepath: Path) -> str:
    """
    计算文件SHA256哈希
    """
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for block in iter(lambda: f.read(8192), b''):
            sha256.update(block)
    return sha256.hexdigest()


# ============================================================
# 主函数
# ============================================================

def main():
    """
    主处理流程
    """
    print("="*60)
    print("DTA数据预处理工具")
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # 检查原始数据是否存在
    davis_dir = RAW_DIR / "davis"
    kiba_dir = RAW_DIR / "kiba"

    if not davis_dir.exists():
        print(f"❌ Davis数据目录不存在: {davis_dir}")
        return 1

    if not kiba_dir.exists():
        print(f"❌ KIBA数据目录不存在: {kiba_dir}")
        return 1

    # 处理Davis
    try:
        df_davis = process_davis()
        validate_dataset(df_davis, "Davis")
    except Exception as e:
        print(f"❌ Davis处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 处理KIBA
    try:
        df_kiba = process_kiba()
        validate_dataset(df_kiba, "KIBA")
    except Exception as e:
        print(f"❌ KIBA处理失败: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 生成数据清单
    manifest = {
        "created_at": datetime.now().isoformat(),
        "source": "DeepDTA (hkmztrk/DeepDTA on GitHub)",
        "datasets": {}
    }

    for name, filepath in [("davis", PROCESSED_DIR / "davis.csv"),
                            ("kiba", PROCESSED_DIR / "kiba.csv")]:
        df = pd.read_csv(filepath)
        manifest["datasets"][name] = {
            "samples": len(df),
            "unique_drugs": int(df['drug_id'].nunique()),
            "unique_targets": int(df['target_id'].nunique()),
            "affinity_type": df['affinity_type'].iloc[0],
            "sha256": compute_file_hash(filepath),
            "file_size_kb": round(filepath.stat().st_size / 1024, 2)
        }

    # 保存清单
    manifest_path = RAW_DIR / "manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("数据清单")
    print(f"{'='*60}")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))

    print(f"\n✅ 所有数据处理完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
