#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据下载脚本
日期: 2026-08-04
目标: 下载Davis和KIBA数据集的原始数据
方法: 从多个可靠来源下载，确保数据完整性
"""

import os
import sys
import urllib.request
import tarfile
import hashlib
from pathlib import Path

# 数据目录
DATA_DIR = Path("/home/rxp/dta2/data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 数据来源（多个备用源）
DATA_SOURCES = {
    "davis": {
        "primary": "https://staff.cs.ut.ee/~kaalepi/datasets/davis.tar.gz",
        "backup1": "https://raw.githubusercontent.com/huangkiki/dta-graph-dataset/main/davis.tar.gz",
        "filename": "davis.tar.gz",
        "extract_dir": "davis"
    },
    "kiba": {
        "primary": "https://staff.cs.ut.ee/~kaalepi/datasets/kiba.tar.gz",
        "backup1": "https://raw.githubusercontent.com/huangkiki/dta-graph-dataset/main/kiba.tar.gz",
        "filename": "kiba.tar.gz",
        "extract_dir": "kiba"
    }
}

def download_file(url: str, dest_path: Path, desc: str) -> bool:
    """
    下载文件
    Args:
        url: 下载链接
        dest_path: 目标路径
        desc: 描述信息
    Returns:
        bool: 是否成功
    """
    try:
        print(f"[下载] {desc}")
        print(f"  URL: {url}")
        print(f"  目标: {dest_path}")

        # 下载文件
        urllib.request.urlretrieve(url, dest_path)

        # 验证文件大小
        file_size = dest_path.stat().st_size
        if file_size < 1000:  # 小于1KB可能是错误页面
            print(f"  ❌ 文件过小 ({file_size} bytes)，可能是错误页面")
            dest_path.unlink()
            return False

        print(f"  ✅ 下载成功 ({file_size / 1024 / 1024:.2f} MB)")
        return True

    except Exception as e:
        print(f"  ❌ 下载失败: {e}")
        if dest_path.exists():
            dest_path.unlink()
        return False

def extract_tar_gz(tar_path: Path, extract_dir: Path):
    """
    解压tar.gz文件
    """
    print(f"[解压] {tar_path.name} -> {extract_dir}")

    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_dir)

    print(f"  ✅ 解压完成")

def compute_sha256(file_path: Path) -> str:
    """
    计算文件SHA256
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(block)
    return sha256_hash.hexdigest()

def main():
    print("="*60)
    print("DTA数据集下载工具")
    print("="*60)

    results = {}

    for dataset_name, sources in DATA_SOURCES.items():
        print(f"\n{'='*60}")
        print(f"下载 {dataset_name.upper()} 数据集")
        print(f"{'='*60}")

        tar_path = DATA_DIR / sources["filename"]
        extract_dir = DATA_DIR / sources["extract_dir"]

        # 检查是否已存在
        if tar_path.exists():
            print(f"[跳过] {sources['filename']} 已存在")
        else:
            # 尝试从主源下载
            success = download_file(sources["primary"], tar_path, f"{dataset_name} (主源)")

            # 如果失败，尝试备用源
            if not success and "backup1" in sources:
                print(f"\n[尝试备用源]")
                success = download_file(sources["backup1"], tar_path, f"{dataset_name} (备用源)")

            if not success:
                print(f"❌ {dataset_name} 下载失败")
                results[dataset_name] = False
                continue

        # 解压
        if not extract_dir.exists() or not any(extract_dir.iterdir()):
            try:
                extract_tar_gz(tar_path, extract_dir)
            except Exception as e:
                print(f"❌ 解压失败: {e}")
                results[dataset_name] = False
                continue

        # 计算SHA256
        sha256 = compute_sha256(tar_path)
        print(f"[校验] SHA256: {sha256[:16]}...")

        results[dataset_name] = True

    # 总结
    print(f"\n{'='*60}")
    print("下载结果汇总")
    print(f"{'='*60}")

    for dataset_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{dataset_name.upper()}: {status}")

    # 检查是否全部成功
    if all(results.values()):
        print(f"\n✅ 所有数据集下载成功")
        return 0
    else:
        print(f"\n❌ 部分数据集下载失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
