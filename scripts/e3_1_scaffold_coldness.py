#!/usr/bin/env python3
"""
E3.1: Davis scaffold 划分冷度验证脚本
================================================================================
时间: 2026-10-08
任务来源: 《实验复核意见与下一阶段实验方案.md》实验E3
目标:
  验证既有 Bemis-Murcko scaffold 划分(data/cold_start_scaffold/, 2026-08-15
  任务4.1产出, 70/10/20, 无泄漏断言+SHA256 manifest)的"冷度":
  测试集(及验证集)每个药物与训练集全部药物的 max Tanimoto 相似度，
  验收标准: max Tanimoto 中位数 < 0.3。
方法:
  - RDKit Morgan 指纹 (radius=2, 2048bit)
  - 对每个 test/val 药物计算与全部 train 药物的 Tanimoto，取最大值
  - 报告中位数、均值、分布，以及对照(标准随机划分下 Davis 药物100%重叠)
输入: data/cold_start_scaffold/davis_scaffold_{train,val,test}.csv
输出: results/analysis/e3_1_scaffold_coldness.json + 终端报告
备注: 只读分析，不修改任何文件
================================================================================
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

SPLIT_DIR = Path('/home/rxp/dta2/data/cold_start_scaffold')
OUT = Path('/home/rxp/dta2/results/analysis/e3_1_scaffold_coldness.json')


def fp_of(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)


def max_tanimoto(query_fps, ref_fps):
    """每个query指纹对全部ref指纹的最大Tanimoto"""
    out = []
    for q in query_fps:
        sims = DataStructs.BulkTanimotoSimilarity(q, ref_fps)
        out.append(max(sims) if sims else 0.0)
    return np.array(out)


def main():
    print('=' * 70)
    print('E3.1: scaffold 划分冷度验证 (max Tanimoto vs 训练集)')
    print('=' * 70)

    dfs = {}
    for name in ('train', 'val', 'test'):
        df = pd.read_csv(SPLIT_DIR / f'davis_scaffold_{name}.csv')
        dfs[name] = df
        print(f'{name}: {len(df)} 样本, {df["SMILES"].nunique()} 唯一药物')

    # 训练集指纹库
    train_smiles = dfs['train']['SMILES'].unique()
    train_fps = {}
    for s in train_smiles:
        f = fp_of(s)
        if f is not None:
            train_fps[s] = f
    print(f'\n训练集有效指纹: {len(train_fps)}/{len(train_smiles)}')
    ref = list(train_fps.values())

    report = {'script': 'scripts/e3_1_scaffold_coldness.py',
              'split_dir': str(SPLIT_DIR)}
    for name in ('val', 'test'):
        smiles = dfs[name]['SMILES'].unique()
        q = []
        valid_smiles = []
        for s in smiles:
            f = fp_of(s)
            if f is not None:
                q.append(f)
                valid_smiles.append(s)
        mts = max_tanimoto(q, ref)
        assert len(valid_smiles) > 0, f'{name}: 无有效SMILES'
        stats = dict(
            n_drugs=len(valid_smiles),
            max_tanimoto_median=float(np.median(mts)),
            max_tanimoto_mean=float(np.mean(mts)),
            max_tanimoto_min=float(np.min(mts)),
            max_tanimoto_max=float(np.max(mts)),
            frac_below_03=float(np.mean(mts < 0.3)),
            frac_below_04=float(np.mean(mts < 0.4)),
        )
        report[name] = stats
        print(f'\n[{name}] n={stats["n_drugs"]} 唯一药物')
        print(f'  max Tanimoto: 中位数={stats["max_tanimoto_median"]:.3f} '
              f'均值={stats["max_tanimoto_mean"]:.3f} '
              f'范围=[{stats["max_tanimoto_min"]:.3f}, {stats["max_tanimoto_max"]:.3f}]')
        print(f'  <0.3 比例: {stats["frac_below_03"]:.1%}  '
              f'<0.4 比例: {stats["frac_below_04"]:.1%}')

    # 验收判定 (方案E3: 测试集-训练集 max Tanimoto 中位数 < 0.3)
    passed = report['test']['max_tanimoto_median'] < 0.3
    report['acceptance_median_below_0.3'] = bool(passed)
    print('\n' + '=' * 70)
    print(f'验收 (test max Tanimoto 中位数 < 0.3): '
          f'{"✅ 通过" if passed else "❌ 未通过"} '
          f'(实测 {report["test"]["max_tanimoto_median"]:.3f})')
    print('=' * 70)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f'报告已保存: {OUT}')


if __name__ == '__main__':
    main()
