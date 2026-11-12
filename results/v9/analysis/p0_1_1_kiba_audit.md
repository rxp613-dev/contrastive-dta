# KIBA实验全量审计报告

**时间**: 2026-09-08T14:51:45.832515

**总目录数**: 115

**有效记录数**: 115

## 按(model, batch)分组统计

| Model | Batch | 目录数 | n(种子) | 真实alpha值 |
|-------|-------|--------|---------|-------------|
| GAT | ablation | 30 | 5 | [0.003, 0.01, 0.1] |
| GAT | best | 5 | 5 | [0.05] |
| GAT | fixed100 | 10 | 5 | [0.003, 0.1] |
| GAT | lr0.0001 | 20 | 10 | [0.001, 0.01, 0.05] |
| GAT | lr1e4 | 15 | 5 | [0.01] |
| GIN | unknown | 35 | 5 | [0.001, 0.005, 0.01, 0.03, 0.05, 0.1] |

## 完整审计表

| 目录名 | Model | 真实α | lr | seed | test_mse | test_ci | weight_decay | beta | augmentation |
|--------|-------|-------|----|------|----------|---------|--------------|------|--------------|
| GIN_KIBA_alpha001_seed123 | GIN | 0.001 | None | 123 | 0.6656 | 0.7029 | 0.0001 | 0.0 | noise |
| GIN_KIBA_alpha001_seed2024 | GIN | 0.001 | None | 2024 | 0.5595 | 0.6861 | 0.0001 | 0.0 | noise |
| GIN_KIBA_alpha001_seed3407 | GIN | 0.001 | None | 3407 | 0.5362 | 0.6956 | 0.0001 | 0.0 | noise |
| GIN_KIBA_alpha001_seed42 | GIN | 0.001 | None | 42 | 0.5814 | 0.6810 | 0.0001 | 0.0 | noise |
| GIN_KIBA_alpha001_seed7 | GIN | 0.001 | None | 7 | 0.4738 | 0.7224 | 0.0001 | 0.0 | noise |
| GIN_KIBA_alpha005_seed123 | GIN | 0.005 | None | 123 | 0.6257 | 0.6718 | 0.0001 | 0.0 | noise |
| GIN_KIBA_alpha005_seed2024 | GIN | 0.005 | None | 2024 | 0.5534 | 0.6898 | 0.0001 | 0.0 | noise |
| GIN_KIBA_alpha005_seed3407 | GIN | 0.005 | None | 3407 | 0.5623 | 0.6844 | 0.0001 | 0.0 | noise |
| GIN_KIBA_alpha005_seed42 | GIN | 0.005 | None | 42 | 0.5722 | 0.7016 | 0.0001 | 0.0 | noise |
| GIN_KIBA_alpha005_seed7 | GIN | 0.005 | None | 7 | 0.4790 | 0.7708 | 0.0001 | 0.0 | noise |
| GIN_KIBA_alpha0_seed123 | GIN | 0.1 | None | 123 | 0.5742 | 0.6784 | 0.0001 | 0.05 | noise |
| GIN_KIBA_alpha0_seed2024 | GIN | 0.1 | None | 2024 | 0.5537 | 0.6920 | 0.0001 | 0.05 | noise |
| GIN_KIBA_alpha0_seed3407 | GIN | 0.1 | None | 3407 | 0.5312 | 0.6956 | 0.0001 | 0.05 | noise |
| GIN_KIBA_alpha0_seed42 | GIN | 0.1 | None | 42 | 0.5520 | 0.6985 | 0.0001 | 0.05 | noise |
| GIN_KIBA_alpha0_seed7 | GIN | 0.1 | None | 7 | 0.2122 | 0.8302 | 0.0001 | 0.05 | noise |
| GIN_KIBA_alpha100_seed123 | GIN | 0.1 | None | 123 | 0.5252 | 0.7003 | 0.0001 | 0.0 | noise |
| GIN_KIBA_alpha100_seed2024 | GIN | 0.1 | None | 2024 | 0.5073 | 0.7060 | 0.0001 | 0.0 | noise |
| GIN_KIBA_alpha100_seed3407 | GIN | 0.1 | None | 3407 | 0.3042 | 0.7875 | 0.0001 | 0.0 | noise |
| GIN_KIBA_alpha100_seed42 | GIN | 0.1 | None | 42 | 0.1797 | 0.8437 | 0.0001 | 0.0 | noise |
| GIN_KIBA_alpha100_seed7 | GIN | 0.1 | None | 7 | 0.1942 | 0.8363 | 0.0001 | 0.0 | noise |
| GIN_KIBA_alpha10_seed123 | GIN | 0.01 | None | 123 | 0.6470 | 0.6598 | 0.0001 | 0.0 | noise |
| GIN_KIBA_alpha10_seed2024 | GIN | 0.01 | None | 2024 | 0.5385 | 0.6945 | 0.0001 | 0.0 | noise |
| GIN_KIBA_alpha10_seed3407 | GIN | 0.01 | None | 3407 | 0.5266 | 0.6959 | 0.0001 | 0.0 | noise |
| GIN_KIBA_alpha10_seed42 | GIN | 0.01 | None | 42 | 0.4936 | 0.7108 | 0.0001 | 0.0 | noise |
| GIN_KIBA_alpha10_seed7 | GIN | 0.01 | None | 7 | 0.3856 | 0.7803 | 0.0001 | 0.0 | noise |
| GIN_KIBA_alpha30_seed123 | GIN | 0.03 | None | 123 | 0.6311 | 0.6743 | 0.0001 | 0.0 | noise |
| GIN_KIBA_alpha30_seed2024 | GIN | 0.03 | None | 2024 | 0.5762 | 0.6841 | 0.0001 | 0.0 | noise |
| GIN_KIBA_alpha30_seed3407 | GIN | 0.03 | None | 3407 | 0.5642 | 0.6852 | 0.0001 | 0.0 | noise |
| GIN_KIBA_alpha30_seed42 | GIN | 0.03 | None | 42 | 0.5204 | 0.7018 | 0.0001 | 0.0 | noise |
| GIN_KIBA_alpha30_seed7 | GIN | 0.03 | None | 7 | 0.2778 | 0.7998 | 0.0001 | 0.0 | noise |
| GIN_KIBA_alpha50_seed123 | GIN | 0.05 | None | 123 | 0.4975 | 0.7060 | 0.0001 | 0.0 | noise |
| GIN_KIBA_alpha50_seed2024 | GIN | 0.05 | None | 2024 | 0.5935 | 0.6677 | 0.0001 | 0.0 | noise |
| GIN_KIBA_alpha50_seed3407 | GIN | 0.05 | None | 3407 | 0.4927 | 0.7152 | 0.0001 | 0.0 | noise |
| GIN_KIBA_alpha50_seed42 | GIN | 0.05 | None | 42 | 0.4502 | 0.7272 | 0.0001 | 0.0 | noise |
| GIN_KIBA_alpha50_seed7 | GIN | 0.05 | None | 7 | 0.2853 | 0.7967 | 0.0001 | 0.0 | noise |
| KIBA_E0_fixed100_seed123 | GAT | 0.1 | None | 123 | 0.6392 | 0.6720 | 0.0001 | 0.05 | none |
| KIBA_E0_fixed100_seed2024 | GAT | 0.1 | None | 2024 | 0.5319 | 0.6933 | 0.0001 | 0.05 | none |
| KIBA_E0_fixed100_seed3407 | GAT | 0.1 | None | 3407 | 0.5637 | 0.7163 | 0.0001 | 0.05 | none |
| KIBA_E0_fixed100_seed42 | GAT | 0.1 | None | 42 | 0.5441 | 0.6890 | 0.0001 | 0.05 | none |
| KIBA_E0_fixed100_seed7 | GAT | 0.1 | None | 7 | 0.4317 | 0.7424 | 0.0001 | 0.05 | none |
| KIBA_E0_lr0.0001_seed123 | GAT | 0.1 | None | 123 | 0.3877 | 0.7583 | 0.0001 | 0.05 | none |
| KIBA_E0_lr0.0001_seed2024 | GAT | 0.1 | None | 2024 | 0.4002 | 0.7510 | 0.0001 | 0.05 | none |
| KIBA_E0_lr0.0001_seed3407 | GAT | 0.1 | None | 3407 | 0.4485 | 0.7358 | 0.0001 | 0.05 | none |
| KIBA_E0_lr0.0001_seed42 | GAT | 0.1 | None | 42 | 0.4380 | 0.7350 | 0.0001 | 0.05 | none |
| KIBA_E0_lr0.0001_seed7 | GAT | 0.1 | None | 7 | 0.4055 | 0.7515 | 0.0001 | 0.05 | none |
| KIBA_E0_seed123 | GAT | 0.1 | None | 123 | 0.5617 | 0.6886 | 0.0001 | 0.05 | none |
| KIBA_E0_seed2024 | GAT | 0.1 | None | 2024 | 0.5769 | 0.6886 | 0.0001 | 0.05 | none |
| KIBA_E0_seed3407 | GAT | 0.1 | None | 3407 | 0.5439 | 0.6908 | 0.0001 | 0.05 | none |
| KIBA_E0_seed42 | GAT | 0.1 | None | 42 | 0.5447 | 0.6909 | 0.0001 | 0.05 | none |
| KIBA_E0_seed7 | GAT | 0.1 | None | 7 | 0.4686 | 0.7216 | 0.0001 | 0.05 | none |
| KIBA_E2_lr1e4_seed123 | GAT | 0.01 | None | 123 | 0.3294 | 0.7765 | 0.0001 | 0.0 | random_mask |
| KIBA_E2_lr1e4_seed2024 | GAT | 0.01 | None | 2024 | 0.4016 | 0.7497 | 0.0001 | 0.0 | random_mask |
| KIBA_E2_lr1e4_seed3407 | GAT | 0.01 | None | 3407 | 0.4645 | 0.7304 | 0.0001 | 0.0 | random_mask |
| KIBA_E2_lr1e4_seed42 | GAT | 0.01 | None | 42 | 0.4350 | 0.7406 | 0.0001 | 0.0 | random_mask |
| KIBA_E2_lr1e4_seed7 | GAT | 0.01 | None | 7 | 0.3915 | 0.7666 | 0.0001 | 0.0 | random_mask |
| KIBA_E2_seed123 | GAT | 0.01 | None | 123 | 0.5858 | 0.6817 | 0.0001 | 0.0 | random_mask |
| KIBA_E2_seed2024 | GAT | 0.01 | None | 2024 | 0.5769 | 0.6777 | 0.0001 | 0.0 | random_mask |
| KIBA_E2_seed3407 | GAT | 0.01 | None | 3407 | 0.5500 | 0.6887 | 0.0001 | 0.0 | random_mask |
| KIBA_E2_seed42 | GAT | 0.01 | None | 42 | 0.5313 | 0.6963 | 0.0001 | 0.0 | random_mask |
| KIBA_E2_seed7 | GAT | 0.01 | None | 7 | 0.2924 | 0.7936 | 0.0001 | 0.0 | random_mask |
| KIBA_E3_lr1e4_seed123 | GAT | 0.01 | None | 123 | 0.3520 | 0.7723 | 0.0001 | 0.0 | sub_mask |
| KIBA_E3_lr1e4_seed2024 | GAT | 0.01 | None | 2024 | 0.4207 | 0.7413 | 0.0001 | 0.0 | sub_mask |
| KIBA_E3_lr1e4_seed3407 | GAT | 0.01 | None | 3407 | 0.4667 | 0.7214 | 0.0001 | 0.0 | sub_mask |
| KIBA_E3_lr1e4_seed42 | GAT | 0.01 | None | 42 | 0.4107 | 0.7523 | 0.0001 | 0.0 | sub_mask |
| KIBA_E3_lr1e4_seed7 | GAT | 0.01 | None | 7 | 0.3996 | 0.7523 | 0.0001 | 0.0 | sub_mask |
| KIBA_E3_seed123 | GAT | 0.01 | None | 123 | 0.5401 | 0.6885 | 0.0001 | 0.0 | sub_mask |
| KIBA_E3_seed2024 | GAT | 0.01 | None | 2024 | 0.5387 | 0.6898 | 0.0001 | 0.0 | sub_mask |
| KIBA_E3_seed3407 | GAT | 0.01 | None | 3407 | 0.5490 | 0.6887 | 0.0001 | 0.0 | sub_mask |
| KIBA_E3_seed42 | GAT | 0.01 | None | 42 | 0.5232 | 0.7027 | 0.0001 | 0.0 | sub_mask |
| KIBA_E3_seed7 | GAT | 0.01 | None | 7 | 0.2799 | 0.7999 | 0.0001 | 0.0 | sub_mask |
| KIBA_E5_lr1e4_seed123 | GAT | 0.01 | None | 123 | 0.3954 | 0.7615 | 0.0001 | 0.0 | noise |
| KIBA_E5_lr1e4_seed2024 | GAT | 0.01 | None | 2024 | 0.4017 | 0.7518 | 0.0001 | 0.0 | noise |
| KIBA_E5_lr1e4_seed3407 | GAT | 0.01 | None | 3407 | 0.4550 | 0.7308 | 0.0001 | 0.0 | noise |
| KIBA_E5_lr1e4_seed42 | GAT | 0.01 | None | 42 | 0.3896 | 0.7613 | 0.0001 | 0.0 | noise |
| KIBA_E5_lr1e4_seed7 | GAT | 0.01 | None | 7 | 0.3680 | 0.7596 | 0.0001 | 0.0 | noise |
| KIBA_E5_seed123 | GAT | 0.01 | None | 123 | 0.5173 | 0.7035 | 0.0001 | 0.0 | noise |
| KIBA_E5_seed2024 | GAT | 0.01 | None | 2024 | 0.5309 | 0.6966 | 0.0001 | 0.0 | noise |
| KIBA_E5_seed3407 | GAT | 0.01 | None | 3407 | 0.5520 | 0.6946 | 0.0001 | 0.0 | noise |
| KIBA_E5_seed42 | GAT | 0.01 | None | 42 | 0.5758 | 0.7354 | 0.0001 | 0.0 | noise |
| KIBA_E5_seed7 | GAT | 0.01 | None | 7 | 0.2639 | 0.8047 | 0.0001 | 0.0 | noise |
| KIBA_E7_best_seed123 | GAT | 0.05 | None | 123 | 0.5599 | 0.7012 | 0.0001 | 0.0 | noise |
| KIBA_E7_best_seed2024 | GAT | 0.05 | None | 2024 | 0.6146 | 0.6945 | 0.0001 | 0.0 | noise |
| KIBA_E7_best_seed3407 | GAT | 0.05 | None | 3407 | 0.5278 | 0.6958 | 0.0001 | 0.0 | noise |
| KIBA_E7_best_seed42 | GAT | 0.05 | None | 42 | 0.4473 | 0.7724 | 0.0001 | 0.0 | noise |
| KIBA_E7_best_seed7 | GAT | 0.05 | None | 7 | 0.2599 | 0.8062 | 0.0001 | 0.0 | noise |
| KIBA_E7_fixed100_seed123 | GAT | 0.003 | None | 123 | 0.6160 | 0.7029 | 0.0001 | 0.05 | noise |
| KIBA_E7_fixed100_seed2024 | GAT | 0.003 | None | 2024 | 0.6567 | 0.6684 | 0.0001 | 0.05 | noise |
| KIBA_E7_fixed100_seed3407 | GAT | 0.003 | None | 3407 | 0.6114 | 0.6671 | 0.0001 | 0.05 | noise |
| KIBA_E7_fixed100_seed42 | GAT | 0.003 | None | 42 | 0.5310 | 0.7052 | 0.0001 | 0.05 | noise |
| KIBA_E7_fixed100_seed7 | GAT | 0.003 | None | 7 | 0.2842 | 0.7948 | 0.0001 | 0.05 | noise |
| KIBA_E7_lr0.0001_seed123 | GAT | 0.003 | None | 123 | 0.3701 | 0.7689 | 0.0001 | 0.05 | noise |
| KIBA_E7_lr0.0001_seed2024 | GAT | 0.003 | None | 2024 | 0.3883 | 0.7588 | 0.0001 | 0.05 | noise |
| KIBA_E7_lr0.0001_seed3407 | GAT | 0.003 | None | 3407 | 0.4603 | 0.7309 | 0.0001 | 0.05 | noise |
| KIBA_E7_lr0.0001_seed42 | GAT | 0.003 | None | 42 | 0.4195 | 0.7486 | 0.0001 | 0.05 | noise |
| KIBA_E7_lr0.0001_seed7 | GAT | 0.003 | None | 7 | 0.3817 | 0.7679 | 0.0001 | 0.05 | noise |
| KIBA_alpha0010_lr0.0001_seed123 | GAT | 0.001 | None | 123 | 0.3164 | 0.7918 | 0.0001 | 0.05 | noise |
| KIBA_alpha0010_lr0.0001_seed2024 | GAT | 0.001 | None | 2024 | 0.4019 | 0.7585 | 0.0001 | 0.05 | noise |
| KIBA_alpha0010_lr0.0001_seed3407 | GAT | 0.001 | None | 3407 | 0.4524 | 0.7348 | 0.0001 | 0.05 | noise |
| KIBA_alpha0010_lr0.0001_seed42 | GAT | 0.001 | None | 42 | 0.3592 | 0.7811 | 0.0001 | 0.05 | noise |
| KIBA_alpha0010_lr0.0001_seed7 | GAT | 0.001 | None | 7 | 0.3753 | 0.7715 | 0.0001 | 0.05 | noise |
| KIBA_alpha010_lr0.0001_seed123 | GAT | 0.01 | None | 123 | 0.3449 | 0.7825 | 0.0001 | 0.05 | noise |
| KIBA_alpha010_lr0.0001_seed2024 | GAT | 0.01 | None | 2024 | 0.3987 | 0.7606 | 0.0001 | 0.05 | noise |
| KIBA_alpha010_lr0.0001_seed3407 | GAT | 0.01 | None | 3407 | 0.4400 | 0.7453 | 0.0001 | 0.05 | noise |
| KIBA_alpha010_lr0.0001_seed42 | GAT | 0.01 | None | 42 | 0.4148 | 0.7690 | 0.0001 | 0.05 | noise |
| KIBA_alpha010_lr0.0001_seed7 | GAT | 0.01 | None | 7 | 0.3833 | 0.7716 | 0.0001 | 0.05 | noise |
| KIBA_alpha050_lr0.0001_seed101 | GAT | 0.05 | None | 101 | 0.3466 | 0.7797 | 0.0001 | 0.05 | noise |
| KIBA_alpha050_lr0.0001_seed123 | GAT | 0.05 | None | 123 | 0.2936 | 0.7990 | 0.0001 | 0.05 | noise |
| KIBA_alpha050_lr0.0001_seed13 | GAT | 0.05 | None | 13 | 0.3736 | 0.7679 | 0.0001 | 0.05 | noise |
| KIBA_alpha050_lr0.0001_seed2024 | GAT | 0.05 | None | 2024 | 0.3987 | 0.7633 | 0.0001 | 0.05 | noise |
| KIBA_alpha050_lr0.0001_seed2026 | GAT | 0.05 | None | 2026 | 0.3840 | 0.7692 | 0.0001 | 0.05 | noise |
| KIBA_alpha050_lr0.0001_seed3407 | GAT | 0.05 | None | 3407 | 0.3449 | 0.7793 | 0.0001 | 0.05 | noise |
| KIBA_alpha050_lr0.0001_seed42 | GAT | 0.05 | None | 42 | 0.3836 | 0.7693 | 0.0001 | 0.05 | noise |
| KIBA_alpha050_lr0.0001_seed555 | GAT | 0.05 | None | 555 | 0.3910 | 0.7657 | 0.0001 | 0.05 | noise |
| KIBA_alpha050_lr0.0001_seed7 | GAT | 0.05 | None | 7 | 0.3472 | 0.7794 | 0.0001 | 0.05 | noise |
| KIBA_alpha050_lr0.0001_seed999 | GAT | 0.05 | None | 999 | 0.3156 | 0.7908 | 0.0001 | 0.05 | noise |

## 关键发现

### n=10批次识别

- **[GAT, lr0.0001]**: n=10, alpha=[0.001, 0.01, 0.05]
  - 种子列表: [7, 13, 42, 101, 123, 555, 999, 2024, 2026, 3407]

### 目录名alpha vs 真实alpha不一致

- ⚠️  `GIN_KIBA_alpha0_seed123`: 目录名推断alpha=0.0, 真实alpha=0.1
- ⚠️  `GIN_KIBA_alpha0_seed2024`: 目录名推断alpha=0.0, 真实alpha=0.1
- ⚠️  `GIN_KIBA_alpha0_seed3407`: 目录名推断alpha=0.0, 真实alpha=0.1
- ⚠️  `GIN_KIBA_alpha0_seed42`: 目录名推断alpha=0.0, 真实alpha=0.1
- ⚠️  `GIN_KIBA_alpha0_seed7`: 目录名推断alpha=0.0, 真实alpha=0.1
- ⚠️  `KIBA_alpha0010_lr0.0001_seed123`: 目录名推断alpha=0.01, 真实alpha=0.001
- ⚠️  `KIBA_alpha0010_lr0.0001_seed2024`: 目录名推断alpha=0.01, 真实alpha=0.001
- ⚠️  `KIBA_alpha0010_lr0.0001_seed3407`: 目录名推断alpha=0.01, 真实alpha=0.001
- ⚠️  `KIBA_alpha0010_lr0.0001_seed42`: 目录名推断alpha=0.01, 真实alpha=0.001
- ⚠️  `KIBA_alpha0010_lr0.0001_seed7`: 目录名推断alpha=0.01, 真实alpha=0.001
