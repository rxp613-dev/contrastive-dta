# Contrastive DTA: Systematic Empirical Benchmark

> **Paper**: "Contrastive Learning for Drug-Target Affinity Prediction: A Systematic Empirical Benchmark"  
> **Authors**: Xiaoping Ren, Chaoya Sui, Xiaoyong Liu, Xiaoli Ren, Ling Tao, Yan Zhang  
> **Contact**: rxp613@gmail.com (Xiaoping Ren, corresponding author)

## Overview

This repository provides **complete reproducibility** for our systematic empirical benchmark of contrastive learning (CL) for drug-target affinity (DTA) prediction. We conducted **300+ training runs** across 5 dimensions (loss components, α values, architectures, data splits, datasets) with 5–10 seeds per configuration.

**Key contribution**: We identified critical data governance issues in prior CL-DTA work and established rigorous validation protocols, including:
- **115-directory audit** of KIBA experiments with config-level verification
- **Holm-corrected statistical analysis** (family size = 6 for KIBA α-scan, family size = 7 for Davis)
- **KIBA n=10 authoritative scan**: monotonic dose-response with significant +42.74% MSE reduction at α=0.1 (Holm p=0.001); an earlier small-α "sign reversal" was shown to be an artifact of a cross-learning-rate baseline comparison and does not replicate under the matched protocol
- **Cold-target vulnerability** characterization (α-dependent; avoidable at α ≤ 0.01)

## Environment Setup

### Requirements
```bash
# Python 3.8+
conda create -n contrastive-dta python=3.8
conda activate contrastive-dta

# Core dependencies
pip install torch==2.0.1 torchvision==0.15.2
pip install torch-geometric==2.3.1
pip install deepchem==2.7.1
pip install scipy==1.10.1 numpy==1.24.3 pandas==2.0.1
pip install scikit-learn==1.2.2 statsmodels==0.14.0
pip install matplotlib==3.7.1 seaborn==0.12.2
```

### Hardware
- **Training**: NVIDIA GPU (RTX 3090/A100 recommended, 24GB+ VRAM)
- **Analysis**: CPU sufficient for all statistical scripts
- **Total GPU time**: ~200 GPU-hours for full reproduction

## Quick Start: Reproduce Key Results

### 1. Download and preprocess datasets
```bash
# Download Davis and KIBA datasets
python scripts/download_data.py

# Preprocess features (molecular graphs + protein sequences)
python scripts/01_preprocess_raw_data.py
python scripts/02_split_dataset.py
python scripts/03_extract_features.py

# Create cold-start splits
python scripts/04_create_cold_start_splits.py
python scripts/05_extract_cold_start_features.py
```

**Expected output**: `data/processed/` directory with train/val/test splits for Davis and KIBA.

### 2. Train baseline models (Table 3-5 results)
```bash
# Davis dataset, GIN, α=0.01 (reproduces +13.26% MSE reduction, p=0.0196)
for seed in 7 42 123 2024 3407; do
    python src/train_contrastive.py \
        --dataset davis \
        --encoder gin \
        --alpha 0.01 \
        --beta 0.0 \
        --seed $seed \
        --output_dir results/davis_gin_alpha001_seed${seed}
done

# Aggregate results and compute statistics
python scripts/aggregate_results.py \
    --result_dirs results/davis_gin_alpha001_seed* \
    --output results/authoritative/davis_gin_alpha001.json

# Expected: MSE reduction = 13.26%, Holm p-value = 0.0196
```

```bash
# KIBA dataset, GIN, α=0.1 (reproduces +42.74% MSE reduction, Holm p=0.001, n=10)
# Note: 5 original seeds {7,42,123,2024,3407} + 5 extension seeds {251,1496,5934,7350,9889}
for seed in 7 42 123 2024 3407 251 1496 5934 7350 9889; do
    python src/train_contrastive.py \
        --dataset kiba \
        --encoder gin \
        --alpha 0.1 \
        --beta 0.0 \
        --seed $seed \
        --output_dir results/kiba_gin_alpha100_seed${seed}
done

python scripts/aggregate_results.py \
    --result_dirs results/kiba_gin_alpha100_seed* \
    --output results/authoritative/kiba_gin_alpha100.json

# Expected: MSE reduction = +42.74%, Holm p-value = 0.001 (n=10)
```

### 3. Reproduce α-scan (Table 16, Figure 10)
```bash
# Full α-scan for KIBA (6 configurations)
for alpha in 0.001 0.005 0.01 0.03 0.05 0.1; do
    alpha_str=$(printf "%03d" $(echo "$alpha * 1000" | bc))
    for seed in 7 42 123 2024 3407; do
        python src/train_contrastive.py \
            --dataset kiba \
            --encoder gin \
            --alpha $alpha \
            --beta 0.0 \
            --seed $seed \
            --output_dir results/kiba_alpha_scan/alpha${alpha_str}_seed${seed}
    done
done

# Compute Holm-corrected p-values (family size = 6)
python scripts/p0_2_holm_final.py \
    --result_dir results/kiba_alpha_scan/ \
    --output results/authoritative/kiba_alpha_scan_holm.json

# Expected output (Table 16, n=10 authoritative scan):
# α=0.001: +5.08%  (Holm p=0.809, non-significant)
# α=0.005: +8.63%  (Holm p=0.809, non-significant)
# α=0.01:  +10.72% (Holm p=0.809, non-significant)
# α=0.03:  +20.54% (Holm p=0.262, non-significant)
# α=0.05:  +25.96% (Holm p=0.054, borderline)
# α=0.1:   +42.74% (Holm p=0.001**, significant)
# Note: monotonic dose-response; the earlier n=5 "sign reversal" was an
# artifact of a cross-learning-rate baseline comparison and does not replicate.
```

### 4. Reproduce cold-target experiments (Table 17-19)
```bash
# Cold-target GAT experiments (reproduces -20.74% degradation, p<0.0001)
for seed in 7 42 123 2024 3407; do
    python src/train_contrastive.py \
        --dataset davis \
        --encoder gat \
        --alpha 0.01 \
        --beta 0.0 \
        --split cold_target \
        --seed $seed \
        --output_dir results/cold_target_gat_alpha001_seed${seed}
done

python scripts/aggregate_results.py \
    --result_dirs results/cold_target_gat_alpha001_seed* \
    --output results/authoritative/cold_target_gat.json

# Expected: MSE change = -20.74%, p<0.0001
```

## Data Governance: Audit and Verification

### KIBA 115-Directory Audit
Our paper identified that prior KIBA claims included an **n=10 batch with β=0.05** (protocol violation). We audited all 115 directories:

```bash
# Full audit: identify protocol violations
python scripts/p0_1_1_kiba_audit.py \
    --scan_dir results/controlled/ \
    --output docs/audit_tables/kiba_115_directory_audit.json

# Deep audit: analyze n=10 batch contamination
python scripts/p0_1_2_kiba_deep_audit.py \
    --audit_result docs/audit_tables/kiba_115_directory_audit.json \
    --output docs/audit_tables/kiba_n10_batch_analysis.json

# Expected findings:
# - 80 directories excluded (n=10 batch with β=0.05, legacy directories)
# - 35 directories retained (5 seeds × 6 α values + 5 E0 seeds)
# - Root cause: directory naming misreading (alpha050 ≠ α=0.50, but α=0.05)
```

### Final Verification Script
Ensures all deprecated values are removed and authoritative values are present:

```bash
python scripts/final_consistency_check.py

# Expected output:
# ✅ ALL CHECKS PASSED
#    - 47/47 required values present, 36/36 deprecated values absent
#    - DOI/author check, cold-start provenance check, narrative terminology check
```

### KIBA n=10 Extension (October 2026)
The authoritative KIBA scan was extended from n=5 to n=10 seeds:

```bash
# 1. Run the 8 missing experiments (idempotent; skips completed runs)
python scripts/e1_1_kiba_n10_completion.py

# 2. Per-seed diagnosis + Welch t + Holm (family=6) analysis
python scripts/e1_2_kiba_n10_analysis.py

# 3. Validation-set α selection re-analysis (n=10)
python scripts/e1_4_kiba_val_selection.py

# 4. T11/T4 cold-start batch config audit (50 directories)
python scripts/e5_1_t11_t4_config_audit.py
```

Key outputs in `results/authoritative/`:
- `canonical_results_v2.csv` — single authoritative results table (pair-counting CI throughout)
- `e1_2_kiba_n10_analysis.json` — full n=10 statistical analysis
- `e1_4_kiba_val_selection_n10.json` — validation-set α selection (no optimistic bias)
- `e3_3_scaffold_stats.json` — Bemis–Murcko scaffold cold-drug results (-38.22%, p=0.075)

## Repository Structure

```
contrastive-dta/
├── README.md                          # This file
├── CITATION.cff                       # Citation metadata
├── LICENSE                            # MIT License
├── src/                               # Core training code
│   ├── train_contrastive.py           # Main training script
│   ├── models/                        # GIN/GAT architectures
│   ├── losses/                        # InfoNCE, Uniformity, CrossModal
│   └── data/                          # Dataset loaders
├── scripts/                           # Analysis and audit scripts
│   ├── download_data.py               # Dataset download
│   ├── 01_preprocess_raw_data.py      # Feature extraction
│   ├── train_controlled.py            # Controlled training (GIN/GAT, CL losses, audits)
│   ├── aggregate_results.py           # Statistical aggregation
│   ├── p0_1_1_kiba_audit.py           # 115-directory audit
│   ├── p0_1_2_kiba_deep_audit.py      # n=10 batch analysis
│   ├── p0_2_holm_final.py             # Holm correction (family=6)
│   ├── e1_1_kiba_n10_completion.py    # KIBA n=10 training driver
│   ├── e1_2_kiba_n10_analysis.py      # n=10 statistics (Welch + Holm)
│   ├── e1_3_update_canonical.py       # Canonical CSV update
│   ├── e1_4_kiba_val_selection.py     # Validation-set α selection (n=10)
│   ├── e3_*.py                        # Bemis–Murcko scaffold split experiments
│   ├── e4_1_transfer.py               # Cross-dataset transfer (Davis↔KIBA)
│   ├── e5_1_t11_t4_config_audit.py    # T11/T4 config audit (50 dirs)
│   ├── e5_2_canonical_ci_fix.py       # CI convention unification
│   └── final_consistency_check.py     # Final verification (47 values + 36 deprecated)
├── results/                           # Authoritative results
│   └── authoritative/                 # JSON/CSV files for all key results
│       ├── canonical_results_v2.csv   # Single authoritative results table
│       ├── e1_2_kiba_n10_analysis.json
│       ├── e1_4_kiba_val_selection_n10.json
│       ├── e3_3_scaffold_stats.json
│       ├── p0_2_holm_final.json
│       └── cold_target_gat.json
└── docs/                              # Documentation
    └── audit_tables/                  # Audit reports
        ├── kiba_115_directory_audit.json
        └── kiba_n10_batch_analysis.json
```

## Key Results Summary

| Finding | Dataset | Metric | Value | p-value | Seeds |
|---------|---------|--------|-------|---------|-------|
| CL benefit (GIN) | Davis | MSE reduction | +13.26% | 0.0196* | 5 |
| CL benefit (GAT) | Davis | MSE reduction | +18.85% | <0.001* | 5 |
| CL benefit (GIN, α=0.1) | KIBA | MSE reduction | +42.74% | 0.001** | 10 |
| Monotonic dose-response | KIBA | improvement grows with α | +5.08% → +42.74% | — | 10 |
| Cold-target degradation | Davis (GAT) | MSE change | -20.74% | <0.0001* | 5 |
| Scaffold cold-drug (α=0.05) | Davis (GIN) | MSE change | -38.22% | 0.075 | 5 |

*Holm-corrected, significant at α=0.05. The KIBA n=10 scan supersedes an earlier n=5 analysis whose "sign reversal" was an artifact of a cross-learning-rate baseline comparison.

## Statistical Methods

- **Primary test**: Paired t-test (Davis, ddof=1) and Welch's t-test (KIBA, equal_var=False) for per-seed MSE comparisons
- **Multiple comparison correction**: Holm-Bonferroni method
- **Family size**: 7 for Davis (E1–E7), 6 for KIBA α-scan (α ∈ {0.001, 0.005, 0.01, 0.03, 0.05, 0.1})
- **Seeds**: 5 per configuration on Davis {7, 42, 123, 2024, 3407}; 10 on the KIBA scan (5 original + 5 extension {251, 1496, 5934, 7350, 9889})
- **Learning rate**: 5×10⁻⁴ for both Davis and KIBA (an earlier KIBA batch at 1×10⁻⁴ was excluded for protocol contamination)

## Data Availability

- **Davis dataset**: Original data from [Davis et al. 2011](https://doi.org/10.1021/ci200204d)
- **KIBA dataset**: Original data from [Tang et al. 2014](https://doi.org/10.1093/bioinformatics/btu070)
- **Processed data**: Available in `data/processed/` after running preprocessing scripts

## Citation

```bibtex
@article{ren2026contrastive,
  title={Contrastive Learning for Drug-Target Affinity Prediction: A Systematic Empirical Benchmark},
  author={Ren, Xiaoping and Sui, Chaoya and Liu, Xiaoyong and Ren, Xiaoli and Tao, Ling and Zhang, Yan},
  journal={Bioinformatics},
  year={2026},
  doi={[TO_BE_ADDED]}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Questions?

For reproducibility issues or questions about the audit process, please contact:
- **Xiaoping Ren**: rxp613@gmail.com
- **GitHub Issues**: [Link to be added after repository creation]
