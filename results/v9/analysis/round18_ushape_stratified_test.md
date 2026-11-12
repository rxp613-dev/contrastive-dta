# Round 18 - U-shape Stratified Improvement Test

**Date**: 2026-09-03

**Seeds**: 5 (7, 42, 123, 2024, 3407)

**Objective**: Test falsifiable U-shape prediction of dual-component framework

## Method

1. Load per-seed test predictions for E0 and E7 (5 seeds each)
2. Compute fixed quantile boundaries from combined similarity
3. For each seed: compute per-quantile MSE for E0 and E7
4. Compute per-seed improvement: (E0_mse - E7_mse) / E0_mse × 100%
5. Paired t-test: Q0 vs Q2 and Q3 vs Q2 (n=5 seeds)

## Falsifiable Predictions

| Hypothesis | Predicted Pattern |
|------------|-------------------|
| Pure memorization | Monotonic increase: Q0 < Q1 < Q2 < Q3 |
| Pure transferable representation | Monotonic decrease: Q0 > Q1 > Q2 > Q3 |
| Dual-component (memorization + transfer) | U-shape: Q0/Q3 > Q1/Q2 |

## Per-seed Improvements

| Seed | Q0 | Q1 | Q2 | Q3 |
|------|----|----|----|-----|
| 7 | +17.68% | +4.09% | +9.38% | +14.88% |
| 42 | +17.40% | +9.02% | +10.03% | +14.55% |
| 123 | +11.48% | +16.73% | +8.46% | +14.69% |
| 2024 | +16.54% | +27.77% | +12.64% | +21.04% |
| 3407 | +10.28% | +5.12% | +6.14% | +12.36% |

## Aggregated Results (mean±std)

| Quantile | n | Avg Similarity | E0 MSE | E7 MSE | Improvement |
|----------|---|----------------|--------|--------|-------------|
| Q0 (lowest) | 752 | — | 0.2299±0.0094 | 0.1960±0.0055 | **+14.68%** ± 3.51% |
| Q1 | 752 | — | 0.1922±0.0164 | 0.1669±0.0085 | **+12.54%** ± 9.85% |
| Q2 | 750 | — | 0.2748±0.0136 | 0.2489±0.0067 | **+9.33%** ± 2.36% |
| Q3 (highest) | 752 | — | 0.3084±0.0154 | 0.2602±0.0060 | **+15.50%** ± 3.26% |

## U-shape Test Results

| Comparison | t-statistic | p-value | Significant? |
|------------|-------------|---------|--------------|
| Q0 vs Q2 | 5.128 | 0.0068 | Yes |
| Q3 vs Q2 | 9.678 | 0.0006 | Yes |

## Conclusion

**U-SHAPE CONFIRMED** (p < 0.05 for both Q0 > Q2 and Q3 > Q2)
- Pure memorization hypothesis: **REJECTED**
- Pure transfer hypothesis: **REJECTED**
- Dual-component framework: **CONSISTENT with U-shape prediction**
- The framework made a falsifiable prediction and the data supports it.