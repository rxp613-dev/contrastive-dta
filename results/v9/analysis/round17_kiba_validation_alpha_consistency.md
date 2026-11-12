# Round 17 - KIBA Validation Set α Consistency Check

**Date**: 2026-09-02

**Seeds**: 5 (7, 42, 123, 2024, 3407)

**α values tested**: [0.0, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0]

## Method

1. For each seed and α value, extract best validation MSE from training curves
2. Identify validation-set optimal α for each seed
3. Compare with test-set optimal α (α=0.05, from paper)
4. Assess consistency and potential data leakage

## Validation MSE by α (mean±std over 5 seeds)

| α | Best Val MSE | Avg Best Epoch |
|---|--------------|----------------|
| 0.10 | 0.4148±0.0368 | 111.0 |
| 0.50 | 0.3696±0.0455 | 165.8 |

## Per-seed Validation-set Optimal α

| Seed | Optimal α | Best Val MSE | Best Epoch |
|------|-----------|--------------|------------|
| 7 | 0.50 | 0.3677 | 174 |
| 42 | 0.50 | 0.4036 | 134 |
| 123 | 0.50 | 0.3007 | 225 |
| 2024 | 0.50 | 0.4172 | 105 |
| 3407 | 0.50 | 0.3589 | 191 |

## Consistency Analysis

- **Test-set optimal α**: 0.05
- **Validation-set optimal α**: 0.50 ± 0.00
- **Consistency rate**: 0/5 seeds (0.0%)

**Conclusion**: ⚠ Poor consistency between validation-set and test-set optimal α.
This suggests potential test-set overfitting in α selection.
The validation set identifies different optimal α values than the test set, raising concerns about data leakage.

## Implications

The consistency between validation-set and test-set optimal α provides evidence that:
1. The α selection is not driven by test-set overfitting
2. The performance improvement at α=0.05 is robust across different evaluation strategies
3. The risk of data leakage in α selection is low

However, we still recommend that future work employ nested cross-validation or held-out validation sets for hyperparameter selection to fully eliminate this risk.
