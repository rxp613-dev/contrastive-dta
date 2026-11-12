# Round 17 - KIBA Validation Set α Consistency Check

**Date**: 2026-09-02

**Total experiments analyzed**: 20

**Available α values**: [np.float64(0.01), np.float64(0.1), np.float64(0.5)]

**Seeds**: [np.int64(7), np.int64(13), np.int64(42), np.int64(101), np.int64(123), np.int64(555), np.int64(999), np.int64(2024), np.int64(2026), np.int64(3407)]

## Method

1. Scan all available KIBA experiments to identify which α values exist
2. For each α and seed, extract best validation MSE and test MSE
3. Identify validation-set and test-set optimal α
4. Compare and assess consistency

## Test Set Results by α

| α | Test MSE (mean±std) | n seeds |
|---|---------------------|---------|
| 0.010 | 0.3810 ± 0.0505 | 5 |
| 0.100 | 0.3963 ± 0.0356 | 5 |
| 0.500 | 0.3579 ± 0.0345 | 10 |

**Test-set optimal α**: 0.500 (MSE = 0.3579)

## Validation Set Results by α

| α | Best Val MSE (mean±std) | n seeds |
|---|-------------------------|---------|
| 0.010 | 0.3995 ± 0.0561 | 5 |
| 0.100 | 0.4148 ± 0.0368 | 5 |
| 0.500 | 0.3744 ± 0.0389 | 10 |

**Validation-set optimal α**: 0.500 (MSE = 0.3744)

## Per-seed Comparison

| Seed | Test Optimal α | Test MSE | Val Optimal α | Val MSE | Match |
|------|----------------|----------|---------------|---------|-------|
| 7 | 0.500 | 0.3472 | 0.500 | 0.3677 | ✓ |
| 13 | 0.500 | 0.3736 | 0.500 | 0.3935 | ✓ |
| 42 | 0.010 | 0.3592 | 0.010 | 0.3733 | ✓ |
| 101 | 0.500 | 0.3466 | 0.500 | 0.3618 | ✓ |
| 123 | 0.500 | 0.2936 | 0.500 | 0.3007 | ✓ |
| 555 | 0.500 | 0.3910 | 0.500 | 0.4134 | ✓ |
| 999 | 0.500 | 0.3156 | 0.500 | 0.3255 | ✓ |
| 2024 | 0.100 | 0.3987 | 0.500 | 0.4172 | ✗ |
| 2026 | 0.500 | 0.3840 | 0.500 | 0.4018 | ✓ |
| 3407 | 0.500 | 0.3449 | 0.500 | 0.3589 | ✓ |

## Consistency Summary

- **Test-set optimal α**: 0.500
- **Validation-set optimal α**: 0.500
- **Per-seed agreement**: 9/10 seeds (90.0%)

**Conclusion**: ✓ Test-set and validation-set agree on optimal α.
This suggests α selection is robust and not driven by test-set overfitting.
