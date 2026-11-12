# Post-hoc Power Analysis

**Date**: 2026-09-02

**Method**: Calculated post-hoc statistical power using observed effect sizes (Cohen's d) and sample sizes from paired t-tests (two-sided, α=0.05).

## Results

| Comparison | Cohen's d | n (seeds) | Power | Interpretation |
|------------|-----------|-----------|-------|----------------|
| E7 vs E0 (GIN) | -3.439 | 5 | 1.000 | Strong |
| E5 vs E0 (Uniformity) | -2.978 | 5 | 0.997 | Strong |
| E1 vs E0 (InfoNCE) | -3.280 | 5 | 0.999 | Strong |
| GAT E7 vs E0 | -5.309 | 5 | 1.000 | Strong |
| KIBA E7 vs E0 | -2.097 | 10 | 1.000 | Strong |
| Cold-drug E7 vs E0 | -0.951 | 5 | 0.370 | Weak |
| Cold-target GAT E7 vs E0 (degradation) | 3.010 | 5 | 0.998 | Strong |

## Interpretation Guidelines

- **Strong**: Power > 0.80 (80% chance of detecting the effect if it exists)
- **Moderate**: Power 0.50-0.80
- **Weak**: Power < 0.50

## Key Findings

1. **E7 vs E0 (GIN)**: Strong power indicates the main result is well-powered
2. **GAT E7 vs E0**: Strong power confirms architecture comparison is reliable
3. **KIBA E7 vs E0**: Strong power with n=10 seeds validates cross-dataset generalization
4. **Cold-start results**: Varying power levels explain significance patterns
   - Cold-drug: Moderate/weak power suggests larger samples needed for definitive conclusion
   - Cold-target GAT: Strong power for degradation effect

## Recommendations

- For underpowered comparisons (power < 0.8), interpret non-significant results cautiously
- Consider these power estimates when designing follow-up studies
- Report power alongside effect sizes for transparency
