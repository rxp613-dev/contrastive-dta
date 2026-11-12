# Round 17 - Multi-seed Similarity Stratification Analysis

**Date**: 2026-09-02

**Seeds**: 5 (7, 42, 123, 2024, 3407)

## Method

1. Load test_predictions.json from all 5 seeds for E0 and E7
2. For each seed, compute per-quantile MSE and CI
3. Aggregate across seeds to get mean±std for each quantile
4. This provides real multi-seed statistics instead of single-seed estimates

## E0 Performance (mean±std over 5 seeds)

| Quantile | n | Avg Similarity | MSE | CI |
|----------|---|----------------|-----|----|
| Q0 | 752 | 0.135 | 0.2299±0.0094 | 0.7446±0.0020 |
| Q1 | 752 | 0.171 | 0.1922±0.0164 | 0.7420±0.0023 |
| Q2 | 750 | 0.244 | 0.2748±0.0136 | 0.7903±0.0038 |
| Q3 | 752 | 0.560 | 0.3084±0.0154 | 0.8090±0.0042 |

## E7 Performance (mean±std over 5 seeds)

| Quantile | n | Avg Similarity | MSE | CI |
|----------|---|----------------|-----|----|
| Q0 | 752 | 0.135 | 0.1960±0.0055 | 0.7508±0.0049 |
| Q1 | 752 | 0.171 | 0.1669±0.0085 | 0.7387±0.0025 |
| Q2 | 750 | 0.244 | 0.2489±0.0067 | 0.7952±0.0033 |
| Q3 | 752 | 0.560 | 0.2602±0.0060 | 0.8193±0.0033 |

## Improvement (E7 vs E0)

| Quantile | Avg Similarity | MSE Improvement | CI Improvement |
|----------|----------------|-----------------|----------------|
| Q0 | 0.135 | 14.76% | +0.0062 |
| Q1 | 0.171 | 13.16% | -0.0033 |
| Q2 | 0.244 | 9.41% | +0.0049 |
| Q3 | 0.560 | 15.62% | +0.0103 |

## Key Findings

- Q0 (lowest similarity): +14.76% improvement
- Q3 (highest similarity): +15.62% improvement
- Contrastive learning benefits high-similarity samples more (+15.62% vs +14.76%)

## LaTeX Table

```latex
\begin{table}[!t]
\caption{Performance stratified by combined similarity quantiles (mean$\pm$std over 5 seeds).}
\label{tab:similarity_stratification}
\centering
\begin{tabular}{lcccccc}
\toprule
Quantile & $n$ & Avg Similarity & E0 MSE & E7 MSE & MSE Improvement & CI Improvement \\
\midrule
Q0 (lowest) & 752 & 0.135 & 0.2299$\pm$0.0094 & 0.1960$\pm$0.0055 & \textbf{+14.76\%} & +0.0062 \\
Q1 & 752 & 0.171 & 0.1922$\pm$0.0164 & 0.1669$\pm$0.0085 & \textbf{+13.16\%} & -0.0033 \\
Q2 & 750 & 0.244 & 0.2748$\pm$0.0136 & 0.2489$\pm$0.0067 & \textbf{+9.41\%} & +0.0049 \\
Q3 (highest) & 752 & 0.560 & 0.3084$\pm$0.0154 & 0.2602$\pm$0.0060 & \textbf{+15.62\%} & +0.0103 \\
\bottomrule
\end{tabular}
\end{table}
```