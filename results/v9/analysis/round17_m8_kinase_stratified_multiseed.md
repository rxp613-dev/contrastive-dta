# Round 17 - Multi-seed Kinase/Non-kinase Stratification Analysis

**Date**: 2026-09-02

**Seeds**: 5 (7, 42, 123, 2024, 3407)

## Method

1. Load test_predictions.json from all 5 seeds for E0 and E7
2. For each seed, classify targets as kinase/non-kinase and compute per-category MSE and CI
3. Aggregate across seeds to get mean±std for each category
4. This provides real multi-seed statistics instead of single-seed estimates

## E0 Performance (mean±std over 5 seeds)

| Target Class | n | MSE | CI |
|--------------|---|-----|----|
| kinase | 727 | 0.2275±0.0274 | 0.7994±0.0066 |
| non-kinase | 2279 | 0.2589±0.0090 | 0.7677±0.0010 |

## E7 Performance (mean±std over 5 seeds)

| Target Class | n | MSE | CI |
|--------------|---|-----|----|
| kinase | 727 | 0.1939±0.0102 | 0.8055±0.0047 |
| non-kinase | 2279 | 0.2257±0.0037 | 0.7723±0.0023 |

## Improvement (E7 vs E0)

| Target Class | n | MSE Improvement | CI Improvement |
|--------------|---|-----------------|----------------|
| kinase | 727 | 14.76% | +0.0060 |
| non-kinase | 2279 | 12.84% | +0.0046 |

## Key Findings

- Kinase targets: +14.76% improvement
- Non-kinase targets: +12.84% improvement
- **Contrastive learning benefits kinase targets more** (+14.76% vs +12.84%)
- Kinase families have conserved structural domains, and contrastive learning may help the model learn kinase family共性特征

## LaTeX Table

```latex
\begin{table}[!t]
\caption{Performance stratified by target class (mean$\pm$std over 5 seeds).}
\label{tab:kinase_stratification}
\centering
\begin{tabular}{lcccccc}
\toprule
Target Class & $n$ & E0 MSE & E7 MSE & MSE Improvement & CI Improvement \\
\midrule
Kinase & 727 & 0.2275$\pm$0.0274 & 0.1939$\pm$0.0102 & \textbf{+14.76\%} & +0.0060 \\
Non-kinase & 2279 & 0.2589$\pm$0.0090 & 0.2257$\pm$0.0037 & \textbf{+12.84\%} & +0.0046 \\
\bottomrule
\end{tabular}
\end{table}
```