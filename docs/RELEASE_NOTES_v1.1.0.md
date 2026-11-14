# V1.1.0: Convergent-Protocol Reassessment (2026-11-15)

This release updates the repository to reflect the **V9 convergent-protocol (Config C) reassessment** findings submitted for Major Revision to Molecules.

## Key Changes

### Protocol Reassessment
- **Original finding**: Apparent gains of +26% to +62% (Config A, α∈{0.05,0.5})
- **Reassessment finding**: These gains reflected CL stabilizing a prematurely stopped baseline; under the convergent protocol (Config C, max_epochs=1000, ES patience=150), **CL does not produce net gain**
- **Cold-target**: CL may even harm cold-target generalization in some settings

### Evidence Package
- **300+ training runs** across 5 dimensions with rigorous statistical validation
- **115-directory audit** with config-level verification
- **Holm-corrected statistical analysis** (family-wise α=0.05)
- **Comprehensive supplementary materials**: S1–S8 covering α-sensitivity scans, regularization controls, batch provenance, mechanistic analyses, and protocol audits

### Dataset-Specific Findings
| Dataset | Encoder | Config | CL Effect | Significance |
|---------|---------|--------|-----------|--------------|
| KIBA | GIN | C (n=10) | −8.28% MSE | p=0.0011 (Holm) |
| Davis | GIN | C (n=10) | +3.35% MSE | p=0.11 (n.s.) |
| Davis | GAT | C + tuning | +9.19% MSE | p=0.0084 |
| Davis scaffold | GIN | C | −7.21% MSE | p=0.23 (n.s.) |
| KIBA cold-target | GIN | C | −3.35% MSE | p=0.11 (n.s.) |

### Documentation Updates
- `CITATION.cff`: Version 1.1.0, DOI 10.5281/zenodo.23308152
- `docs/manuscript_v9/`: Complete V9 manuscript source (14 pages, 6 figures, 6 tables)
- `docs/supplementary/`: S1–S8 supplementary materials
- `results/v9/`: Full analysis evidence (202 files, 171 per-run records)
- `V9_SYNC_MANIFEST.json`: Every staged file with SHA-256 checksum

## Citation

```bibtex
@misc{ren2026contrastive,
  author = {Ren, Xiaoping and Sui, Chaoya and Liu, Xiaoyong and Ren, Xiaoli and Tao, Ling and Zhang, Yan},
  title = {Contrastive Learning for Drug-Target Affinity Prediction: A Systematic Empirical Benchmark},
  year = {2026},
  publisher = {Zenodo},
  version = {v1.1.0},
  doi = {10.5281/zenodo.23308152},
  url = {https://doi.org/10.5281/zenodo.23308152}
}
```

## Previous Release

- **v1.0.0** (2026-10-12): Original Molecules submission version