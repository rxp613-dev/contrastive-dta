# V9 Update (2026-11-12)

This repository has been extended for the manuscript version **V9**:
*Apparent Contrastive Learning Gains in Drug-Target Affinity Prediction Are
Manufactured by Under-Converged Supervised Baselines: A Convergent-Protocol
Reassessment*.

- `scripts/` now includes the V9 batch scripts (`v9_*.py`, `task_v9_*.py`).
- `results/v9/analysis/` holds the V9 statistical/audit evidence (JSON/MD).
- `results/v9/paper_tables/` holds the generated LaTeX table fragments.
- `results/v9/run_records/` holds the per-seed `results.json` /
  `config.json` evidence for every V9 run (no model weights).
- `docs/manuscript_v9/` is the V9 manuscript source.
- `docs/supplementary/` contains Supplementary File S3 (batch provenance).
- `results/v9/V9_SYNC_MANIFEST.json` lists every staged file with size and
  SHA-256, mapped to its workspace source.

The initial 2026-09 package is retained unchanged; its Config-B-era numbers
are superseded by the V9 convergent-protocol results documented above.
