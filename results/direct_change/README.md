# Native change performance reference

Aggregate outputs of the completed 2026-10-22 direct-change analysis.
There are 180 continuous point estimates and 18 paired ablation comparisons.
All confidence intervals use 2,000 patient bootstrap draws, seed 2026.

The full model uses its saved TAC correction before endpoint reconstruction.
Comparators retain their own native outputs; no correction is added to them.
The direct-endpoint ablation subtracts baseline coordinates from its native
untruncated TBR/log-TAC outputs. Negative changes and negative R2 are retained.

Table 2 is A323 OOF; Table 3 is the A323 ablation analysis; Table 4 is C546
external validation; Table 5 is reciprocal development; Table S3 is B31
external validation. Full precision values are in
`all_continuous_metrics_95ci.csv` and `paired_ablation_bootstrap_95ci.csv`.
Legacy comparison columns in those files are retained for traceability.

Patient-level predictions, IDs, fold assignments, weights and private inputs
are excluded. See `docs/DIRECT_CHANGE_EVALUATION.md` for code and usage.
