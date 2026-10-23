# Direct change evaluation (2026-10-22 results, 2026-10-23 code release)

The current performance tables compare final predicted changes directly with
observed changes. For VascChangeFormer, predictions are the two residuals
returned by the saved decision layer after its existing TAC GBDT correction.
The evaluator never reconstructs a follow-up endpoint, applies `expm1`, or
clips a predicted change. This matters when a negative TAC residual would
otherwise reconstruct to a negative follow-up log endpoint and be clipped.

For each target, MAE is `mean(abs(prediction - truth))`, RMSE is
`sqrt(mean((prediction - truth)**2))`, and R2 is
`1 - sum((prediction - truth)**2) / sum((truth - mean(truth))**2)`.
Constant observed changes produce an undefined (NaN) R2. Negative R2 values
and negative predicted changes are retained.

## Frozen output extraction

`scripts/paper_direct_extract.py` is adapted from the October 22 analysis.
It uses frozen preprocessors, weights, and decision layers only:

- Persistence exports zero changes.
- Elastic Net and XGBoost export their two fitted residual regressors.
- Neural comparators export `delta_tbr` and `delta_log_cac` heads.
- VascChangeFormer exports saved post-decision-layer residuals. Identity and
  central OOF paths prefer archived float32 native heads, avoiding small
  repeated-inference differences in paired comparisons.
- The direct-endpoint ablation exports native, untruncated TBR/log-TAC
  coordinates minus their respective baseline coordinates. It requires the
  original ablation source snapshot, which defines that parameterization;
  passing an endpoint checkpoint to the standard residual model is invalid.

Endpoint reconstruction is used only to check frozen inference against the
archived endpoint predictions. It does not supply the evaluated predictions.
External sealed labels are opened after this check and are replaced with
dummy targets during prediction. Failed extraction audits abort evaluation.
Existing extraction files are never silently reused or overwritten.

## Re-evaluate completed A323/C546 runs

From the public repository root, with authorized private run artifacts:

```bash
python scripts/reproduce_direct_changes.py --a-root runs/A323 --c-root runs/C546 --output runs/direct_changes --device cuda
```

Add `--models vascmtl persistence elastic_net xgboost apn_dr itransformer_mtl
first_icu_mtl learning_to_route` when all comparators have been trained.
`--source` selects a verified numerical source snapshot; the default is this
repository's `src/`. The default device is auto, with CPU and CUDA options.
Use the archived CUDA/Python environment for numerical reference comparison.
This command performs inference and evaluation, with no training or GBDT fit.

To include the manuscript ablations, also provide:

```bash
--ablation-root PRIVATE_ABLATION_RUN --ablation-source PRIVATE_ABLATION_SOURCE
--endpoint-ablation-root PRIVATE_ENDPOINT_RUN --endpoint-source PRIVATE_ENDPOINT_SOURCE
```

The run directories must retain the same `models/<model>/fold_1` ... `fold_5`
and pooled `out_of_fold_predictions.csv` structure as the archived experiments.
The wrapper supplies the original private A fold-plan order, C training-array
order, and external sealed-array order. It explicitly preserves the archived
A primary-model observed-change columns and B primary-model float32 baseline
log convention. These truth conventions affect tiny rounding differences.

The normal `scripts/reproduce.py` training workflow now ends by calling this
direct evaluation stage. Earlier stage CSVs remain endpoint-derived historical
diagnostics. Current tables are under `runs/direct_changes/metrics/`.
An A-only run writes `runs/direct_changes_a/`; running C afterwards writes the
combined A/C table to `runs/direct_changes/`.

## Evaluate existing private native-change exports

```bash
python scripts/evaluate_direct_changes.py --predictions private_exports/native_changes.csv --output runs/direct_metric_check
```

Required columns are `experiment`, `development`, `centre`, `model`,
`patient_id`, `true_delta_tbr`, `true_delta_log_tac`, `pred_delta_tbr`, and
`pred_delta_log_tac`. Experiments use `primary`, `reciprocal`, or `ablation`;
the full-model key is `vascmtl`. Each complete model/cohort group has 323,
31, or 546 unique patients for A, B, or C respectively. An optional unique
integer `patient_order` column sets the frozen evaluation order; otherwise
CSV row order is retained. All inputs and extraction/audit exports stay local.

Confidence intervals use 2,000 patient-level percentile bootstrap resamples
with seed 2026. The original index dtype is int32 for primary C evaluation
and int64 elsewhere. Full/ablation comparisons align patient IDs and reuse
the same draws. Positive `full_model_benefit` favours the full model:
ablation minus full for MAE/RMSE, full minus ablation for R2.

## Public results and scope

`results/direct_change/` contains 180 aggregate continuous estimates, their
confidence intervals, 18 paired ablation comparisons, and manuscript-ready
Tables 2/3/4/5/S3 from the completed October 22 analysis. The 36-row root
reference table selects the full model from the same source. Old reference
tables are retained under `results/legacy_endpoint_derived/`.

This update publishes continuous performance extraction and evaluation.
Probability mapping, DCA, risk strata, and patient-case figure workflows are
separate analyses; they are not refitted by these commands. Neural model
architecture, losses, weights, and GBDT training are unchanged.
Public aggregate results are archived measurements, not substitute predictions
used by reproduction commands. Exact private inputs, folds, source snapshots,
patient order, and runtime remain necessary for numerical reproduction.
