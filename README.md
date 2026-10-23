# VascChangeFormer

Research code for six-month-ahead prediction of vascular change from longitudinal clinical information, with development in Centre A (323 patients) or Centre C (546 patients), and external validation in the other centres, including Centre B (31 patients).

Release 0.2.0 updates evaluation to the completed 2026-10-22 direct-change analysis. All continuous metrics compare native predicted changes with observed changes before endpoint reconstruction or nonnegative truncation. Main-model TAC predictions include the existing frozen GBDT correction. **Raw datasets, patient identifiers, patient-level predictions, trained weights, and private fold assignments are not distributed.** Exact numerical reproduction requires the original authorized private inputs and fold plan.

## Current direct change results

The full 180-metric comparator/ablation results and 18 paired comparisons are in [the direct-change results](results/direct_change/README.md). The old endpoint-derived reference is retained in `results/legacy_endpoint_derived/`.

Point estimates below are archived results, not results newly computed when this repository was prepared. Full precision values and patient-bootstrap 95% confidence intervals are in [the reference results](results/REFERENCE_RESULTS.md) and [machine-readable CSV](results/reference_metrics.csv).

| Development | Evaluation | Validation | n | Delta TBR MAE | Delta TBR RMSE | Delta TBR R2 | Delta log-TAC MAE | Delta log-TAC RMSE | Delta log-TAC R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | A | Internal OOF | 323 | 0.2373 | 0.3600 | 0.2552 | 0.5080 | 1.0393 | -0.0079 |
| C | A | External | 323 | 0.2476 | 0.3716 | 0.2066 | 0.5030 | 1.0208 | 0.0277 |
| A | B | External | 31 | 0.2930 | 0.4399 | 0.0221 | 0.5048 | 1.0252 | 0.0154 |
| C | B | External | 31 | 0.2724 | 0.4221 | 0.0997 | 0.4856 | 1.0218 | 0.0220 |
| A | C | External | 546 | 0.2321 | 0.3170 | 0.2715 | 0.4856 | 0.9166 | 0.0133 |
| C | C | Internal OOF | 546 | 0.2400 | 0.3120 | 0.2944 | 0.5066 | 0.9283 | -0.0121 |

The reported R² values concern **change**, not the follow-up endpoint. Δlog-TAC means `log1p(follow-up TAC) − log1p(baseline TAC)`.

## Install

Use Python 3.10 or newer in a dedicated environment. From the repository root:

```bash
python -m venv .venv
# Activate .venv with the command appropriate for your shell.
python -m pip install -r requirements-reference.txt -e .
```

`requirements-reference.txt` records five versions captured in the archived C546 runtime manifest. It is a partial historical record, not a complete environment lock. GPU execution also depends on the compatible PyTorch/CUDA installation and hardware. Installing with `python -m pip install -e .` uses broader dependency ranges for convenience; it does not guarantee identical numerical results. No full retraining was performed while packaging this release.

## Reproduce

Place authorized local inputs in `private_data/` according to [the data contract](docs/DATA_CONTRACT.md). The directory is ignored by Git. Then run:

```bash
python scripts/reproduce.py --development all --device cuda
```

Use `--device cpu` for a CPU-only machine. The default `auto` chooses an available device. The default workflow trains the manuscript model (`vascmtl`). To include the seven Centre-A comparison models:

```bash
python scripts/reproduce.py --development all --device cuda --with-baselines
```

For separate runs, execute `--development a` first, then `--development c` with the same output root. `--private-data` and `--output-root` accept custom local directories. Full nested cross-validation can take substantial time. The wrapper refuses to overwrite populated experiment roots; individual stage scripts support explicit continuation after inspecting existing artifacts.

### Direct evaluation of existing frozen runs

```bash
python scripts/reproduce_direct_changes.py --a-root runs/A323 --c-root runs/C546 --output runs/direct_changes --device cuda
```

This performs frozen inference and direct-change evaluation without retraining. For existing native-change CSV exports, use `scripts/evaluate_direct_changes.py`. See [input columns, comparator/ablation options and bootstrap conventions](docs/DIRECT_CHANGE_EVALUATION.md). The default full reproduction workflow also ends with this stage; current metrics are in `runs/direct_changes/metrics/`. Earlier per-stage metric files retain the historical endpoint-derived convention.

### Fixed workflow

1. Split the private A354 arrays into A323 and sealed B31; restrict the original frozen A354 outer fold plan to A323.
2. Train with patient-level outer five-fold and inner five-fold validation, with preprocessing and selection inside development partitions.
3. Aggregate complete A323 out-of-fold predictions. Lock the 90% PI residual quantiles before the full A323 refit.
4. Fit one final A323 model on all 323 development patients. Freeze external point predictions before evaluating B31 and C546 outcomes.
5. Independently develop the C546 model with nested validation, then refit on C546 and evaluate A323 and B31 externally.
6. Extract final native changes from the frozen models and compute continuous performance directly in change coordinates. Previously completed ablation runs can be supplied for paired differences. No endpoint reconstruction or clipping enters this final evaluation.

The reference external predictions come from a **single full-data refit**, not an ensemble of the five outer-fold models. Confidence intervals use 2,000 patient-level percentile bootstrap replicates (seed 2026), without retraining inside bootstrap.

### Training configuration

| Setting | Value |
|---|---|
| Input dimensions | static 19, longitudinal 16, baseline 2, treatment 5 |
| Temporal representation | 3 patches; up to 48 irregular events |
| Maximum joint training epochs | 99 |
| Early-stopping patience | 25 |
| Pretraining epochs | 0 |
| Reference A323 final refit | 73 joint epochs; seed 3232026 |
| Reference C546 final refit | 69 joint epochs; seed 35462026 |
| Reference encoded epoch values | A: 73001000; C: 69001000 |

Final refit epochs are selected from development OOF artifacts, not hardcoded to force the reference result. The encoded stage value `[73, 1, 0]` is interpreted by the joint trainer as **73 joint epochs**, not 74. Legacy `stage_tbr_epochs`/`stage_cac_epochs` fields do not replace the joint limit for this model. See [reproduction details](docs/REPRODUCIBILITY.md).

## Code organization

- `scripts/paper_direct_extract.py`: frozen native-change extraction for full models, comparators and ablations.
- `scripts/evaluate_direct_changes.py`: direct MAE/RMSE/R2 and paired patient bootstrap.
- `scripts/reproduce_direct_changes.py`: portable post-training evaluation wrapper.
- `scripts/reproduce.py`: ordered entry point for both development centres.
- `scripts/run_a323_*.py`: nested development, refit, and external validation.
- `scripts/run_c546_development_external_ab.py`: C546 development and A/B external validation.
- `src/lac_itransformer/`: original numerical implementation and workbook readers.
- `configs/`: model settings, comparator settings, and feature schema.
- `results/`: aggregate reference metrics only.
- `docs/source_provenance.json`: original source and aggregate-result checksums.

The public project is named **VascChangeFormer**. Legacy Python package names (`lac_itransformer`), model keys (`vascmtl`), class names, and internal `cac` fields are retained for compatibility with the experiment implementation. In the reported task, those internal `cac` fields refer to the TAC outcome; public result tables use TAC. Older modules are retained because the current trainer imports shared numerical components from them.

## 中文说明

0.2.0 版同步 2026-10-22 的直接变化量评价：使用最终 GBDT 校正后的 ΔTBR、Δlog(1+TAC) 直接计算连续性能，不经随访终点重建或非负截断。七个基线保留各自原生输出；直接终点消融以原始未截断的 TBR/log-TAC 坐标减基线。训练、权重与预处理不变。

公开表包含 180 项连续指标和 18 项配对消融比较；旧口径汇总值保留以便追溯。完整复现命令会调用新评价入口。精确重现仍需要有权限使用的私有原始输入、冻结分折和原运行环境。本次上传没有重新训练模型，不包含患者编号、逐例预测或模型权重。
