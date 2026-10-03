# VascChangeFormer

Research code for six-month-ahead prediction of vascular change from longitudinal clinical information, with development in Centre A (323 patients) or Centre C (546 patients), and external validation in the other centres, including Centre B (31 patients).

This release corresponds to the six evaluation rows supplied with the latest manuscript results. **Raw datasets, patient identifiers, patient-level predictions, trained weights, and private fold assignments are not distributed.** Exact numerical reproduction requires the original authorized private inputs and fold plan.

## Reference results

Point estimates below are archived results, not results newly computed when this repository was prepared. Full precision values and patient-bootstrap 95% confidence intervals are in [the reference results](results/REFERENCE_RESULTS.md) and [machine-readable CSV](results/reference_metrics.csv).

| Development | Evaluation | Validation | n | ΔTBR MAE | ΔTBR RMSE | ΔTBR R² | Δlog-TAC MAE | Δlog-TAC RMSE | Δlog-TAC R² |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | A | Internal OOF | 323 | 0.2373 | 0.3600 | 0.2552 | 0.5078 | 1.0389 | −0.0071 |
| C | A | External | 323 | 0.2476 | 0.3716 | 0.2066 | 0.5015 | 1.0207 | 0.0279 |
| A | B | External | 31 | 0.2930 | 0.4399 | 0.0221 | 0.5048 | 1.0252 | 0.0154 |
| C | B | External | 31 | 0.2724 | 0.4221 | 0.0997 | 0.4856 | 1.0218 | 0.0220 |
| A | C | External | 546 | 0.2321 | 0.3170 | 0.2715 | 0.4856 | 0.9166 | 0.0133 |
| C | C | Internal OOF | 546 | 0.2400 | 0.3120 | 0.2944 | 0.5065 | 0.9282 | −0.0118 |

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

### Fixed workflow

1. Split the private A354 arrays into A323 and sealed B31; restrict the original frozen A354 outer fold plan to A323.
2. Train with patient-level outer five-fold and inner five-fold validation, with preprocessing and selection inside development partitions.
3. Aggregate complete A323 out-of-fold predictions. Lock the 90% PI residual quantiles before the full A323 refit.
4. Fit one final A323 model on all 323 development patients. Freeze external point predictions before evaluating B31 and C546 outcomes.
5. Independently develop the C546 model with nested validation, then refit on C546 and evaluate A323 and B31 externally.

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

- `scripts/reproduce.py`: ordered entry point for both development centres.
- `scripts/run_a323_*.py`: nested development, refit, and external validation.
- `scripts/run_c546_development_external_ab.py`: C546 development and A/B external validation.
- `src/lac_itransformer/`: original numerical implementation and workbook readers.
- `configs/`: model settings, comparator settings, and feature schema.
- `results/`: aggregate reference metrics only.
- `docs/source_provenance.json`: original source and aggregate-result checksums.

The public project is named **VascChangeFormer**. Legacy Python package names (`lac_itransformer`), model keys (`vascmtl`), class names, and internal `cac` fields are retained for compatibility with the experiment implementation. In the reported task, those internal `cac` fields refer to the TAC outcome; public result tables use TAC. Older modules are retained because the current trainer imports shared numerical components from them.

## 中文说明

本版本对应最新截图中的六组结果：A/C 中心内部 OOF，以及 A→B、A→C、C→A、C→B 外部验证。仓库只发布代码、配置、复现说明和汇总指标，不发布原始数据、患者编号、逐例预测或模型权重。

请将有权限使用的私有输入放入 `private_data/`，按上方命令运行。A323 最终模型实际训练 73 epochs，C546 为 69 epochs；99 是联合训练的上限。没有原始私有数据及冻结分折表时，公开代码无法单独重新产生截图中的数值。此次整理未重新训练全队列。
