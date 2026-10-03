# Private input contract

Keep all patient data outside version control. The workflow reads these local files:

| File under `private_data/` | Content |
|---|---|
| `A354_prepared_arrays.npz` | Original A354 pool before the identifier-only A323/B31 split |
| `A354_patient_outer_fold_plan.csv` | Original frozen patient assignments; columns `patient_id`, `test_fold` (integers 1–5) |
| `B31_ids.txt` | Exactly 31 unique B identifiers, one per line, supplied from the private cohort manifest |
| `C547_prepared_arrays.npz` | Centre-C source arrays before the single prespecified ineligible-patient exclusion |
| `C_excluded_id.txt` | Exactly one excluded C identifier, from the private cohort manifest |

Identifiers must be consistently encoded across arrays and manifests. Keep the original patient order and IDs in the private environment: replacing or reordering IDs can change fold construction or numerical training order. No actual identifiers or private fold assignments are shipped here.

## Array representation

All arrays in each NPZ share the patient dimension `N`. Save plain numeric or Unicode arrays using `numpy.savez_compressed`; object arrays are not accepted by the experiment loaders (`allow_pickle=False`). Do not put a `schema_json` object array inside these NPZ files; schema is supplied separately in `configs/schema.json`.

| Key | Shape for the 19/16 profile | Meaning |
|---|---|---|
| `patient_ids` | `(N,)` | Private Unicode identifiers |
| `static` | `(N, 19)` | Static covariates in schema order |
| `baseline` | `(N, 2)` | TBR and raw TAC at baseline |
| `targets` | `(N, 2)` | TBR and raw TAC at follow-up |
| `values` | `(N, 3, 16)` | Longitudinal measurements by temporal patch |
| `mask` | `(N, 3, 16)` | Observed-value indicators |
| `delta` | `(N, 3, 16)` | Measurement recency representation produced by the reader |
| `times` | `(N, 3)` | Patch positions |
| `treatments` | `(N, 3, 5)` | Treatment channels |
| `irregular_values`, `irregular_mask` | `(N, 48, 16)` | Irregular event representation and mask |
| `irregular_times` | `(N, 48)` | Irregular event times |
| `irregular_treatments` | `(N, 48, 5)` | Treatments at irregular events |

Retain all additional patient-aligned time metadata from the original preparation, including `followup_months`, `endpoint_elapsed_years`, and `prediction_cutoff_elapsed_years`. Do not standardize the entire cohort before cross-validation: fold preprocessors fit on each training partition. TAC baseline and targets remain in the original nonnegative scale; transforms are applied by the model and evaluation code.

A354 preparation also accepts the historical full 93-static/27-longitudinal arrays; the runner selects the first 19 static and 16 longitudinal features as in the reference run. This only applies when their column order matches the original preparation. C547 must already use the 19/16 profile. The schema file fixes the order; matching dimensions alone is insufficient.

## From original workbooks

Workbook adapters are included in `src/lac_itransformer/data/development_workbook.py` and `shandong_workbook.py`. They assume the original site-specific workbook layout. They are not general-purpose readers for arbitrary spreadsheets. The release workflow starts from the original prepared arrays so that cohort adjudication, exclusions, event ordering, and temporal cutoffs remain reproducible.

Any independent raw-data re-extraction must use the same eligibility definitions, the six-month prediction lead, and the original feature encoding. Do not use a reader's zero-month default for a six-month-ahead experiment. Verify the private preparation against the original archived inputs before comparing manuscript numbers. Local audits and generated manifests can contain private IDs and must stay in the ignored output directory.
