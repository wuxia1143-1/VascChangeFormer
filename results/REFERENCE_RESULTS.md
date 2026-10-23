# Direct change reference results (2026-10-22)

Archived aggregate results computed from native predictions before endpoint reconstruction or nonnegative truncation. No model training was run for this code upload.

| Development | Evaluation | Validation | n | Delta TBR MAE | Delta TBR RMSE | Delta TBR R2 | Delta log-TAC MAE | Delta log-TAC RMSE | Delta log-TAC R2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | A | Internal OOF | 323 | 0.2373 [0.2085, 0.2677] | 0.3600 [0.2838, 0.4423] | 0.2552 [0.1244, 0.3954] | 0.5080 [0.4131, 0.6123] | 1.0393 [0.8125, 1.2449] | -0.0079 [-0.0592, 0.0574] |
| C | A | External | 323 | 0.2476 [0.2188, 0.2804] | 0.3716 [0.2945, 0.4546] | 0.2066 [0.0467, 0.3723] | 0.5030 [0.4117, 0.6026] | 1.0208 [0.8000, 1.2270] | 0.0277 [-0.0283, 0.1019] |
| A | B | External | 31 | 0.2930 [0.1956, 0.4262] | 0.4399 [0.2430, 0.6471] | 0.0221 [-0.9004, 0.6095] | 0.5048 [0.2428, 0.8433] | 1.0252 [0.2997, 1.5850] | 0.0154 [-0.1693, 0.4272] |
| C | B | External | 31 | 0.2724 [0.1734, 0.4008] | 0.4221 [0.2243, 0.6233] | 0.0997 [-0.6618, 0.6447] | 0.4856 [0.2254, 0.8266] | 1.0218 [0.2705, 1.5994] | 0.0220 [-0.1648, 0.5649] |
| A | C | External | 546 | 0.2321 [0.2134, 0.2494] | 0.3170 [0.2871, 0.3452] | 0.2715 [0.1625, 0.3721] | 0.4856 [0.4254, 0.5524] | 0.9166 [0.7667, 1.0691] | 0.0133 [-0.0244, 0.0584] |
| C | C | Internal OOF | 546 | 0.2400 [0.2230, 0.2562] | 0.3120 [0.2868, 0.3360] | 0.2944 [0.2143, 0.3657] | 0.5066 [0.4458, 0.5748] | 0.9283 [0.7814, 1.0812] | -0.0121 [-0.0672, 0.0448] |

95% percentile intervals use 2,000 patient-level bootstrap resamples, seed 2026, with frozen predictions and no retraining.

The previous endpoint-derived reference is preserved in `legacy_endpoint_derived/`. The complete comparator and ablation tables are in `direct_change/`.
