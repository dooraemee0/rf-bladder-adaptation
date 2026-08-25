# Android-Python preprocessing parity

Status: passed on the physical Galaxy S8+; preprocessing implementation frozen.

## Protocol

The predeclared wearable validation samples were `50_30.txt`, `150_30.txt`, and
`300_30.txt`, one from each calibration volume. No held-out raw RF sample was
used. Android was compared with the Python/SciPy golden reference at all ten
stages. Shape, dtype, non-finite counts, maximum and mean absolute difference,
RMSE, relative L2 error, and per-channel errors were recorded without selecting
a numerical tolerance in advance.

Device: Samsung Galaxy S8+ SM-G955N, Android 9/API 28. The instrumentation test
completed successfully (`OK (1 test)`). Battery temperature was 28.3 C before
and 28.5 C after the run.

## Stage-level results

The table aggregates the three equal-sized validation samples. `Max abs.` is
the maximum observed difference, `Mean abs.` is the mean of sample-level means,
and aggregate RMSE is the root mean square of sample-level RMSE values.

| Stage | Max abs. | Mean abs. | Aggregate RMSE |
|---|---:|---:|---:|
| Parsed six-channel raw | 0 | 0 | 0 |
| Amplitude scaling | 0 | 0 | 0 |
| TGC | 3.0518e-5 | 2.5917e-7 | 1.1936e-6 |
| Band-pass filtering | 1.5259e-5 | 1.9431e-7 | 6.3234e-7 |
| Clipping and cropping | 1.0915e-6 | 6.8284e-9 | 2.4751e-8 |
| Seven-channel construction | 1.0915e-6 | 6.8292e-9 | 2.4389e-8 |
| Gaussian smoothing | 6.2585e-7 | 6.5529e-9 | 1.7890e-8 |
| Downsampling | 6.3330e-8 | 4.4504e-9 | 9.0744e-9 |
| Standardization | 2.7716e-6 | 1.8411e-7 | 3.5070e-7 |
| Final padded `(7,400)` tensor | 2.7716e-6 | 1.1742e-7 | 2.6383e-7 |

Raw parsing and amplitude scaling were bitwise identical. The first numerical
difference appeared in exponential TGC evaluation and remained at float-level
magnitude through filtering and standardization. There were no shape or dtype
mismatches and no non-finite values.

## Final prediction concordance

| Sample | Target | Python U8U8 | Android U8U8 | Absolute difference |
|---|---:|---:|---:|---:|
| `50_30.txt` | 50 mL | 124.01664 mL | 124.01664 mL | 0 mL |
| `150_30.txt` | 150 mL | 171.71535 mL | 171.71535 mL | 0 mL |
| `300_30.txt` | 300 mL | 143.09612 mL | 143.09612 mL | 0 mL |

The observed final tensor differences did not change the U8U8 output for any
validation sample. Because the pipeline is structurally identical, finite, and
prediction-identical without parameter adjustment, the Android preprocessing
implementation was accepted and frozen before packaging held-out raw RF.

## Preserved evidence

- `raw/preprocessing_parity/preprocessing_parity_android.json`
- `raw/preprocessing_parity/logs/instrumentation_stdout.log`
- `raw/preprocessing_parity/logs/build_stdout.log`
- `raw/preprocessing_parity/apk_hashes.txt`
- uploaded archive SHA-256:
  `a895f7e986aa5933a14dc9149d89b00e3b5b2a08c993615e47110b5d89629b7b`
- Android JSON SHA-256:
  `e84d3295e8a7d722b6542b2dedaead444bbcb78a847e61bdd26039fe28094dc2`
