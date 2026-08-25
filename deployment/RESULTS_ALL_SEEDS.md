# Four-seed ONNX and quantization validation

Selection rule: Select one global format across all four seeds by lowest mean wearable-validation MAE; break an exact tie by highest mean validation R2, then lowest mean absolute prediction difference from FP32. Held-out test metrics are not loaded until after selection.

Global selected format: **U8U8 QDQ per-channel**.
The test partition was loaded only after this global selection.

## Aggregate validation selection

| Format | MAE (mL) | R2 | Mean abs. difference from FP32 (mL) |
|---|---:|---:|---:|
| U8U8 | 44.176 | 0.6692 | 6.313 |
| S8S8 | 47.430 | 0.6480 | 13.235 |

## Numerical parity and quantization deviation

| Partition | Comparison | Maximum difference (mL) | Mean absolute difference (mL) |
|---|---|---:|---:|
| validation | FP32 ONNX vs PyTorch | 0.000061 | 0.000016 |
| test | FP32 ONNX vs PyTorch | 0.000061 | 0.000016 |
| test | U8U8 vs FP32 ONNX | 30.994 | 6.578 |

## Seed-wise held-out test results

| Seed | Format | Size (MB) | MAE (mL) | R2 | Within 50 mL | 200 mL alarm |
|---:|---|---:|---:|---:|---:|---:|
| 1 | FP32 | 26.023 | 38.919 | 0.7092 | 70.00% | 86.67% |
| 1 | U8U8 | 6.652 | 40.751 | 0.6926 | 75.00% | 90.00% |
| 2 | FP32 | 26.023 | 40.369 | 0.7041 | 71.67% | 88.33% |
| 2 | U8U8 | 6.652 | 41.266 | 0.6963 | 66.67% | 90.00% |
| 3 | FP32 | 26.023 | 41.907 | 0.7064 | 68.33% | 91.67% |
| 3 | U8U8 | 6.652 | 43.240 | 0.6942 | 65.00% | 90.00% |
| 42 | FP32 | 26.023 | 41.348 | 0.6848 | 66.67% | 88.33% |
| 42 | U8U8 | 6.652 | 42.472 | 0.6727 | 68.33% | 88.33% |

## Mean +/- seed SD

| Partition | Format | MAE (mL) | R2 | Within 50 mL | 200 mL alarm | Size (MB) |
|---|---|---:|---:|---:|---:|---:|
| test | FP32 | 40.636 +/- 1.309 | 0.7011 +/- 0.0111 | 69.17 +/- 2.15% | 88.75 +/- 2.10% | 26.023 +/- 0.000 |
| test | U8U8 | 41.932 +/- 1.131 | 0.6890 +/- 0.0110 | 68.75 +/- 4.38% | 89.58 +/- 0.83% | 6.652 +/- 0.000 |
| validation | FP32 | 44.797 +/- 0.643 | 0.6703 +/- 0.0167 | 66.67 +/- 3.80% | 82.81 +/- 3.56% | 26.023 +/- 0.000 |
| validation | S8S8 | 47.430 +/- 1.649 | 0.6480 +/- 0.0181 | 59.38 +/- 1.20% | 85.94 +/- 5.74% | 6.652 +/- 0.000 |
| validation | U8U8 | 44.176 +/- 0.887 | 0.6692 +/- 0.0173 | 64.58 +/- 1.70% | 82.81 +/- 2.62% | 6.652 +/- 0.000 |

## Reproducibility

Execution log: `<SOURCE_ROOT>/deployment/artifacts/all_seeds_execution.log`
Each seed directory contains ONNX artifacts, metadata, hashes, and sample predictions. `RESULTS_ALL_SEEDS.csv` records every artifact SHA-256 and all requested seed-level metrics.
