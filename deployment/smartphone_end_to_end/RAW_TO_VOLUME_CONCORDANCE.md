# Held-out Android raw-RF-to-volume concordance

Status: completed once after validation-only parity and implementation freeze.

## Evaluation control

The production Android implementation was frozen with aggregate SHA-256
`6785e1229fbf0b81a38158f9e2d7d37c4f681ca3bbb7a510b6838a8fb9bf9e1f`
before held-out raw RF was packaged. The selected DER++ seed-3 checkpoint,
static U8U8 QDQ ONNX model, preprocessing constants, channel order, split,
labels, and sample order were unchanged. The test used the existing 60-sample
held-out wearable manifest and was run once on the Galaxy S8+.

Before Android packaging, all 60 raw files were independently reconstructed by
the audited Python pipeline. Their final tensors were bitwise identical to the
existing frozen `test_inputs.bin`, and the rechecked desktop U8U8 predictions
were identical to the existing manifest predictions.

## Android-Python concordance

| Quantity | Result |
|---|---:|
| Samples | 60 |
| Missing / duplicate samples | 0 / 0 |
| Non-finite tensor / prediction values | 0 / 0 |
| Tensor maximum absolute difference | 2.6599e-6 |
| Tensor mean absolute difference | 8.2465e-8 |
| Tensor RMSE | 1.6356e-7 |
| Android-desktop prediction maximum difference | 0 mL |
| Android-desktop prediction mean difference | 0 mL |

Thus, float-level differences introduced by Android exponential and arithmetic
implementations did not change any U8U8 prediction across the held-out set.

## Raw-RF-to-volume performance

| Metric | Android result |
|---|---:|
| MAE | 43.2399 mL |
| RMSE | 56.8123 mL |
| R2 | 0.6942241 |
| Within 50 mL accuracy | 65.0% |
| 200 mL alarm accuracy | 90.0% |
| Alarm precision | 0.850 |
| Alarm recall | 0.850 |
| Alarm F1 | 0.850 |
| TN / FP / FN / TP | 37 / 3 / 3 / 17 |

The server independently recomputed all metrics from the returned Android CSV
and obtained the same values. These results match the prior model-only Android
evaluation exactly, demonstrating that smartphone-resident raw RF preprocessing
preserves the frozen deployment model's output and performance.

## Preserved evidence

- `raw/heldout_concordance/heldout_raw_rf_concordance_android.json`
- `raw/heldout_concordance/heldout_raw_rf_predictions_android.csv`
- `raw/heldout_concordance/logs/instrumentation_stdout.log`
- `raw/heldout_concordance/ONE_TIME_EVALUATION_COMPLETED.txt`
- returned archive SHA-256:
  `eab2d03d70f1f5f7cf2ceafa78f9ddbb65dd75126aaaf7d91320e6f5ae7d0332`
