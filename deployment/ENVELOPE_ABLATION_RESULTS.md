# Wearable envelope ablation results

Condition A retains the phase-preserving wearable waveform; Condition B applies a time-axis Hilbert envelope to wearable H/S arrays. Both retain legacy Gaussian `axis=0`; all other preprocessing and Stage-2 DER++ settings are fixed.

The preprocessing condition was selected from aggregate validation only. Held-out test results were evaluated after that decision was recorded.

## Validation aggregate

| Condition | Wearable MAE (mL) | Wearable R2 | Within 50 mL | Alarm accuracy | Alarm F1 | Clinical MAE (mL) | Clinical R2 | Combined R2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Current RF | 44.796 +/- 0.642 | 0.6703 +/- 0.0167 | 66.67 +/- 3.80% | 82.81 +/- 3.56% | 0.6936 +/- 0.0778 | 70.892 +/- 1.623 | 0.4128 +/- 0.0206 | 1.0830 +/- 0.0129 |
| Common envelope | 62.701 +/- 1.396 | 0.3271 +/- 0.0128 | 52.08 +/- 3.80% | 79.69 +/- 1.04% | 0.5975 +/- 0.0283 | 87.109 +/- 0.460 | 0.1990 +/- 0.0389 | 0.5261 +/- 0.0362 |

## Test aggregate

| Condition | Wearable MAE (mL) | Wearable R2 | Within 50 mL | Alarm accuracy | Alarm F1 | Clinical MAE (mL) | Clinical R2 | Combined R2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Current RF | 40.634 +/- 1.309 | 0.7011 +/- 0.0111 | 69.17 +/- 2.15% | 88.75 +/- 2.10% | 0.8256 +/- 0.0341 | 61.314 +/- 2.570 | 0.6344 +/- 0.0254 | 1.3355 +/- 0.0167 |
| Common envelope | 58.613 +/- 1.441 | 0.4557 +/- 0.0264 | 52.50 +/- 2.15% | 85.42 +/- 1.60% | 0.7191 +/- 0.0398 | 76.212 +/- 4.646 | 0.3568 +/- 0.0429 | 0.8125 +/- 0.0299 |

## Paired seed differences

Values are Condition B minus Condition A. Negative MAE and positive R2 favor the common-envelope condition.

| Partition | Metric | Seed 1 | Seed 2 | Seed 3 | Seed 42 | Mean +/- SD | dz |
|---|---|---:|---:|---:|---:|---:|---:|
| validation | wearable_mae_ml | 16.835 | 17.330 | 18.643 | 18.811 | 17.905 +/- 0.973 | 18.397 |
| validation | wearable_r2 | -0.322 | -0.348 | -0.367 | -0.336 | -0.343 +/- 0.019 | -18.181 |
| validation | clinical_mae_ml | 14.953 | 15.544 | 16.677 | 17.694 | 16.217 +/- 1.217 | 13.324 |
| validation | clinical_r2 | -0.199 | -0.248 | -0.213 | -0.196 | -0.214 +/- 0.024 | -8.996 |
| validation | combined_r2 | -0.521 | -0.595 | -0.580 | -0.532 | -0.557 +/- 0.036 | -15.411 |
| test | wearable_mae_ml | 18.553 | 16.926 | 17.664 | 18.771 | 17.978 +/- 0.849 | 21.164 |
| test | wearable_r2 | -0.220 | -0.242 | -0.280 | -0.239 | -0.245 +/- 0.025 | -9.816 |
| test | clinical_mae_ml | 19.326 | 15.813 | 11.382 | 13.069 | 14.898 +/- 3.471 | 4.292 |
| test | clinical_r2 | -0.326 | -0.265 | -0.260 | -0.260 | -0.278 +/- 0.032 | -8.631 |
| test | combined_r2 | -0.546 | -0.507 | -0.540 | -0.500 | -0.523 +/- 0.023 | -22.443 |

## Frozen Stage-1 direct transfer

| Partition | Wearable representation | MAE (mL) | R2 | 50 mL mean prediction | 150 mL mean prediction | 300 mL mean prediction | Raw-input MMD2 | Feature MMD2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| validation | Current RF | 148.991 | -2.0268 | 311.44 | 308.04 | 297.75 | 0.2879 | 1.0265 |
| validation | Common envelope | 298.843 | -9.9616 | 500.40 | 496.70 | 399.43 | 0.2595 | 0.2679 |
| test | Current RF | 147.949 | -1.9502 | 306.82 | 307.42 | 311.79 | 0.2841 | 0.9596 |
| test | Common envelope | 293.671 | -9.5522 | 497.66 | 488.47 | 394.87 | 0.2869 | 0.2663 |

Common envelope reduced the frozen Stage-1 feature-space MMD, but direct-transfer R2 and volume calibration became substantially worse. Distribution proximity in this unsupervised feature metric therefore did not imply preservation of the clinical RF-to-volume mapping.

Biased RBF MMD2 with a pooled median-distance bandwidth was computed under this controlled split. The implementation that produced the manuscript's previous 0.373/0.647 values was not found, so these values are comparative within this ablation and are not treated as a reproduction of those numbers.
