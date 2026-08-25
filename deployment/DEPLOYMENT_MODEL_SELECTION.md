# Representative deployment checkpoint selection

Selection rule: Maximize wearable validation R2 + clinical-retention validation R2. Break an exact tie by the lowest sum of wearable and clinical validation MAE. Held-out test predictions and metrics are not loaded.

The held-out test partition was not evaluated or used by this selection script.
The clinical test index was read only to reconstruct the original non-test train/validation pool.

| Seed | Wearable val R2 | Clinical val R2 | Combined R2 | Wearable val MAE | Clinical val MAE |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.666292 | 0.402280 | 1.068572 | 44.916 mL | 72.300 mL |
| 2 | 0.674606 | 0.401269 | 1.075875 | 44.062 mL | 72.017 mL |
| 3 | 0.690075 | 0.403947 | 1.094022 | 44.608 mL | 70.475 mL |
| 42 | 0.650068 | 0.443601 | 1.093668 | 45.603 mL | 68.776 mL |

Selected seed: **3**.
Selected checkpoint SHA-256: `ba9e8bd59b8eeb14ea09cbb2fbd613fad4490699064c25e8d882681d04ce2572`.

This choice applies only to the single artifact required for physical-device deployment. Manuscript accuracy remains a mean +/- seed SD result across all four seeds.
