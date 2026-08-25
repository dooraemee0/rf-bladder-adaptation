# Wearable envelope ablation decision

Selected from validation before held-out testing: **current_rf**.

Case 2: the phase-preserving wearable representation is consistently superior on the primary combined-validation criterion across all four seeds.

The held-out test metrics were not used to select preprocessing or alter hyperparameters. With only four paired seeds, the conclusion is based on effect magnitude and seed-wise direction rather than a small p-value.

## Recommendation

Retain the current domain-specific preprocessing: clinical inputs use a Hilbert envelope, while wearable inputs preserve the filtered RF waveform. The full Stage-2 adaptation comparison, ONNX quantization, and Android assets do not need to be regenerated for a common-envelope pipeline.

The observed clinical-to-wearable gap must be described as an acquisition-pipeline gap reflecting both hardware and domain-specific preprocessing. This ablation does not isolate hardware effects alone.

Common envelope reduced frozen Stage-1 feature MMD while worsening direct-transfer R2. MMD is therefore retained only as a descriptive auxiliary analysis, not as evidence that the continuous RF-to-volume relation transferred successfully.
