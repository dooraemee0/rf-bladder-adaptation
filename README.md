# Stage 2: Clinical-to-Wearable Adaptation and Android Deployment

This repository contains the Stage 2 implementation used in *From RF Echoes to
Bladder Volume: End-to-End Learning for Image-Free Wearable Ultrasound
Monitoring*. It adapts the clinically pretrained BladderRFNet to wearable
phantom RF acquisitions and contains the ONNX and Android deployment workflow.

Stage 1 is available at
[dooraemee0/rf-bladder](https://github.com/dooraemee0/rf-bladder).

## Repository map

```text
model.py, dataloader.py             model and RF preprocessing
ablation_common.py                  partitions, loaders, evaluation, selection
strategies.py                       adaptation methods and DER++ loss variants
ablation_A_adaptation.py            four-seed adaptation runner
ablation_scope.py, lambda_sweep.py  supplementary experiments
aggregate_summary.py                seed-level result aggregation
table3_*.yaml, table4_*.yaml        released experiment settings
scripts/                            training and evaluation commands
release/                            checkpoint release manifest
deployment/                         ONNX export, quantization, and verification
android_mobile_benchmark/           Android app and Jetpack Microbenchmark
reproducibility/                    artifact and recorded-output checks
```

The compared methods are device-only fine-tuning, rehearsal, EWC, SI, LwF,
CORAL, MMD, LoRA-conv, DER++, CFP-DER++, and R-DER++. The DER++ loss-term
comparison is configured by `table4_alpha0.yaml` and
`table4_beta0.yaml`.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `DATA_ROOT`, `CLINICAL_CKPT`, and `TEST_IDX_PATH` in `.env`. Clinical
and raw wearable RF data are not included.

## Adaptation

All runs use seeds 1, 2, 3, and 42 and validation-based checkpoint selection.

```bash
source .env
bash scripts/run_final_gpu0.sh
bash scripts/run_final_gpu1.sh
bash scripts/run_baselines_gpu0.sh
bash scripts/run_baselines_gpu1.sh
python aggregate_summary.py
```

DER++ loss-term comparison:

```bash
bash scripts/run_ab_gpu0.sh
bash scripts/run_ab_gpu1.sh
python summarize_ab_ablation.py
```

Evaluate the twelve released DER++, CFP-DER++, and R-DER++ checkpoints:

```bash
python scripts/verify_released_checkpoints.py \
  --zip rf-bladder-adaptation-checkpoints.zip
unzip rf-bladder-adaptation-checkpoints.zip
bash scripts/run_eval_invitro.sh
```

## Release assets

Release assets are available from
[`v1.0.0`](https://github.com/dooraemee0/rf-bladder-adaptation/releases/tag/v1.0.0):

| Asset | Purpose |
|---|---|
| `rf-bladder-adaptation-checkpoints.zip` | 12 checkpoints: 3 methods x 4 seeds |
| `derpp_seed3_u8u8_qdq.onnx` | selected Android deployment model |
| `mobile_test_inputs.bin` | processed 60-sample phantom validation tensor |

Expected sizes and SHA-256 values are stored in
`release/checkpoint_manifest.json` and
`reproducibility/RELEASE_ARTIFACTS.json`. Large models are release assets and
are not duplicated in the Git tree.

## ONNX deployment

The released deployment uses DER++ seed 3, ONNX opset 17, and static U8U8 QDQ
quantization with per-channel weights. Calibration uses wearable training data;
format and seed selection use validation data.

```bash
python deployment/export_onnx.py --help
python deployment/quantize_onnx.py --help
python deployment/validate_onnx.py --help
python deployment/validate_quantized.py --help
```

The complete commands, input contract, and selection workflow are in
[`deployment/README.md`](deployment/README.md).

## Android

Install the released model and processed tensor:

```bash
python reproducibility/install_release_assets.py \
  --repo-root . \
  --onnx /path/to/derpp_seed3_u8u8_qdq.onnx \
  --test-inputs /path/to/mobile_test_inputs.bin
```

Build with JDK 17 and Android SDK 36:

```bash
cd android_mobile_benchmark
./gradlew :pipeline:test
./gradlew :app:assembleBenchmark
./gradlew :benchmark:assembleReleaseAndroidTest
```

The Android implementation uses ONNX Runtime Android 1.23.2, CPU execution,
batch size 1, and one intra-op and inter-op thread. The Jetpack benchmarks
separately measure model-only inference, RF preprocessing, and preprocessing
plus inference. See
[`android_mobile_benchmark/README.md`](android_mobile_benchmark/README.md).

## Recorded verification

```bash
python reproducibility/check_onnx_artifact.py \
  /path/to/derpp_seed3_u8u8_qdq.onnx
python reproducibility/verify_recorded_android_results.py --repo-root .
python -m unittest reproducibility/test_release.py
```

The versioned records contain Android-Python preprocessing parity, 60-sample
Android-desktop prediction concordance, physical-device latency JSON, and
memory point observations. The manuscript values are derived from these files:

- final preprocessing tensor maximum absolute difference:
  `2.771615982e-6`;
- Android-desktop prediction maximum difference: `0 mL`;
- raw-RF preprocessing pooled median: `14.33 ms`;
- preprocessing, ONNX inference, and scalar retrieval pooled median:
  `39.23 ms`.

The latency boundary starts with six-channel RF already resident in smartphone
memory. RF acquisition, BLE transfer, networking, initialization, and screen
rendering are excluded.

## Data availability

Clinical RF and raw wearable phantom RF recordings are not included. The
release contains model artifacts, a processed 60-sample phantom tensor, source
code, manifests, and recorded physical-device outputs.

## License

Code is released under the MIT License. Model and data artifacts remain subject
to the terms stated with the release.
