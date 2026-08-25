# RFNet ONNX and Android deployment

This directory implements a model-only deployment path for the adopted DER++
model. It does not modify the manuscript and it does not treat desktop timing as
smartphone timing.

See `AUDIT.md` for the initial checkpoint and legacy-latency audit,
`PREPROCESSING_RECONCILIATION.md` for the axis-0/axis-1 investigation,
`RESULTS_ALL_SEEDS.md` for the four-seed export and quantization results, and
`DEPLOYMENT_MODEL_SELECTION.md` for validation-only selection of the physical
deployment checkpoint. The completed public Android snapshot and its reporting
boundaries are documented in `PUBLIC_ANDROID_README.md`.

The controlled wearable-envelope experiment is reported in
`ENVELOPE_ABLATION_RESULTS.md` and `ENVELOPE_ABLATION_DECISION.md`, with its
execution record in `ENVELOPE_ABLATION_REPRODUCIBILITY.md`. It selected the
existing phase-preserving wearable preprocessing, so the current ONNX and
Android assets remain tied to the selected pipeline.

## 1. Environment

```bash
conda activate torch310
pip install -r deployment/requirements.txt
```

Run all commands from the `release_bladder_derpp` root. The commands below use
one checkpoint only as a low-level example. The reproducible four-seed workflow
in Section 4 must be used for result reporting.

```bash
CKPT=checkpoints/final/der++/seed1/model.pt
DEVICE_ROOT=/absolute/path/to/data/device
```

## 2. Export and FP32 parity

```bash
python deployment/export_onnx.py \
  --checkpoint "$CKPT" \
  --output deployment/artifacts/rfnet_fp32.onnx

python deployment/validate_onnx.py \
  --checkpoint "$CKPT" \
  --onnx deployment/artifacts/rfnet_fp32.onnx \
  --device-data-root "$DEVICE_ROOT"
```

The export is fixed to batch 1, seven channels, and 400 samples. Quantization
must not proceed if held-out PyTorch/ONNX parity fails.

## 3. Static 8-bit quantization and held-out validation

```bash
python deployment/quantize_onnx.py \
  --onnx deployment/artifacts/rfnet_fp32.onnx \
  --output deployment/artifacts/rfnet_int8_qdq.onnx \
  --device-data-root "$DEVICE_ROOT" \
  --calibration-partition train \
  --per-channel

python deployment/validate_quantized.py \
  --fp32 deployment/artifacts/rfnet_fp32.onnx \
  --quantized deployment/artifacts/rfnet_int8_qdq.onnx \
  --device-data-root "$DEVICE_ROOT"
```

Calibration can use only `train` or `validation`; `test` is intentionally not a
CLI option. The default format is static QDQ S8S8 with per-channel weights,
which ONNX Runtime recommends as the first choice for CNN CPU inference.

If S8S8 loses too much validation accuracy, generate alternatives without
looking at the held-out test partition, then compare them on `validation`:

```bash
python deployment/evaluate_onnx_variants.py \
  --device-data-root "$DEVICE_ROOT" \
  --partition validation \
  --model fp32=deployment/artifacts/rfnet_fp32.onnx \
  --model s8s8=deployment/artifacts/rfnet_int8_qdq.onnx \
  --model u8u8=deployment/artifacts/rfnet_u8u8_qdq.onnx
```

Use the held-out test partition once for the validation-selected candidate. Do
not choose a quantization configuration by its test score.

## 4. Four-seed validation and global format selection

The complete workflow exports seeds 1, 2, 3, and 42, calibrates both 8-bit
candidates using only the wearable training partition, selects one global
format from aggregate validation MAE, and only then loads the held-out test
partition:

```bash
python deployment/run_all_seeds_deployment.py \
  --device-data-root "$DEVICE_ROOT"
```

The fixed validation rule selected U8U8 QDQ per-channel. Test results were not
used to choose the quantization format. See `RESULTS_ALL_SEEDS.csv` and
`RESULTS_ALL_SEEDS.md`; the execution log and per-seed hashes are under
`deployment/artifacts/`.

## 5. Select and package the Android checkpoint

The representative checkpoint was selected without held-out test metrics by
balancing wearable-validation and clinical-retention-validation R2. The rule
selected seed 3; details and all seed-wise validation values are in
`DEPLOYMENT_MODEL_SELECTION.md`.

```bash
python deployment/prepare_mobile_inputs.py \
  --device-data-root "$DEVICE_ROOT" \
  --model deployment/artifacts/seed3/model_u8u8_qdq.onnx \
  --deployment-seed 3 \
  --checkpoint checkpoints/final/der++/seed3/model.pt
```

This copies the validated model as `model_8bit.onnx` and writes every held-out wearable test tensor
as little-endian float32 binary. `test_manifest.json` stores sample IDs, labels,
desktop predictions, shape, channel order, deployment seed, checkpoint hash,
and artifact SHA-256 hashes.

After the Android app writes `android_predictions.csv`, compare it with desktop
ONNX Runtime:

```bash
python deployment/compare_android_outputs.py \
  --manifest android_mobile_benchmark/app/src/main/assets/test_manifest.json \
  --android-csv /path/from/adb/android_predictions.csv
```

## Measurement boundaries and recorded device results

The model-only Android benchmark measures `OrtSession.run()` with batch size 1
and pre-created `(1, 7, 400)` tensors. Its recorded mean is `10.533 ms`. It
excludes model loading, raw RF preprocessing, BLE transfer, server
communication, and UI updates and must be reported as model-only latency.

Separate Jetpack Microbenchmarks start with six physical-channel RF traces
already resident in smartphone memory. Across three recorded sessions, the
median-of-session-medians was `14.277 ms` for preprocessing and `39.202 ms` for
preprocessing plus ONNX inference and scalar retrieval. The corresponding
paper-facing pooled summaries are `14.33 ms` and `39.23 ms`. These measurements
exclude live RF acquisition, BLE transfer, networking, and UI rendering and
must not be described as acquisition-to-readout or end-to-end latency.

Android and desktop predictions were identical for all 60 processed held-out
phantom inputs. The source JSON and CSV files are versioned under
`deployment/public_results/`; use
`reproducibility/verify_recorded_android_results.py` to check them. Raw phantom
RF files are not publicly distributed pending separate redistribution approval,
so repeating the raw-RF instrumentation tests requires authorized local assets.
