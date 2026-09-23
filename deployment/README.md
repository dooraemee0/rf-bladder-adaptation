# ONNX and Android Deployment

This directory contains the released PyTorch-to-ONNX export, static
quantization, validation, and smartphone preprocessing workflow.

## Wearable RF input

Six physical traces are ordered `[S1, S2, S3, S4, H_left, H_right]`. The
virtual horizontal channel is the sample-wise mean of `S2` and `S3`, and the
model channel order is
`[H_left, H_virtual, H_right, S1, S2, S3, S4]`.

The wearable pipeline applies:

1. amplitude scaling by `1700`;
2. piecewise exponential time-gain compensation;
3. seventh-order Butterworth band-pass filtering at 1.5-4.5 MHz for 20 MSPS;
4. cropping to raw indices 300:3000;
5. virtual-channel construction;
6. Gaussian smoothing across neighboring channels within each orientation
   group (`sigma=1`);
7. 10x downsampling and retention of 220 samples;
8. independent standardization of each channel;
9. padding from 220 to 400 samples.

The Python reference is implemented in `dataset.py`. The corresponding
Android implementation is under
`../android_mobile_benchmark/pipeline/src/main/java/`.

## Export and quantization

Run commands from the repository root:

```bash
pip install -r deployment/requirements.txt

python deployment/export_onnx.py \
  --checkpoint checkpoints/final/der++/seed3/model.pt \
  --output deployment/artifacts/model_fp32.onnx

python deployment/quantize_onnx.py \
  --onnx deployment/artifacts/model_fp32.onnx \
  --output deployment/artifacts/model_u8u8_qdq.onnx \
  --device-data-root /path/to/wearable/data \
  --calibration-partition train \
  --per-channel
```

Export uses float32 input shape `(1, 7, 400)` and scalar output shape
`(1, 1)`. The released configuration is ONNX opset 17 and static U8U8 QDQ
with per-channel weight quantization.

Validate FP32 and quantized models:

```bash
python deployment/validate_onnx.py \
  --checkpoint checkpoints/final/der++/seed3/model.pt \
  --onnx deployment/artifacts/model_fp32.onnx \
  --device-data-root /path/to/wearable/data

python deployment/validate_quantized.py \
  --fp32 deployment/artifacts/model_fp32.onnx \
  --quantized deployment/artifacts/model_u8u8_qdq.onnx \
  --device-data-root /path/to/wearable/data
```

The four-seed workflow exports seeds 1, 2, 3, and 42, calibrates candidate
formats on wearable training data, selects the format from aggregate validation
MAE, and then evaluates the held-out partition:

```bash
python deployment/run_all_seeds_deployment.py \
  --device-data-root /path/to/wearable/data
```

Recorded seed-level results are in `RESULTS_ALL_SEEDS.csv`. The selected
format was U8U8 QDQ, and validation-only model selection chose DER++ seed 3.

## Android assets

Download `derpp_seed3_u8u8_qdq.onnx` and `mobile_test_inputs.bin` from the
GitHub release, then install verified copies:

```bash
python reproducibility/install_release_assets.py \
  --repo-root . \
  --onnx /path/to/derpp_seed3_u8u8_qdq.onnx \
  --test-inputs /path/to/mobile_test_inputs.bin
```

The expected ONNX SHA-256 is
`f473b9b8b4a745a9d7579896619af063d2e9cabfaad71fd699728837073a684e`,
and its size is 6,651,648 bytes.

## Verification records

`public_results/` contains the recorded Android outputs used for numerical
concordance, latency, initialization, and memory summaries. Verify them with:

```bash
python reproducibility/verify_recorded_android_results.py --repo-root .
```

The raw-RF latency boundary begins after six-channel RF has entered smartphone
memory. It includes smartphone RF preprocessing, ONNX inference, and scalar
retrieval, but excludes RF acquisition, BLE transfer, networking,
initialization, and interface rendering.
