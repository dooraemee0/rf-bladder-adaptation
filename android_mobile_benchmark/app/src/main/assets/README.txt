Generated deployment assets belong in this directory and are intentionally not
committed to Git:

- model_8bit.onnx
- test_inputs.bin
- test_manifest.json

Install the released ONNX model and processed validation tensor from the
repository root with:

python reproducibility/install_release_assets.py \
  --repo-root . \
  --onnx /path/to/derpp_seed3_u8u8_qdq.onnx \
  --test-inputs /path/to/mobile_test_inputs.bin

To regenerate the processed tensor from authorized wearable phantom data, use:

python deployment/prepare_mobile_inputs.py \
  --device-data-root /absolute/path/to/data/device \
  --model deployment/artifacts/rfnet_int8_qdq.onnx
