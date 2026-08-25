from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from common import CHANNEL_ORDER, sha256_file, write_json
from dataset import load_preprocessed_partition
from validate_onnx import predict_onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package held-out wearable tensors for Android")
    parser.add_argument("--device-data-root", required=True)
    parser.add_argument("--model", required=True, help="validated static 8-bit ONNX model")
    parser.add_argument("--deployment-seed", type=int, required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--assets-dir",
        default="android_mobile_benchmark/app/src/main/assets",
    )
    parser.add_argument(
        "--benchmark-assets-dir",
        default="android_mobile_benchmark/benchmark/src/androidTest/assets",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assets = Path(args.assets_dir).resolve()
    assets.mkdir(parents=True, exist_ok=True)
    inputs, labels, sample_ids = load_preprocessed_partition(args.device_data_root, "test")
    predictions = predict_onnx(args.model, inputs)

    input_path = assets / "test_inputs.bin"
    inputs.astype("<f4", copy=False).tofile(input_path)
    model_path = assets / "model_8bit.onnx"
    shutil.copy2(args.model, model_path)
    source_metadata_path = Path(args.model).with_suffix(".metadata.json")
    source_metadata = (
        json.loads(source_metadata_path.read_text(encoding="utf-8"))
        if source_metadata_path.is_file()
        else {}
    )

    samples = [
        {"index": index, "sample_id": sample_id, "target_ml": float(label), "desktop_8bit_ml": float(prediction)}
        for index, (sample_id, label, prediction) in enumerate(zip(sample_ids, labels, predictions))
    ]
    manifest = {
        "format_version": 2,
        "dtype": "float32-little-endian",
        "shape": list(inputs.shape),
        "sample_shape": [7, 400],
        "channel_order": list(CHANNEL_ORDER),
        "partition": "held-out wearable test",
        "deployment_seed": args.deployment_seed,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "model_format": source_metadata.get("format", "ONNX static 8-bit"),
        "activation_type": source_metadata.get("activation_type", "unknown"),
        "weight_type": source_metadata.get("weight_type", "unknown"),
        "model_sha256": sha256_file(model_path),
        "inputs_sha256": sha256_file(input_path),
        "samples": samples,
    }
    write_json(assets / "test_manifest.json", manifest)
    benchmark_assets = Path(args.benchmark_assets_dir).resolve()
    benchmark_assets.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, benchmark_assets / model_path.name)
    shutil.copy2(input_path, benchmark_assets / input_path.name)
    shutil.copy2(assets / "test_manifest.json", benchmark_assets / "test_manifest.json")
    print(f"model: {model_path} ({model_path.stat().st_size / 1_000_000:.3f} MB)")
    print(f"inputs: {input_path} ({len(inputs)} x 7 x 400)")
    print(f"manifest: {assets / 'test_manifest.json'}")
    print(f"benchmark assets: {benchmark_assets}")


if __name__ == "__main__":
    main()
