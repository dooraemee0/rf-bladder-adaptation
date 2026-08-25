from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from common import (
    CHANNEL_ORDER,
    INPUT_NAME,
    INPUT_SHAPE,
    OUTPUT_NAME,
    load_deploy_model,
    runtime_metadata,
    sha256_file,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a DER++ RFNet checkpoint to FP32 ONNX")
    parser.add_argument("--checkpoint", required=True, help="exact DER++ state_dict checkpoint")
    parser.add_argument("--output", default="deployment/artifacts/rfnet_fp32.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--parity-tolerance", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    model = load_deploy_model(checkpoint)
    dummy = torch.zeros(INPUT_SHAPE, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes=None,
        dynamo=False,
    )

    graph = onnx.load(output)
    onnx.checker.check_model(graph)

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(42)
    max_abs_error = 0.0
    for _ in range(5):
        sample = rng.standard_normal(INPUT_SHAPE, dtype=np.float32)
        with torch.inference_mode():
            torch_output = model(torch.from_numpy(sample)).numpy()
        ort_output = session.run([OUTPUT_NAME], {INPUT_NAME: sample})[0]
        max_abs_error = max(max_abs_error, float(np.max(np.abs(torch_output - ort_output))))
    if max_abs_error > args.parity_tolerance:
        raise RuntimeError(
            f"ONNX parity failed: max absolute error {max_abs_error:.8g} "
            f"> tolerance {args.parity_tolerance:.8g}"
        )

    metadata = {
        "format": "FP32 ONNX",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "onnx": str(output),
        "onnx_sha256": sha256_file(output),
        "onnx_size_bytes": output.stat().st_size,
        "input_name": INPUT_NAME,
        "input_shape": list(INPUT_SHAPE),
        "input_dtype": "float32",
        "output_name": OUTPUT_NAME,
        "output_shape": [1, 1],
        "channel_order": list(CHANNEL_ORDER),
        "dynamic_axes": False,
        "opset": args.opset,
        "random_input_max_abs_error": max_abs_error,
        "runtime": runtime_metadata(),
    }
    metadata_path = output.with_suffix(".metadata.json")
    write_json(metadata_path, metadata)
    print(f"exported: {output}")
    print(f"size: {output.stat().st_size / 1_000_000:.3f} MB")
    print(f"random-input max abs error: {max_abs_error:.8g}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
