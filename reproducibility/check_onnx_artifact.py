#!/usr/bin/env python3
"""Run twelve structural and runtime checks on the released U8U8 ONNX model."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto


EXPECTED_SHA256 = "f473b9b8b4a745a9d7579896619af063d2e9cabfaad71fd699728837073a684e"
EXPECTED_BYTES = 6_651_648


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    args = parser.parse_args()
    model_path = args.model.resolve()
    checks: list[tuple[str, bool]] = []

    checks.append(("01_file_exists", model_path.is_file()))
    checks.append(("02_file_size", model_path.stat().st_size == EXPECTED_BYTES))
    checks.append(("03_sha256", sha256(model_path) == EXPECTED_SHA256))
    model = onnx.load(model_path)
    checks.append(("04_protobuf_load", True))
    onnx.checker.check_model(model, full_check=True)
    checks.append(("05_onnx_full_checker", True))
    checks.append(("06_opset_17", any(item.version == 17 for item in model.opset_import)))

    inputs = list(model.graph.input)
    outputs = list(model.graph.output)
    input_shape = [dimension.dim_value for dimension in inputs[0].type.tensor_type.shape.dim]
    output_shape = [dimension.dim_value for dimension in outputs[0].type.tensor_type.shape.dim]
    checks.append(("07_input_name_and_count", len(inputs) == 1 and inputs[0].name == "rf"))
    checks.append(("08_input_float32", inputs[0].type.tensor_type.elem_type == TensorProto.FLOAT))
    checks.append(("09_input_shape_1x7x400", input_shape == [1, 7, 400]))
    checks.append(("10_output_contract", len(outputs) == 1 and outputs[0].name == "volume" and output_shape == [1, 1]))
    node_types = {node.op_type for node in model.graph.node}
    checks.append(("11_qdq_nodes_present", {"QuantizeLinear", "DequantizeLinear"} <= node_types))

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    prediction = session.run(["volume"], {"rf": np.zeros((1, 7, 400), np.float32)})[0]
    checks.append(("12_cpu_runtime_finite_scalar", prediction.shape == (1, 1) and np.isfinite(prediction).all()))

    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    passed_count = sum(passed for _, passed in checks)
    print(f"SUMMARY {passed_count}/{len(checks)}")
    if passed_count != 12:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
