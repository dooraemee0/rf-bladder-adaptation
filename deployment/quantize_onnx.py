from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

from common import INPUT_NAME, sha256_file, write_json
from dataset import load_preprocessed_partition, select_balanced_calibration


class ArrayCalibrationReader(CalibrationDataReader):
    def __init__(self, inputs: np.ndarray):
        self._inputs = inputs.astype(np.float32, copy=False)
        self._index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._index >= len(self._inputs):
            return None
        sample = self._inputs[self._index : self._index + 1]
        self._index += 1
        return {INPUT_NAME: sample}

    def rewind(self) -> None:
        self._index = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Static PTQ for the RFNet ONNX graph")
    parser.add_argument("--onnx", required=True, help="validated FP32 ONNX model")
    parser.add_argument("--output", default="deployment/artifacts/rfnet_int8_qdq.onnx")
    parser.add_argument("--device-data-root", required=True)
    parser.add_argument(
        "--calibration-partition",
        choices=("train", "validation"),
        default="train",
        help="test is intentionally unavailable",
    )
    parser.add_argument("--max-per-volume", type=int, default=None)
    parser.add_argument("--calibration-method", choices=("minmax", "entropy", "percentile"), default="minmax")
    parser.add_argument("--per-channel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--activation-type", choices=("qint8", "quint8"), default="qint8"
    )
    parser.add_argument("--weight-type", choices=("qint8", "quint8"), default="qint8")
    parser.add_argument(
        "--op-types",
        default="Conv,Gemm,MatMul",
        help="comma-separated ONNX operators to quantize",
    )
    parser.add_argument(
        "--exclude-node",
        action="append",
        default=[],
        help="exact ONNX node name to keep in FP32; may be repeated",
    )
    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="skip ORT graph optimization during preprocessing; shape inference is retained",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.onnx).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared = output.with_name(output.stem + ".preprocessed.onnx")
    temporary_output = output.with_name(output.stem + ".tmp.onnx")

    inputs, labels, _ = load_preprocessed_partition(
        args.device_data_root, args.calibration_partition
    )
    inputs, labels = select_balanced_calibration(inputs, labels, args.max_per_volume)
    counts = {str(int(volume)): int(np.sum(labels == volume)) for volume in (50, 150, 300)}
    if len(set(counts.values())) != 1:
        raise ValueError(f"calibration data are not balanced: {counts}")

    prepared.unlink(missing_ok=True)
    temporary_output.unlink(missing_ok=True)
    quant_pre_process(
        input_model=source,
        output_model_path=prepared,
        skip_optimization=args.skip_optimization,
        skip_onnx_shape=False,
        skip_symbolic_shape=True,
    )
    calibration_methods = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }
    quant_types = {"qint8": QuantType.QInt8, "quint8": QuantType.QUInt8}
    activation_type = quant_types[args.activation_type]
    weight_type = quant_types[args.weight_type]
    symmetric = args.activation_type == "qint8"
    quantize_static(
        model_input=prepared,
        model_output=temporary_output,
        calibration_data_reader=ArrayCalibrationReader(inputs),
        quant_format=QuantFormat.QDQ,
        activation_type=activation_type,
        weight_type=weight_type,
        per_channel=args.per_channel,
        calibrate_method=calibration_methods[args.calibration_method],
        op_types_to_quantize=[value.strip() for value in args.op_types.split(",") if value.strip()],
        nodes_to_exclude=args.exclude_node,
        extra_options={
            "ActivationSymmetric": symmetric,
            "WeightSymmetric": args.weight_type == "qint8",
        },
    )

    model = onnx.load(temporary_output)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(temporary_output), providers=["CPUExecutionProvider"])
    session.run(None, {INPUT_NAME: inputs[:1]})
    temporary_output.replace(output)

    node_counts: dict[str, int] = {}
    for node in model.graph.node:
        node_counts[node.op_type] = node_counts.get(node.op_type, 0) + 1
    metadata = {
        "format": f"ONNX static 8-bit QDQ {args.activation_type.upper()}/{args.weight_type.upper()}",
        "source": str(source),
        "source_sha256": sha256_file(source),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "actual_file_size_bytes": output.stat().st_size,
        "calibration_partition": args.calibration_partition,
        "calibration_counts": counts,
        "test_used_for_calibration": False,
        "calibration_method": args.calibration_method,
        "per_channel": args.per_channel,
        "activation_type": args.activation_type,
        "weight_type": args.weight_type,
        "op_types": [value.strip() for value in args.op_types.split(",") if value.strip()],
        "excluded_nodes": args.exclude_node,
        "preprocess_optimization_skipped": args.skip_optimization,
        "node_counts": node_counts,
        "onnxruntime": ort.__version__,
    }
    write_json(output.with_suffix(".metadata.json"), metadata)
    print(f"quantized: {output}")
    print(f"actual size: {output.stat().st_size / 1_000_000:.3f} MB")
    print(f"calibration: {args.calibration_partition} {counts}")
    print(f"node counts: {node_counts}")


if __name__ == "__main__":
    main()
