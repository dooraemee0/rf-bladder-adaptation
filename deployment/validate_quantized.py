from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from common import regression_metrics, runtime_metadata, sha256_file, write_json
from dataset import load_preprocessed_partition
from validate_onnx import predict_onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare FP32 and static 8-bit ONNX on held-out wearable data")
    parser.add_argument("--fp32", required=True)
    parser.add_argument(
        "--quantized",
        "--int8",
        dest="quantized",
        required=True,
        help="validated static 8-bit ONNX model; --int8 is retained as a compatibility alias",
    )
    parser.add_argument("--device-data-root", required=True)
    parser.add_argument("--output-dir", default="deployment/artifacts/int8_validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs, labels, sample_ids = load_preprocessed_partition(args.device_data_root, "test")
    fp32_predictions = predict_onnx(args.fp32, inputs)
    quantized_predictions = predict_onnx(args.quantized, inputs)
    difference = quantized_predictions - fp32_predictions

    fp32_metrics = regression_metrics(labels, fp32_predictions)
    quantized_metrics = regression_metrics(labels, quantized_predictions)
    metric_delta = {
        key: quantized_metrics[key] - fp32_metrics[key]
        for key in fp32_metrics
        if key != "n"
    }
    result = {
        "partition": "held-out wearable test",
        "fp32": {
            "sha256": sha256_file(args.fp32),
            "actual_file_size_bytes": Path(args.fp32).stat().st_size,
            **fp32_metrics,
        },
        "quantized_8bit": {
            "sha256": sha256_file(args.quantized),
            "actual_file_size_bytes": Path(args.quantized).stat().st_size,
            **quantized_metrics,
        },
        "quantized_8bit_minus_fp32": metric_delta,
        "prediction_difference": {
            "max_absolute_ml": float(np.max(np.abs(difference))),
            "mean_absolute_ml": float(np.mean(np.abs(difference))),
        },
        "runtime": runtime_metadata(),
    }
    write_json(output_dir / "metrics.json", result)

    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample_id", "target_ml", "onnx_fp32_ml", "onnx_8bit_ml", "quantized_8bit_minus_fp32_ml"))
        writer.writerows(zip(sample_ids, labels, fp32_predictions, quantized_predictions, difference))

    print(format_table(result))
    print(f"saved: {output_dir}")


def format_table(result: dict) -> str:
    header = "format       actual size   MAE (mL)    R2       +/-50 mL    alarm >=200 mL"
    lines = [header]
    for key, name in (("fp32", "ONNX FP32"), ("quantized_8bit", "ONNX 8-bit")):
        row = result[key]
        lines.append(
            f"{name:<12} {row['actual_file_size_bytes'] / 1_000_000:>8.3f} MB  "
            f"{row['mae_ml']:>8.3f}    {row['r2']:>6.3f}    "
            f"{row['accuracy_within_50ml_pct']:>7.2f}%    {row['alarm_200ml_accuracy_pct']:>7.2f}%"
        )
    delta = result["quantized_8bit_minus_fp32"]
    lines.append(
        f"8-bit-FP32: MAE {delta['mae_ml']:+.3f} mL, R2 {delta['r2']:+.4f}, "
        f"+/-50 {delta['accuracy_within_50ml_pct']:+.2f} pp, "
        f"alarm {delta['alarm_200ml_accuracy_pct']:+.2f} pp"
    )
    lines.append(
        "prediction difference: "
        f"max {result['prediction_difference']['max_absolute_ml']:.4f} mL, "
        f"mean {result['prediction_difference']['mean_absolute_ml']:.4f} mL"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
