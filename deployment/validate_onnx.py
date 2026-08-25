from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import onnxruntime as ort

from common import (
    INPUT_NAME,
    OUTPUT_NAME,
    load_deploy_model,
    predict_torch,
    regression_metrics,
    runtime_metadata,
    sha256_file,
    write_json,
)
from dataset import load_preprocessed_partition


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate FP32 ONNX on held-out wearable data")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--device-data-root", required=True)
    parser.add_argument("--output-dir", default="deployment/artifacts/fp32_validation")
    parser.add_argument("--parity-tolerance", type=float, default=1e-4)
    return parser.parse_args()


def predict_onnx(model_path: str | Path, inputs: np.ndarray) -> np.ndarray:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    predictions = []
    for sample in inputs:
        output = session.run([OUTPUT_NAME], {INPUT_NAME: sample[None].astype(np.float32)})[0]
        predictions.append(float(np.asarray(output).reshape(-1)[0]))
    return np.asarray(predictions, dtype=np.float32)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    inputs, labels, sample_ids = load_preprocessed_partition(args.device_data_root, "test")
    torch_model = load_deploy_model(args.checkpoint)
    torch_predictions = predict_torch(torch_model, inputs)
    onnx_predictions = predict_onnx(args.onnx, inputs)
    absolute_difference = np.abs(torch_predictions - onnx_predictions)

    max_abs_difference = float(absolute_difference.max())
    if max_abs_difference > args.parity_tolerance:
        raise RuntimeError(
            f"held-out parity failed: max absolute difference {max_abs_difference:.8g} "
            f"> tolerance {args.parity_tolerance:.8g}"
        )

    result = {
        "partition": "held-out wearable test",
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "onnx_sha256": sha256_file(args.onnx),
        "pytorch": regression_metrics(labels, torch_predictions),
        "onnx_fp32": regression_metrics(labels, onnx_predictions),
        "parity": {
            "max_absolute_difference_ml": max_abs_difference,
            "mean_absolute_difference_ml": float(absolute_difference.mean()),
        },
        "runtime": runtime_metadata(),
    }
    write_json(output_dir / "metrics.json", result)

    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("sample_id", "target_ml", "pytorch_ml", "onnx_fp32_ml", "abs_diff_ml"))
        writer.writerows(
            zip(sample_ids, labels, torch_predictions, onnx_predictions, absolute_difference)
        )

    print(json_table(result))
    print(f"saved: {output_dir}")


def json_table(result: dict) -> str:
    rows = ["format       size       MAE (mL)    R2       +/-50 mL    alarm >=200 mL"]
    for name, metrics in (("PyTorch", result["pytorch"]), ("ONNX FP32", result["onnx_fp32"])):
        rows.append(
            f"{name:<12} {'-':<10} {metrics['mae_ml']:>8.3f}    {metrics['r2']:>6.3f}    "
            f"{metrics['accuracy_within_50ml_pct']:>7.2f}%    "
            f"{metrics['alarm_200ml_accuracy_pct']:>7.2f}%"
        )
    rows.append(
        f"parity max={result['parity']['max_absolute_difference_ml']:.8g} mL, "
        f"mean={result['parity']['mean_absolute_difference_ml']:.8g} mL"
    )
    return "\n".join(rows)


if __name__ == "__main__":
    main()
