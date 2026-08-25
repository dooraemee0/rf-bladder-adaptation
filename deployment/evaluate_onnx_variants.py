from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from common import regression_metrics, sha256_file, write_json
from dataset import load_preprocessed_partition
from validate_onnx import predict_onnx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ONNX quantization candidates")
    parser.add_argument("--device-data-root", required=True)
    parser.add_argument("--partition", choices=("validation", "test"), default="validation")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="repeat for each candidate; the first model is the FP32 reference",
    )
    parser.add_argument("--output-dir", default="deployment/artifacts/quantization_variants")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = []
    for value in args.model:
        if "=" not in value:
            raise ValueError(f"expected NAME=PATH, got {value}")
        name, path = value.split("=", 1)
        models.append((name, Path(path).resolve()))

    inputs, labels, _ = load_preprocessed_partition(args.device_data_root, args.partition)
    rows = []
    reference = None
    for name, path in models:
        prediction = predict_onnx(path, inputs)
        if reference is None:
            reference = prediction
        difference = np.abs(prediction - reference)
        rows.append(
            {
                "name": name,
                "path": str(path),
                "sha256": sha256_file(path),
                "actual_file_size_bytes": path.stat().st_size,
                **regression_metrics(labels, prediction),
                "max_absolute_difference_from_reference_ml": float(difference.max()),
                "mean_absolute_difference_from_reference_ml": float(difference.mean()),
            }
        )

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / f"{args.partition}_summary.json",
        {"partition": args.partition, "reference": models[0][0], "models": rows},
    )
    with (output_dir / f"{args.partition}_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("name,size_MB,MAE_mL,R2,within_50_pct,alarm_200_pct,max_diff_mL,mean_diff_mL")
    for row in rows:
        print(
            f"{row['name']},{row['actual_file_size_bytes']/1_000_000:.3f},"
            f"{row['mae_ml']:.3f},{row['r2']:.5f},"
            f"{row['accuracy_within_50ml_pct']:.2f},{row['alarm_200ml_accuracy_pct']:.2f},"
            f"{row['max_absolute_difference_from_reference_ml']:.3f},"
            f"{row['mean_absolute_difference_from_reference_ml']:.3f}"
        )
    print(f"saved: {output_dir}")


if __name__ == "__main__":
    main()
