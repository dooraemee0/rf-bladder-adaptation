from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from common import load_deploy_model, predict_torch, regression_metrics, runtime_metadata, sha256_file, write_json
from dataset import load_preprocessed_partition
from validate_onnx import predict_onnx


SEEDS = (1, 2, 3, 42)
QUANTIZED_FORMATS = ("s8s8", "u8u8")
SELECTION_RULE = (
    "Select one global format across all four seeds by lowest mean wearable-validation "
    "MAE; break an exact tie by highest mean validation R2, then lowest mean absolute "
    "prediction difference from FP32. Held-out test metrics are not loaded until after selection."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Four-seed ONNX and static 8-bit validation")
    parser.add_argument("--device-data-root", required=True)
    parser.add_argument("--checkpoint-root", default="checkpoints/final/der++")
    parser.add_argument("--artifact-root", default="deployment/artifacts")
    parser.add_argument("--results-csv", default="deployment/RESULTS_ALL_SEEDS.csv")
    parser.add_argument("--results-md", default="deployment/RESULTS_ALL_SEEDS.md")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_root = Path(args.artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    log_path = artifact_root / "all_seeds_execution.log"
    if log_path.exists() and not args.resume:
        raise FileExistsError(f"execution log already exists: {log_path}; use --resume")
    log_mode = "a" if args.resume else "w"

    checkpoint_root = Path(args.checkpoint_root).resolve()
    artifacts: dict[int, dict[str, Path]] = {}
    with log_path.open(log_mode, encoding="utf-8") as log:
        log.write(f"started_utc={datetime.now(timezone.utc).isoformat()}\n")
        log.write(f"command={' '.join(sys.argv)}\n")
        log.write(f"selection_rule={SELECTION_RULE}\n")
        log.write(f"runtime={json.dumps(runtime_metadata(), sort_keys=True)}\n")
        log.flush()
        for seed in SEEDS:
            seed_dir = artifact_root / f"seed{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            checkpoint = checkpoint_root / f"seed{seed}" / "model.pt"
            paths = {
                "checkpoint": checkpoint,
                "fp32": seed_dir / "model_fp32.onnx",
                "s8s8": seed_dir / "model_s8s8_qdq.onnx",
                "u8u8": seed_dir / "model_u8u8_qdq.onnx",
            }
            artifacts[seed] = paths
            ensure_generated(
                paths["fp32"],
                [
                    sys.executable,
                    "deployment/export_onnx.py",
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(paths["fp32"]),
                ],
                log,
                args.resume,
            )
            for name, quant_type in (("s8s8", "qint8"), ("u8u8", "quint8")):
                ensure_generated(
                    paths[name],
                    [
                        sys.executable,
                        "deployment/quantize_onnx.py",
                        "--onnx",
                        str(paths["fp32"]),
                        "--output",
                        str(paths[name]),
                        "--device-data-root",
                        str(Path(args.device_data_root).resolve()),
                        "--calibration-partition",
                        "train",
                        "--per-channel",
                        "--skip-optimization",
                        "--activation-type",
                        quant_type,
                        "--weight-type",
                        quant_type,
                    ],
                    log,
                    args.resume,
                )

    validation_inputs, validation_labels, validation_ids = load_preprocessed_partition(
        args.device_data_root, "validation", gaussian_axis=0
    )
    rows: list[dict] = []
    validation_predictions: dict[int, dict[str, np.ndarray]] = {}
    for seed in SEEDS:
        paths = artifacts[seed]
        model = load_deploy_model(paths["checkpoint"])
        pytorch_prediction = predict_torch(model, validation_inputs)
        predictions = {
            "fp32": predict_onnx(paths["fp32"], validation_inputs),
            "s8s8": predict_onnx(paths["s8s8"], validation_inputs),
            "u8u8": predict_onnx(paths["u8u8"], validation_inputs),
        }
        validation_predictions[seed] = predictions
        parity = np.abs(predictions["fp32"] - pytorch_prediction)
        if float(parity.max()) > 1e-4:
            raise RuntimeError(f"seed {seed} FP32 ONNX validation parity failed: {parity.max()}")
        for model_format in ("fp32", "s8s8", "u8u8"):
            rows.append(
                result_row(
                    seed,
                    "validation",
                    model_format,
                    paths[model_format],
                    validation_labels,
                    predictions[model_format],
                    pytorch_prediction,
                    predictions["fp32"],
                )
            )
        write_prediction_csv(
            paths["fp32"].parent / "validation_predictions.csv",
            validation_ids,
            validation_labels,
            pytorch_prediction,
            predictions,
        )

    selected_format, aggregate_validation = select_global_format(rows)
    print(f"global quantization selection: {selected_format}")
    print(SELECTION_RULE)

    # The held-out test partition is first loaded after global format selection.
    test_inputs, test_labels, test_ids = load_preprocessed_partition(
        args.device_data_root, "test", gaussian_axis=0
    )
    for seed in SEEDS:
        paths = artifacts[seed]
        model = load_deploy_model(paths["checkpoint"])
        pytorch_prediction = predict_torch(model, test_inputs)
        fp32_prediction = predict_onnx(paths["fp32"], test_inputs)
        selected_prediction = predict_onnx(paths[selected_format], test_inputs)
        parity = np.abs(fp32_prediction - pytorch_prediction)
        if float(parity.max()) > 1e-4:
            raise RuntimeError(f"seed {seed} FP32 ONNX test parity failed: {parity.max()}")
        predictions = {"fp32": fp32_prediction, selected_format: selected_prediction}
        for model_format in ("fp32", selected_format):
            rows.append(
                result_row(
                    seed,
                    "test",
                    model_format,
                    paths[model_format],
                    test_labels,
                    predictions[model_format],
                    pytorch_prediction,
                    fp32_prediction,
                )
            )
        write_prediction_csv(
            paths["fp32"].parent / "test_predictions_selected.csv",
            test_ids,
            test_labels,
            pytorch_prediction,
            predictions,
        )
        write_json(
            paths["fp32"].parent / "metrics.json",
            {
                "seed": seed,
                "selection_rule": SELECTION_RULE,
                "global_selected_format": selected_format,
                "validation": [row for row in rows if row["seed"] == seed and row["partition"] == "validation"],
                "test": [row for row in rows if row["seed"] == seed and row["partition"] == "test"],
                "runtime": runtime_metadata(),
            },
        )

    aggregate = aggregate_rows(rows)
    results_csv = Path(args.results_csv).resolve()
    results_md = Path(args.results_md).resolve()
    for output in (results_csv, results_md):
        if output.exists() and not args.resume:
            raise FileExistsError(f"result already exists: {output}; use --resume")
    write_rows_csv(results_csv, rows)
    results_md.write_text(
        render_markdown(rows, aggregate, aggregate_validation, selected_format, log_path),
        encoding="utf-8",
    )
    write_json(
        artifact_root / "all_seeds_summary.json",
        {
            "selection_rule": SELECTION_RULE,
            "global_selected_format": selected_format,
            "aggregate_validation_candidates": aggregate_validation,
            "aggregate_results": aggregate,
            "runtime": runtime_metadata(),
            "execution_log": str(log_path),
        },
    )
    print(f"saved: {results_csv}")
    print(f"saved: {results_md}")


def ensure_generated(path: Path, command: list[str], log, resume: bool) -> None:
    if path.exists():
        if not resume:
            raise FileExistsError(f"artifact already exists: {path}")
        log.write(f"resume_existing={path} sha256={sha256_file(path)}\n")
        log.flush()
        return
    log.write(f"run={' '.join(command)}\n")
    log.flush()
    completed = subprocess.run(command, text=True, capture_output=True)
    log.write(completed.stdout)
    log.write(completed.stderr)
    log.write(f"returncode={completed.returncode}\n")
    log.flush()
    print(completed.stdout, end="")
    if completed.returncode != 0:
        raise RuntimeError(f"command failed ({completed.returncode}): {' '.join(command)}")


def result_row(
    seed: int,
    partition: str,
    model_format: str,
    artifact: Path,
    labels: np.ndarray,
    prediction: np.ndarray,
    pytorch_prediction: np.ndarray,
    fp32_prediction: np.ndarray,
) -> dict:
    pytorch_difference = np.abs(prediction - pytorch_prediction)
    fp32_difference = np.abs(prediction - fp32_prediction)
    return {
        "seed": seed,
        "partition": partition,
        "format": model_format,
        "actual_file_size_bytes": artifact.stat().st_size,
        "artifact_sha256": sha256_file(artifact),
        **regression_metrics(labels, prediction),
        "max_abs_difference_vs_pytorch_ml": float(pytorch_difference.max()),
        "mean_abs_difference_vs_pytorch_ml": float(pytorch_difference.mean()),
        "max_abs_difference_vs_fp32_ml": float(fp32_difference.max()),
        "mean_abs_difference_vs_fp32_ml": float(fp32_difference.mean()),
    }


def select_global_format(rows: list[dict]) -> tuple[str, list[dict]]:
    aggregate = []
    for candidate in QUANTIZED_FORMATS:
        subset = [
            row
            for row in rows
            if row["partition"] == "validation" and row["format"] == candidate
        ]
        aggregate.append(
            {
                "format": candidate,
                "mean_validation_mae_ml": float(np.mean([row["mae_ml"] for row in subset])),
                "mean_validation_r2": float(np.mean([row["r2"] for row in subset])),
                "mean_abs_difference_vs_fp32_ml": float(
                    np.mean([row["mean_abs_difference_vs_fp32_ml"] for row in subset])
                ),
            }
        )
    aggregate.sort(
        key=lambda row: (
            row["mean_validation_mae_ml"],
            -row["mean_validation_r2"],
            row["mean_abs_difference_vs_fp32_ml"],
        )
    )
    return aggregate[0]["format"], aggregate


def aggregate_rows(rows: list[dict]) -> list[dict]:
    output = []
    metric_names = (
        "mae_ml",
        "rmse_ml",
        "r2",
        "accuracy_within_50ml_pct",
        "alarm_200ml_accuracy_pct",
        "max_abs_difference_vs_pytorch_ml",
        "mean_abs_difference_vs_pytorch_ml",
        "max_abs_difference_vs_fp32_ml",
        "mean_abs_difference_vs_fp32_ml",
        "actual_file_size_bytes",
    )
    keys = sorted({(row["partition"], row["format"]) for row in rows})
    for partition, model_format in keys:
        subset = [row for row in rows if row["partition"] == partition and row["format"] == model_format]
        summary = {"partition": partition, "format": model_format, "n_seeds": len(subset)}
        for metric in metric_names:
            values = np.asarray([row[metric] for row in subset], dtype=np.float64)
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        output.append(summary)
    return output


def write_prediction_csv(
    path: Path,
    sample_ids: list[str],
    labels: np.ndarray,
    pytorch_prediction: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> None:
    columns = ["sample_id", "target_ml", "pytorch_ml", *[f"{key}_ml" for key in predictions]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        for index, sample_id in enumerate(sample_ids):
            writer.writerow(
                [sample_id, labels[index], pytorch_prediction[index]]
                + [prediction[index] for prediction in predictions.values()]
            )


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render_markdown(
    rows: list[dict],
    aggregate: list[dict],
    aggregate_validation: list[dict],
    selected_format: str,
    log_path: Path,
) -> str:
    lines = [
        "# Four-seed ONNX and quantization validation",
        "",
        f"Selection rule: {SELECTION_RULE}",
        "",
        f"Global selected format: **{selected_format.upper()} QDQ per-channel**.",
        "The test partition was loaded only after this global selection.",
        "",
        "## Aggregate validation selection",
        "",
        "| Format | MAE (mL) | R2 | Mean abs. difference from FP32 (mL) |",
        "|---|---:|---:|---:|",
    ]
    for row in aggregate_validation:
        lines.append(
            f"| {row['format'].upper()} | {row['mean_validation_mae_ml']:.3f} | "
            f"{row['mean_validation_r2']:.4f} | {row['mean_abs_difference_vs_fp32_ml']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Numerical parity and quantization deviation",
            "",
            "| Partition | Comparison | Maximum difference (mL) | Mean absolute difference (mL) |",
            "|---|---|---:|---:|",
        ]
    )
    for partition in ("validation", "test"):
        fp32_rows = [
            row for row in rows if row["partition"] == partition and row["format"] == "fp32"
        ]
        lines.append(
            f"| {partition} | FP32 ONNX vs PyTorch | "
            f"{max(row['max_abs_difference_vs_pytorch_ml'] for row in fp32_rows):.6f} | "
            f"{np.mean([row['mean_abs_difference_vs_pytorch_ml'] for row in fp32_rows]):.6f} |"
        )
    selected_test_rows = [
        row for row in rows if row["partition"] == "test" and row["format"] == selected_format
    ]
    lines.append(
        f"| test | {selected_format.upper()} vs FP32 ONNX | "
        f"{max(row['max_abs_difference_vs_fp32_ml'] for row in selected_test_rows):.3f} | "
        f"{np.mean([row['mean_abs_difference_vs_fp32_ml'] for row in selected_test_rows]):.3f} |"
    )
    lines.extend(
        [
            "",
            "## Seed-wise held-out test results",
            "",
            "| Seed | Format | Size (MB) | MAE (mL) | R2 | Within 50 mL | 200 mL alarm |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        if row["partition"] != "test":
            continue
        lines.append(
            f"| {row['seed']} | {row['format'].upper()} | "
            f"{row['actual_file_size_bytes']/1_000_000:.3f} | {row['mae_ml']:.3f} | "
            f"{row['r2']:.4f} | {row['accuracy_within_50ml_pct']:.2f}% | "
            f"{row['alarm_200ml_accuracy_pct']:.2f}% |"
        )
    lines.extend(
        [
            "",
            "## Mean +/- seed SD",
            "",
            "| Partition | Format | MAE (mL) | R2 | Within 50 mL | 200 mL alarm | Size (MB) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in aggregate:
        lines.append(
            f"| {row['partition']} | {row['format'].upper()} | "
            f"{row['mae_ml_mean']:.3f} +/- {row['mae_ml_sd']:.3f} | "
            f"{row['r2_mean']:.4f} +/- {row['r2_sd']:.4f} | "
            f"{row['accuracy_within_50ml_pct_mean']:.2f} +/- {row['accuracy_within_50ml_pct_sd']:.2f}% | "
            f"{row['alarm_200ml_accuracy_pct_mean']:.2f} +/- {row['alarm_200ml_accuracy_pct_sd']:.2f}% | "
            f"{row['actual_file_size_bytes_mean']/1_000_000:.3f} +/- "
            f"{row['actual_file_size_bytes_sd']/1_000_000:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"Execution log: `{log_path}`",
            "Each seed directory contains ONNX artifacts, metadata, hashes, and sample predictions. "
            "`RESULTS_ALL_SEEDS.csv` records every artifact SHA-256 and all requested seed-level metrics.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
