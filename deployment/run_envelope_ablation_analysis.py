from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deployment.envelope_ablation_utils import (  # noqa: E402
    SEEDS,
    SELECTION_RULE,
    extended_metrics,
    git_state,
    per_volume_predictions,
    preprocessing_configuration,
    runtime_metadata,
    save_predictions,
    sha256_file,
    split_manifest,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the wearable envelope ablation")
    parser.add_argument("--phase", choices=("reproduce-a", "decide-and-test"), required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--test-index", required=True)
    parser.add_argument("--device-root", required=True)
    parser.add_argument("--clinical-excel", required=True)
    parser.add_argument("--clinical-rf-root", required=True)
    parser.add_argument("--released-checkpoint-root", required=True)
    parser.add_argument("--common-envelope-checkpoint-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument(
        "--condition-a-result",
        default=None,
        help="optional Condition A reproduction JSON when final outputs use another root",
    )
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    os.environ["ABL_DEVICE_BASE"] = str(Path(args.device_root).resolve())
    os.environ["ABL_SNUH_EXCEL"] = str(Path(args.clinical_excel).resolve())
    os.environ["ABL_SNUH_DATA"] = str(Path(args.clinical_rf_root).resolve())
    os.environ["ABL_DEV_VAL_RATIO"] = "0.2"
    os.environ["ABL_CLIN_VAL_RATIO"] = "0.2"
    os.environ["ABL_NUM_WORKERS"] = "0"
    os.environ["ABL_EVAL_NUM_WORKERS"] = "0"
    os.environ["ABL_PIN_MEMORY"] = "0"
    os.environ["ABL_USE_CACHE"] = "0"
    os.environ["ABL_WEARABLE_ENVELOPE"] = "0"


def load_data(args: argparse.Namespace):
    from ablation_common import load_all_data

    test_index = torch.load(args.test_index, map_location="cpu", weights_only=False)
    if hasattr(test_index, "tolist"):
        test_index = test_index.tolist()
    return load_all_data(list(test_index))


def aggregate(rows: list[dict], condition: str, partition: str) -> dict:
    subset = [
        row for row in rows if row["condition"] == condition and row["partition"] == partition
    ]
    output = {"condition": condition, "partition": partition, "n_seeds": len(subset)}
    metric_names = sorted(
        {
            key
            for row in subset
            for key, value in row.items()
            if isinstance(value, (int, float)) and key != "seed"
        }
    )
    for metric in metric_names:
        values = np.asarray([row[metric] for row in subset], dtype=np.float64)
        output[f"{metric}_mean"] = float(values.mean())
        output[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return output


def result_row(condition: str, seed: int, partition: str, wearable: dict, clinical: dict) -> dict:
    wearable_metrics = extended_metrics(wearable, wearable=True)
    clinical_metrics = extended_metrics(clinical, wearable=False)
    row = {"condition": condition, "seed": seed, "partition": partition}
    row.update({f"wearable_{key}": value for key, value in wearable_metrics.items() if key != "alarm_confusion"})
    row.update({f"clinical_{key}": value for key, value in clinical_metrics.items()})
    row["combined_r2"] = wearable_metrics["r2"] + clinical_metrics["r2"]
    row["combined_mae_ml"] = wearable_metrics["mae_ml"] + clinical_metrics["mae_ml"]
    return row


def reproduce_condition_a(args: argparse.Namespace) -> None:
    configure_environment(args)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    preselection = {
        "phase": "fixed_condition_a_reproduction_before_condition_b_training",
        "selection_rule_recorded_before_test_evaluation": SELECTION_RULE,
        "condition_a_test_is_reproduction_only_and_not_used_for_selection": True,
        "command": [sys.executable, *sys.argv],
        "runtime": runtime_metadata(),
        "git": git_state(PROJECT_ROOT),
        "preprocessing": preprocessing_configuration(False),
    }
    write_json(output_root / "condition_a_preselection.json", preselection)

    from ablation_common import evaluate, fresh_model

    data = load_data(args)
    manifest, manifest_hash = split_manifest(data)
    write_json(
        output_root / "dataset_split_manifest.json",
        {"dataset_split_sha256": manifest_hash, "splits": manifest},
    )
    rows = []
    checkpoint_root = Path(args.released_checkpoint_root).resolve()
    for seed in SEEDS:
        checkpoint = checkpoint_root / f"seed{seed}" / "model.pt"
        model = fresh_model()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        model.eval()
        seed_dir = output_root / "current_rf" / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_metrics = {
            "condition": "current_rf",
            "seed": seed,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "dataset_split_sha256": manifest_hash,
            "preprocessing": preprocessing_configuration(False),
            "validation": {},
            "test": {},
        }
        for partition, wearable_key, clinical_key in (
            ("validation", "dev_va", "clin_va"),
            ("test", "dev_te", "clin_te"),
        ):
            wearable = evaluate(model, data[wearable_key], wearable_envelope=False)
            clinical = evaluate(model, data[clinical_key], wearable_envelope=False)
            row = result_row("current_rf", seed, partition, wearable, clinical)
            rows.append(row)
            seed_metrics[partition] = row
            save_predictions(
                seed_dir / f"predictions_wearable_{partition}.csv",
                data[wearable_key],
                wearable,
            )
            save_predictions(
                seed_dir / f"predictions_clinical_{partition}.csv",
                data[clinical_key],
                clinical,
            )
        write_json(seed_dir / "metrics.json", seed_metrics)

    aggregates = [
        aggregate(rows, "current_rf", partition) for partition in ("validation", "test")
    ]
    wearable_test = next(row for row in aggregates if row["partition"] == "test")
    expected = {
        "wearable_mae_ml_mean": 40.636,
        "wearable_r2_mean": 0.7011,
        "wearable_accuracy_within_50ml_pct_mean": 69.17,
        "wearable_alarm_200ml_accuracy_pct_mean": 88.75,
    }
    differences = {
        key: float(wearable_test[key] - value) for key, value in expected.items()
    }
    reproduction_passed = all(
        abs(differences[key]) <= (0.01 if "r2" not in key else 0.0001)
        for key in differences
    )
    payload = {
        "condition": "current_rf",
        "condition_b_training_allowed": reproduction_passed,
        "selection_rule": SELECTION_RULE,
        "condition_a_test_used_for_selection": False,
        "dataset_split_sha256": manifest_hash,
        "rows": rows,
        "aggregates": aggregates,
        "expected_wearable_test": expected,
        "difference_from_expected": differences,
        "runtime": runtime_metadata(),
    }
    write_json(output_root / "condition_a_reproduction.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not reproduction_passed:
        raise RuntimeError("Condition A did not reproduce; do not start Condition B training")


def rbf_mmd2(source: np.ndarray, target: np.ndarray) -> dict:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    pooled = np.concatenate((source, target), axis=0)
    squared = np.sum((pooled[:, None] - pooled[None, :]) ** 2, axis=2)
    nonzero = squared[np.triu_indices_from(squared, k=1)]
    bandwidth_squared = float(np.median(nonzero[nonzero > 0]))
    kernel = np.exp(-squared / (2.0 * bandwidth_squared))
    n_source = len(source)
    k_xx = kernel[:n_source, :n_source]
    k_yy = kernel[n_source:, n_source:]
    k_xy = kernel[:n_source, n_source:]
    value = float(k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean())
    return {
        "mmd2_biased": value,
        "bandwidth_squared_median_heuristic": bandwidth_squared,
        "n_clinical": int(len(source)),
        "n_wearable": int(len(target)),
    }


@torch.no_grad()
def collect_input_and_features(model, dataframe, wearable_envelope: bool):
    from ablation_common import DEV, make_loader, pad_to_400, split_lists

    model.eval()
    inputs = []
    features = []
    loader = make_loader(
        dataframe, shuffle=False, bs=32, pad=True, wearable_envelope=wearable_envelope
    )
    for batch in loader:
        x7 = pad_to_400(torch.cat((batch["horizon"], batch["sagittal"]), dim=1)).to(DEV)
        rf_h, rf_s = split_lists(x7)
        channel_features = [
            model.h_encoders[index](rf_h[index]) for index in range(model.num_h_rf)
        ] + [model.s_encoders[index](rf_s[index]) for index in range(model.num_s_rf)]
        inputs.append(x7.cpu().numpy().reshape(len(x7), -1))
        features.append(torch.cat(channel_features, dim=1).cpu().numpy())
    return np.concatenate(inputs), np.concatenate(features)


def decide_and_test(args: argparse.Namespace) -> None:
    configure_environment(args)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    condition_a_path = (
        Path(args.condition_a_result).resolve()
        if args.condition_a_result
        else output_root / "condition_a_reproduction.json"
    )
    condition_a = json.loads(condition_a_path.read_text())
    rows = [row for row in condition_a["rows"] if row["partition"] == "validation"]

    common_root = Path(args.common_envelope_checkpoint_root).resolve()
    for seed in SEEDS:
        metrics_path = common_root / f"seed{seed}" / "validation_metrics.json"
        metrics = json.loads(metrics_path.read_text())
        wearable = metrics["wearable_validation"]
        clinical = metrics["clinical_retention_validation"]
        row = {"condition": "common_envelope", "seed": seed, "partition": "validation"}
        row.update({f"wearable_{key}": value for key, value in wearable.items() if key != "alarm_confusion"})
        row.update({f"clinical_{key}": value for key, value in clinical.items()})
        row["combined_r2"] = metrics["combined_validation_r2"]
        row["combined_mae_ml"] = metrics["combined_validation_mae_ml"]
        rows.append(row)

    validation_aggregates = [
        aggregate(rows, condition, "validation")
        for condition in ("current_rf", "common_envelope")
    ]
    ranked = sorted(
        validation_aggregates,
        key=lambda row: (-row["combined_r2_mean"], row["combined_mae_ml_mean"]),
    )
    selected = ranked[0]["condition"]
    decision = {
        "selection_rule": SELECTION_RULE,
        "selected_condition_before_common_envelope_test_evaluation": selected,
        "held_out_test_used_for_selection": False,
        "validation_aggregates": validation_aggregates,
        "decision_recorded_before_test_evaluation": True,
        "command": [sys.executable, *sys.argv],
        "runtime": runtime_metadata(),
    }
    write_json(output_root / "validation_decision_before_test.json", decision)
    print(json.dumps(decision, indent=2, ensure_ascii=False))

    from ablation_common import evaluate, fresh_model, load_clinical_model

    data = load_data(args)
    final_rows = list(condition_a["rows"]) + [
        row for row in rows if row["condition"] == "common_envelope"
    ]
    for seed in SEEDS:
        checkpoint = common_root / f"seed{seed}" / "model.pt"
        model = fresh_model()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
        model.eval()
        wearable = evaluate(model, data["dev_te"], wearable_envelope=True)
        clinical = evaluate(model, data["clin_te"], wearable_envelope=True)
        final_rows.append(result_row("common_envelope", seed, "test", wearable, clinical))
        save_predictions(
            output_root / "common_envelope" / f"seed{seed}" / "predictions_wearable_test.csv",
            data["dev_te"],
            wearable,
        )
        save_predictions(
            output_root / "common_envelope" / f"seed{seed}" / "predictions_clinical_test.csv",
            data["clin_te"],
            clinical,
        )

    stage1_model = load_clinical_model(args.stage1_checkpoint)
    direct_transfer = {}
    for partition, wearable_key, clinical_key in (
        ("validation", "dev_va", "clin_va"),
        ("test", "dev_te", "clin_te"),
    ):
        direct_transfer[partition] = {}
        clinical_inputs, clinical_features = collect_input_and_features(
            stage1_model, data[clinical_key], wearable_envelope=False
        )
        for condition, envelope in (("current_rf", False), ("common_envelope", True)):
            result = evaluate(stage1_model, data[wearable_key], wearable_envelope=envelope)
            wearable_inputs, wearable_features = collect_input_and_features(
                stage1_model, data[wearable_key], wearable_envelope=envelope
            )
            direct_transfer[partition][condition] = {
                "metrics": extended_metrics(result, wearable=True),
                "prediction_by_volume": per_volume_predictions(result),
                "raw_input_mmd": rbf_mmd2(clinical_inputs, wearable_inputs),
                "learned_feature_mmd": rbf_mmd2(clinical_features, wearable_features),
            }
            save_predictions(
                output_root / f"direct_transfer_{condition}_{partition}.csv",
                data[wearable_key],
                result,
            )

    aggregates = [
        aggregate(final_rows, condition, partition)
        for condition in ("current_rf", "common_envelope")
        for partition in ("validation", "test")
    ]
    final = {
        "selection": decision,
        "rows": final_rows,
        "aggregates": aggregates,
        "direct_transfer": direct_transfer,
        "mmd_note": (
            "Biased RBF MMD2 with a pooled median-distance bandwidth was computed under "
            "this controlled split. The implementation that produced the manuscript's "
            "previous 0.373/0.647 values was not found, so these values are comparative "
            "within this ablation and are not treated as a reproduction of those numbers."
        ),
        "stage1_checkpoint_sha256": sha256_file(args.stage1_checkpoint),
        "stage2_checkpoint_sha256": {
            str(seed): sha256_file(common_root / f"seed{seed}" / "model.pt") for seed in SEEDS
        },
        "runtime": runtime_metadata(),
        "git": git_state(PROJECT_ROOT),
    }
    write_json(output_root / "envelope_ablation_final.json", final)
    print(json.dumps(final, indent=2, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    if args.phase == "reproduce-a":
        reproduce_condition_a(args)
    else:
        decide_and_test(args)


if __name__ == "__main__":
    main()
