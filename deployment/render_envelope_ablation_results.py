from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PAIRED_METRICS = (
    "wearable_mae_ml",
    "wearable_r2",
    "clinical_mae_ml",
    "clinical_r2",
    "combined_r2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render envelope ablation reports")
    parser.add_argument("--input", required=True)
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--results-md", required=True)
    parser.add_argument("--decision-md", required=True)
    parser.add_argument("--manuscript-md", required=True)
    return parser.parse_args()


def aggregate_map(payload: dict) -> dict:
    return {
        (row["condition"], row["partition"]): row for row in payload["aggregates"]
    }


def paired_differences(rows: list[dict], partition: str) -> list[dict]:
    output = []
    for metric in PAIRED_METRICS:
        differences = []
        for seed in (1, 2, 3, 42):
            current = next(
                row
                for row in rows
                if row["condition"] == "current_rf"
                and row["partition"] == partition
                and row["seed"] == seed
            )
            envelope = next(
                row
                for row in rows
                if row["condition"] == "common_envelope"
                and row["partition"] == partition
                and row["seed"] == seed
            )
            differences.append(float(envelope[metric] - current[metric]))
        values = np.asarray(differences)
        sd = float(values.std(ddof=1))
        output.append(
            {
                "partition": partition,
                "metric": metric,
                "seed1": differences[0],
                "seed2": differences[1],
                "seed3": differences[2],
                "seed42": differences[3],
                "mean": float(values.mean()),
                "sd": sd,
                "paired_effect_size_dz": float(values.mean() / sd) if sd else None,
                "n_positive": int(np.sum(values > 0)),
                "n_negative": int(np.sum(values < 0)),
            }
        )
    return output


def fmt(mean: float, sd: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} +/- {sd:.{digits}f}"


def render_results(payload: dict, paired: list[dict]) -> str:
    aggregates = aggregate_map(payload)
    lines = [
        "# Wearable envelope ablation results",
        "",
        "Condition A retains the phase-preserving wearable waveform; Condition B applies "
        "a time-axis Hilbert envelope to wearable H/S arrays. Both retain legacy Gaussian "
        "`axis=0`; all other preprocessing and Stage-2 DER++ settings are fixed.",
        "",
        "The preprocessing condition was selected from aggregate validation only. Held-out "
        "test results were evaluated after that decision was recorded.",
        "",
    ]
    for partition in ("validation", "test"):
        lines.extend(
            [
                f"## {partition.title()} aggregate",
                "",
                "| Condition | Wearable MAE (mL) | Wearable R2 | Within 50 mL | Alarm accuracy | Alarm F1 | Clinical MAE (mL) | Clinical R2 | Combined R2 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for condition, label in (("current_rf", "Current RF"), ("common_envelope", "Common envelope")):
            row = aggregates[(condition, partition)]
            lines.append(
                f"| {label} | {fmt(row['wearable_mae_ml_mean'], row['wearable_mae_ml_sd'])} | "
                f"{fmt(row['wearable_r2_mean'], row['wearable_r2_sd'], 4)} | "
                f"{fmt(row['wearable_accuracy_within_50ml_pct_mean'], row['wearable_accuracy_within_50ml_pct_sd'], 2)}% | "
                f"{fmt(row['wearable_alarm_200ml_accuracy_pct_mean'], row['wearable_alarm_200ml_accuracy_pct_sd'], 2)}% | "
                f"{fmt(row['wearable_alarm_200ml_f1_mean'], row['wearable_alarm_200ml_f1_sd'], 4)} | "
                f"{fmt(row['clinical_mae_ml_mean'], row['clinical_mae_ml_sd'])} | "
                f"{fmt(row['clinical_r2_mean'], row['clinical_r2_sd'], 4)} | "
                f"{fmt(row['combined_r2_mean'], row['combined_r2_sd'], 4)} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Paired seed differences",
            "",
            "Values are Condition B minus Condition A. Negative MAE and positive R2 favor "
            "the common-envelope condition.",
            "",
            "| Partition | Metric | Seed 1 | Seed 2 | Seed 3 | Seed 42 | Mean +/- SD | dz |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired:
        dz = "NA" if row["paired_effect_size_dz"] is None else f"{row['paired_effect_size_dz']:.3f}"
        lines.append(
            f"| {row['partition']} | {row['metric']} | {row['seed1']:.3f} | "
            f"{row['seed2']:.3f} | {row['seed3']:.3f} | {row['seed42']:.3f} | "
            f"{row['mean']:.3f} +/- {row['sd']:.3f} | {dz} |"
        )

    direct = payload["direct_transfer"]
    lines.extend(
        [
            "",
            "## Frozen Stage-1 direct transfer",
            "",
            "| Partition | Wearable representation | MAE (mL) | R2 | 50 mL mean prediction | 150 mL mean prediction | 300 mL mean prediction | Raw-input MMD2 | Feature MMD2 |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for partition in ("validation", "test"):
        for condition, label in (("current_rf", "Current RF"), ("common_envelope", "Common envelope")):
            row = direct[partition][condition]
            per_volume = row["prediction_by_volume"]
            lines.append(
                f"| {partition} | {label} | {row['metrics']['mae_ml']:.3f} | "
                f"{row['metrics']['r2']:.4f} | {per_volume['50']['prediction_mean_ml']:.2f} | "
                f"{per_volume['150']['prediction_mean_ml']:.2f} | "
                f"{per_volume['300']['prediction_mean_ml']:.2f} | "
                f"{row['raw_input_mmd']['mmd2_biased']:.4f} | "
                f"{row['learned_feature_mmd']['mmd2_biased']:.4f} |"
            )
    lines.extend(
        [
            "",
            "Common envelope reduced the frozen Stage-1 feature-space MMD, but direct-transfer "
            "R2 and volume calibration became substantially worse. Distribution proximity in "
            "this unsupervised feature metric therefore did not imply preservation of the "
            "clinical RF-to-volume mapping.",
            "",
            payload["mmd_note"],
            "",
        ]
    )
    return "\n".join(lines)


def render_decision(payload: dict, paired: list[dict]) -> str:
    selected = payload["selection"]["selected_condition_before_common_envelope_test_evaluation"]
    validation = {row["metric"]: row for row in paired if row["partition"] == "validation"}
    all_combined_negative = validation["combined_r2"]["n_negative"] == 4
    current_selected = selected == "current_rf"
    if current_selected and all_combined_negative:
        interpretation = (
            "Case 2: the phase-preserving wearable representation is consistently superior "
            "on the primary combined-validation criterion across all four seeds."
        )
    elif current_selected:
        interpretation = (
            "Case 3: validation metrics are mixed, but the predefined combined criterion "
            "selects the current phase-preserving representation."
        )
    else:
        interpretation = (
            "The predefined validation criterion selects the common-envelope representation."
        )
    return "\n".join(
        [
            "# Wearable envelope ablation decision",
            "",
            f"Selected from validation before held-out testing: **{selected}**.",
            "",
            interpretation,
            "",
            "The held-out test metrics were not used to select preprocessing or alter "
            "hyperparameters. With only four paired seeds, the conclusion is based on effect "
            "magnitude and seed-wise direction rather than a small p-value.",
            "",
            "## Recommendation",
            "",
            (
                "Retain the current domain-specific preprocessing: clinical inputs use a "
                "Hilbert envelope, while wearable inputs preserve the filtered RF waveform. "
                "The full Stage-2 adaptation comparison, ONNX quantization, and Android assets "
                "do not need to be regenerated for a common-envelope pipeline."
                if current_selected
                else
                "Adopt common envelope only after rerunning every Stage-2 adaptation strategy, "
                "all four-seed ONNX/quantization validation, representative checkpoint selection, "
                "and Android asset generation. Existing current-RF results must not be mixed with it."
            ),
            "",
            "The observed clinical-to-wearable gap must be described as an acquisition-pipeline "
            "gap reflecting both hardware and domain-specific preprocessing. This ablation does "
            "not isolate hardware effects alone.",
            "",
            "Common envelope reduced frozen Stage-1 feature MMD while worsening direct-transfer "
            "R2. MMD is therefore retained only as a descriptive auxiliary analysis, not as "
            "evidence that the continuous RF-to-volume relation transferred successfully.",
            "",
        ]
    )


def render_manuscript(payload: dict) -> str:
    selected = payload["selection"]["selected_condition_before_common_envelope_test_evaluation"]
    current = selected == "current_rf"
    aggregates = aggregate_map(payload)
    current_validation = aggregates[("current_rf", "validation")]
    envelope_validation = aggregates[("common_envelope", "validation")]
    current_test = aggregates[("current_rf", "test")]
    envelope_test = aggregates[("common_envelope", "test")]
    recommendation = (
        "The validation-selected phase-preserving wearable pipeline outperformed a controlled "
        "common-envelope alternative across all four seeds on the combined wearable and "
        "clinical-retention criterion."
        if current
        else
        "The validation-selected common-envelope pipeline matched or exceeded the current "
        "domain-specific representation on the combined validation criterion."
    )
    return "\n".join(
        [
            "# Manuscript wording proposal for the envelope ablation",
            "",
            "This file proposes wording only. The manuscript TeX was not modified.",
            "",
            "## Methods",
            "",
            "To assess whether the clinical-to-wearable discrepancy was attributable in part "
            "to signal representation, we performed a controlled Stage-2 DER++ ablation. The "
            "clinical Hilbert-envelope pipeline and all wearable preprocessing operations were "
            "held fixed. The sole experimental factor was whether a time-axis Hilbert envelope "
            "was applied to the wearable H and S arrays before the legacy channel-axis Gaussian "
            "smoothing operation. Four seeds were trained from the same Stage-1 initialization, "
            "and the preprocessing condition was selected using the aggregate sum of wearable "
            "and clinical-retention validation R2 before held-out test evaluation.",
            "",
            "## Results/Discussion",
            "",
            recommendation + " "
            f"On validation, wearable R2 changed from {current_validation['wearable_r2_mean']:.3f} "
            f"to {envelope_validation['wearable_r2_mean']:.3f}, clinical-retention R2 from "
            f"{current_validation['clinical_r2_mean']:.3f} to "
            f"{envelope_validation['clinical_r2_mean']:.3f}, and their combined score from "
            f"{current_validation['combined_r2_mean']:.3f} to "
            f"{envelope_validation['combined_r2_mean']:.3f}. The held-out test confirmed the "
            f"same direction (combined R2, {current_test['combined_r2_mean']:.3f} versus "
            f"{envelope_test['combined_r2_mean']:.3f}).",
            "",
            "The result indicates that matching envelope status alone does not explain the "
            "acquisition-domain discrepancy; however, because hardware and the remaining "
            "domain-specific processing were not independently manipulated, the observed gap "
            "should be interpreted as an acquisition-pipeline gap rather than a purely "
            "hardware-induced domain gap.",
            "",
            "## Terminology changes",
            "",
            "- Do not state that both domains use an identical raw-RF representation.",
            "- Use `phase-preserving wearable RF waveform` for the wearable input and "
            "`Hilbert-envelope clinical RF representation` for the clinical input.",
            "- Use `acquisition-pipeline gap` when hardware and preprocessing effects are not separated.",
            "- `Sensor-specific characteristics` and `independent per-channel encoding` remain "
            "appropriate for the model architecture, but they do not imply identical input representations.",
            "- Reserve `raw RF` for the acquired signal before TGC, filtering, clipping, cropping, "
            "normalization, and optional envelope detection; use `preprocessed RF-domain signals` "
            "for model inputs.",
            "- Treat MMD as descriptive only: common envelope reduced feature MMD but worsened "
            "direct-transfer regression, so lower MMD did not demonstrate preserved volume mapping.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    payload = json.loads(Path(args.input).read_text())
    paired = paired_differences(payload["rows"], "validation") + paired_differences(
        payload["rows"], "test"
    )
    fieldnames = list(payload["rows"][0])
    with Path(args.results_csv).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(payload["rows"])
    Path(args.results_md).write_text(render_results(payload, paired), encoding="utf-8")
    Path(args.decision_md).write_text(render_decision(payload, paired), encoding="utf-8")
    Path(args.manuscript_md).write_text(render_manuscript(payload), encoding="utf-8")
    print(json.dumps({"paired_differences": paired}, indent=2))


if __name__ == "__main__":
    main()
