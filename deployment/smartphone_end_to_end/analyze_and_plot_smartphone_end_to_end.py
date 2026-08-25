#!/usr/bin/env python3
"""Recompute and plot physical Android raw-RF deployment results.

The script reads only archived physical-device outputs. It keeps the timed
boundaries explicit: preprocessing starts with six in-memory RF arrays, while
the combined benchmark ends after scalar ONNX output retrieval and resource
closure. BLE transfer, asset reads, session creation, and UI rendering are not
part of the standardized warm-latency results.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parent
RAW_RESULTS = ROOT / "raw" / "final_windows_results" / "results"
PARITY_JSON = ROOT / "raw" / "preprocessing_parity" / "preprocessing_parity_android.json"
MODEL_ONLY_ROOT = ROOT.parent / "android_physical_validation" / "raw"
ANALYSIS = ROOT / "analysis"
FIGURES = ROOT / "figures"

BENCHMARK_FILES = [
    RAW_RESULTS / f"raw_pipeline_benchmark_session{session}" / "benchmarkData.json"
    for session in (1, 2, 3)
]
MODEL_ONLY_FILES = [
    MODEL_ONLY_ROOT / f"microbenchmark_session{session}.json"
    for session in (1, 2, 3)
]

PREPROCESSING_NAME = "sixChannelRawRfToModelTensor"
COMBINED_NAME = "sixChannelRawRfToScalarVolume"

COLORS = {
    50: "#2878B5",
    150: "#E58A2B",
    300: "#2A9D78",
    "dark": "#20242A",
    "gray": "#747B83",
    "light_gray": "#E8EAED",
    "blue": "#3A79A8",
    "green": "#3B8F71",
    "red": "#C83E3E",
    "gold": "#D9A126",
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.titleweight": "bold",
            "axes.labelsize": 8,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def nearest_rank(values: np.ndarray, percentile: float) -> float:
    ordered = np.sort(np.asarray(values, dtype=float))
    index = max(0, math.ceil(percentile / 100.0 * len(ordered)) - 1)
    return float(ordered[index])


def summarize(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        raise ValueError("At least two values are required")
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    return {
        "n": int(values.size),
        "mean": mean,
        "median": float(np.median(values)),
        "p90_nearest_rank": nearest_rank(values, 90),
        "p95_nearest_rank": nearest_rank(values, 95),
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "sample_sd": sd,
        "coefficient_of_variation": sd / mean,
    }


def benchmark_runs(payload: dict[str, Any], name: str) -> tuple[np.ndarray, dict[str, Any]]:
    benchmark = next(item for item in payload["benchmarks"] if item["name"] == name)
    runs = np.asarray(benchmark["metrics"]["timeNs"]["runs"], dtype=float) / 1e6
    return runs, benchmark


def load_latency() -> tuple[dict[str, Any], dict[str, list[np.ndarray]]]:
    payloads = [load_json(path) for path in BENCHMARK_FILES]
    series: dict[str, list[np.ndarray]] = {"preprocessing": [], "raw_rf_to_volume": []}
    metadata: list[dict[str, Any]] = []
    names = {"preprocessing": PREPROCESSING_NAME, "raw_rf_to_volume": COMBINED_NAME}

    for session_index, payload in enumerate(payloads, start=1):
        context = payload["context"]
        metadata.append(
            {
                "session": session_index,
                "device_model": context["build"]["model"],
                "android_sdk": int(context["build"]["version"]["sdk"]),
                "cpu_core_count": int(context["cpuCoreCount"]),
                "cpu_max_frequency_hz": int(context["cpuMaxFreqHz"]),
                "cpu_locked": bool(context["cpuLocked"]),
                "sustained_performance_mode": bool(context["sustainedPerformanceModeEnabled"]),
                "compilation_mode": context["compilationMode"],
            }
        )
        for scope, benchmark_name in names.items():
            runs, benchmark = benchmark_runs(payload, benchmark_name)
            if len(runs) != 50 or benchmark["thermalThrottleSleepSeconds"] != 0:
                raise RuntimeError(f"Unexpected benchmark status for {scope}, session {session_index}")
            series[scope].append(runs)

    metrics: dict[str, Any] = {"metadata": metadata, "scopes": {}}
    for scope, sessions in series.items():
        session_summaries = [summarize(values) for values in sessions]
        for index, summary in enumerate(session_summaries, start=1):
            summary["session"] = index
        pooled = np.concatenate(sessions)
        medians = np.asarray([np.median(values) for values in sessions], dtype=float)
        metrics["scopes"][scope] = {
            "sessions": session_summaries,
            "pooled": summarize(pooled),
            "session_median_mean": float(np.mean(medians)),
            "session_median_sample_sd": float(np.std(medians, ddof=1)),
            "session_median_range": float(np.max(medians) - np.min(medians)),
        }

    model_only_sessions: list[np.ndarray] = []
    for path in MODEL_ONLY_FILES:
        payload = load_json(path)
        runs = np.asarray(payload["benchmarks"][0]["metrics"]["timeNs"]["runs"], dtype=float) / 1e6
        if len(runs) != 50:
            raise RuntimeError("Unexpected model-only run count")
        model_only_sessions.append(runs)
    model_only_pooled = np.concatenate(model_only_sessions)
    metrics["prior_model_only_reference"] = {
        "scope": "OrtSession.run with a pre-created tensor; separate benchmark campaign",
        "pooled": summarize(model_only_pooled),
        "session_medians": [float(np.median(values)) for values in model_only_sessions],
    }
    return metrics, series


def load_predictions() -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    csv_path = RAW_RESULTS / "heldout_raw_rf" / "heldout_raw_rf_predictions_android.csv"
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8-sig")))
    if len(rows) != 60:
        raise RuntimeError("Expected 60 held-out predictions")
    arrays = {
        "target": np.asarray([float(row["target_ml"]) for row in rows]),
        "android": np.asarray([float(row["android_raw_rf_u8u8_ml"]) for row in rows]),
        "desktop": np.asarray([float(row["desktop_u8u8_ml"]) for row in rows]),
        "tensor_max_difference": np.asarray(
            [float(row["tensor_max_abs_difference"]) for row in rows]
        ),
        "tensor_mean_difference": np.asarray(
            [float(row["tensor_mean_abs_difference"]) for row in rows]
        ),
    }
    target = arrays["target"]
    prediction = arrays["android"]
    residual = prediction - target
    actual_alarm = target >= 200.0
    predicted_alarm = prediction >= 200.0
    tn = int(np.sum(~actual_alarm & ~predicted_alarm))
    fp = int(np.sum(~actual_alarm & predicted_alarm))
    fn = int(np.sum(actual_alarm & ~predicted_alarm))
    tp = int(np.sum(actual_alarm & predicted_alarm))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    metrics = {
        "sample_count": len(rows),
        "mae_ml": float(np.mean(np.abs(residual))),
        "rmse_ml": float(np.sqrt(np.mean(residual**2))),
        "r2": 1.0 - ss_res / ss_tot,
        "within_50_ml_accuracy": float(np.mean(np.abs(residual) <= 50.0)),
        "alarm_threshold_ml": 200.0,
        "alarm_accuracy": float((tn + tp) / len(rows)),
        "alarm_precision": float(tp / (tp + fp)),
        "alarm_recall": float(tp / (tp + fn)),
        "alarm_f1": float(2 * tp / (2 * tp + fp + fn)),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "android_desktop_prediction_maximum_difference_ml": float(
            np.max(np.abs(arrays["android"] - arrays["desktop"]))
        ),
        "tensor_maximum_absolute_difference": float(np.max(arrays["tensor_max_difference"])),
        "per_reference_volume": {},
    }
    for volume in (50, 150, 300):
        selected = prediction[target == volume]
        metrics["per_reference_volume"][str(volume)] = {
            "n": int(selected.size),
            "prediction_mean_ml": float(np.mean(selected)),
            "prediction_sample_sd_ml": float(np.std(selected, ddof=1)),
        }

    android_report = load_json(
        RAW_RESULTS / "heldout_raw_rf" / "heldout_raw_rf_concordance_android.json"
    )["performance"]
    checks = {
        "mae_ml": "mae_ml",
        "rmse_ml": "rmse_ml",
        "r2": "r2",
        "within_50_ml_accuracy": "within_50_ml_accuracy",
        "alarm_accuracy": "alarm_accuracy",
        "alarm_precision": "alarm_precision",
        "alarm_recall": "alarm_recall",
        "alarm_f1": "alarm_f1",
    }
    for local_key, report_key in checks.items():
        if not math.isclose(metrics[local_key], android_report[report_key], rel_tol=0, abs_tol=1e-9):
            raise RuntimeError(f"Recomputed metric mismatch: {local_key}")
    return metrics, arrays


def load_initialization_and_memory() -> tuple[dict[str, Any], dict[str, Any]]:
    cold = load_json(RAW_RESULTS / "initialization_and_memory" / "cold_start_android.json")
    component_keys = [
        "environment_acquisition_ms",
        "model_asset_read_ms",
        "session_options_ms",
        "session_create_ms",
        "input_asset_decode_and_tensor_create_ms",
        "total_initialization_ms",
    ]
    initialization: dict[str, Any] = {
        "scope": cold["scope"],
        "true_process_cold_start_measured": cold["true_process_cold_start_measured"],
        "components": {},
    }
    for key in component_keys:
        values = np.asarray([float(item[key]) for item in cold["iterations"]])
        initialization["components"][key] = summarize(values)
        initialization["components"][key]["values_ms"] = values.tolist()

    memory_report = load_json(
        RAW_RESULTS / "initialization_and_memory" / "memory_profile_android.json"
    )
    baseline = int(memory_report["observations"][0]["total_pss_kb"])
    observations = []
    for item in memory_report["observations"]:
        observations.append(
            {
                "stage": item["stage"],
                "total_pss_kb": int(item["total_pss_kb"]),
                "total_pss_mib": float(item["total_pss_kb"] / 1024.0),
                "delta_from_baseline_kb": int(item["total_pss_kb"] - baseline),
                "runtime_java_heap_used_bytes": int(item["runtime_java_heap_used_bytes"]),
                "native_heap_allocated_bytes": int(item["native_heap_allocated_bytes"]),
            }
        )
    maximum = max(observations, key=lambda item: item["total_pss_kb"])
    memory = {
        "scope": memory_report["scope"],
        "peak_memory_measured": memory_report["peak_memory_measured"],
        "continuous_sampling_used": memory_report["continuous_sampling_used"],
        "model_file_size_bytes": int(memory_report["model_file_size_bytes"]),
        "baseline_total_pss_kb": baseline,
        "maximum_observed_total_pss_kb": maximum["total_pss_kb"],
        "maximum_observed_stage": maximum["stage"],
        "maximum_observed_delta_from_baseline_kb": maximum["delta_from_baseline_kb"],
        "observations": observations,
    }
    return initialization, memory


def load_parity() -> tuple[dict[str, Any], dict[str, np.ndarray | list[str]]]:
    payload = load_json(PARITY_JSON)
    stage_names = [item["stage"] for item in payload["samples"][0]["stages"]]
    stage_maxima = np.asarray(
        [
            max(sample["stages"][index]["maximum_absolute_difference"] for sample in payload["samples"])
            for index in range(len(stage_names))
        ],
        dtype=float,
    )
    prediction_differences = np.asarray(
        [sample["prediction_absolute_difference_ml"] for sample in payload["samples"]], dtype=float
    )
    metrics = {
        "validation_sample_count": len(payload["samples"]),
        "held_out_test_used": payload["held_out_test_used"],
        "final_tensor_maximum_absolute_difference": float(stage_maxima[-1]),
        "prediction_maximum_absolute_difference_ml": float(np.max(prediction_differences)),
    }
    return metrics, {"stage_names": stage_names, "stage_maxima": stage_maxima}


def load_local_report() -> dict[str, Any]:
    report = load_json(RAW_RESULTS / "local_app" / "local_inference_report.json")
    manifest = load_json(
        ROOT
        / "android_mobile_benchmark"
        / "app"
        / "src"
        / "main"
        / "assets"
        / "local_raw_manifest.json"
    )
    sample = next(item for item in manifest["samples"] if item["sample_id"] == report["sample_id"])
    report["reference_volume_ml"] = float(sample["target_ml"])
    report["absolute_error_ml"] = abs(
        float(report["predicted_volume_ml"]) - report["reference_volume_ml"]
    )
    report["standardized_latency_result"] = False
    return report


def write_latency_csv(series: dict[str, list[np.ndarray]]) -> None:
    path = ANALYSIS / "latency_runs.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scope", "session", "run_index", "latency_ms"])
        for scope, sessions in series.items():
            for session_index, runs in enumerate(sessions, start=1):
                for run_index, value in enumerate(runs, start=1):
                    writer.writerow([scope, session_index, run_index, f"{value:.12f}"])


def add_panel_label(ax: plt.Axes, label: str, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
    )


def style_axes(ax: plt.Axes, grid: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(color="#D7DADF", linewidth=0.55, linestyle="--", alpha=0.65)
        ax.set_axisbelow(True)


def panel_pipeline(ax: plt.Axes, local: dict[str, Any]) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Smartphone-resident, server-free RF-to-volume pipeline", pad=5)

    labels = [
        ("Six-channel raw RF", "6 x 5120 float32"),
        ("Android preprocessing", "TGC, filter, crop,\n7-channel construction"),
        ("Model tensor", "1 x 7 x 400"),
        ("U8U8 ONNX Runtime", "CPU EP, one thread"),
        (f"{local['predicted_volume_ml']:.1f} mL", "Normal (<200 mL)"),
    ]
    colors = ["#E8F0F6", "#E8F3EE", "#F4EFE1", "#EDEAF5", "#F0F2F3"]
    widths = [0.155, 0.20, 0.13, 0.18, 0.14]
    gaps = 0.035
    total = sum(widths) + gaps * (len(widths) - 1)
    left = (1 - total) / 2
    centers = []
    for index, ((title, subtitle), width, color) in enumerate(zip(labels, widths, colors)):
        box = FancyBboxPatch(
            (left, 0.28),
            width,
            0.42,
            boxstyle="round,pad=0.008,rounding_size=0.015",
            linewidth=0.9,
            edgecolor="#879099",
            facecolor=color,
        )
        ax.add_patch(box)
        center = left + width / 2
        centers.append(center)
        ax.text(center, 0.54, title, ha="center", va="center", fontsize=7.6, fontweight="bold")
        ax.text(center, 0.41, subtitle, ha="center", va="center", fontsize=6.6, color=COLORS["gray"])
        if index < len(labels) - 1:
            next_left = left + width + gaps
            ax.annotate(
                "",
                xy=(next_left - 0.005, 0.49),
                xytext=(left + width + 0.005, 0.49),
                arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": COLORS["dark"]},
            )
        left += width + gaps

    ax.text(
        0.02,
        0.87,
        "Physical Samsung Galaxy S8+ | Android 9 | DER++ seed 3 | static U8U8 QDQ",
        ha="left",
        va="center",
        fontsize=7,
        color=COLORS["dark"],
    )
    ax.text(
        0.98,
        0.87,
        "Server used: no",
        ha="right",
        va="center",
        fontsize=7,
        fontweight="bold",
        color=COLORS["green"],
    )
    ax.text(
        0.5,
        0.09,
        f"Local demonstration: {local['sample_id']} ({local['partition']}); "
        f"reference {local['reference_volume_ml']:.0f} mL, "
        f"absolute error {local['absolute_error_ml']:.1f} mL",
        ha="center",
        va="center",
        fontsize=6.7,
        color=COLORS["gray"],
    )


def panel_regression(ax: plt.Axes, arrays: dict[str, np.ndarray], metrics: dict[str, Any]) -> None:
    target = arrays["target"]
    prediction = arrays["android"]
    lo, hi = 0.0, 350.0
    xline = np.linspace(lo, hi, 300)
    ax.fill_between(
        xline,
        np.maximum(lo, xline - 50.0),
        np.minimum(hi, xline + 50.0),
        color="#DFEDE2",
        alpha=0.9,
        linewidth=0,
        label=r"$\pm$50 mL",
    )
    ax.plot(xline, xline, color=COLORS["dark"], linewidth=1.05, linestyle="--", label="Identity")
    rng = np.random.default_rng(20260817)
    for volume in (50, 150, 300):
        mask = target == volume
        jitter = rng.uniform(-6.0, 6.0, int(np.sum(mask)))
        ax.scatter(
            target[mask] + jitter,
            prediction[mask],
            s=23,
            color=COLORS[volume],
            edgecolor="white",
            linewidth=0.45,
            alpha=0.9,
            zorder=3,
            label=f"{volume} mL (n={int(np.sum(mask))})",
        )
        ax.errorbar(
            volume,
            float(np.mean(prediction[mask])),
            yerr=float(np.std(prediction[mask], ddof=1)),
            fmt="D",
            markersize=4.1,
            color=COLORS[volume],
            markeredgecolor=COLORS["dark"],
            markeredgewidth=0.45,
            capsize=3,
            linewidth=1,
            zorder=4,
        )
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xticks([0, 50, 150, 200, 300, 350])
    ax.set_yticks(np.arange(0, 351, 50))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Reference phantom volume (mL)")
    ax.set_ylabel("Android raw-RF prediction (mL)")
    ax.set_title("Held-out on-device regression")
    ax.legend(loc="upper left", frameon=False, ncol=2, handletextpad=0.3, columnspacing=0.65)
    ax.text(
        0.97,
        0.04,
        (
            f"$R^2$ = {metrics['r2']:.3f}\n"
            f"MAE = {metrics['mae_ml']:.2f} mL\n"
            f"RMSE = {metrics['rmse_ml']:.2f} mL\n"
            f"$\\pm$50 mL = {100 * metrics['within_50_ml_accuracy']:.1f}%"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.9,
        bbox={"boxstyle": "square,pad=0.34", "fc": "white", "ec": "#9DA3AA", "lw": 0.7},
    )
    style_axes(ax)


def panel_regression_no_text(
    ax: plt.Axes,
    arrays: dict[str, np.ndarray],
    metrics: dict[str, Any],
) -> None:
    """Hide text while retaining every graphical element and its layout."""
    panel_regression(ax, arrays, metrics)
    transparent = (0.0, 0.0, 0.0, 0.0)
    ax.title.set_color(transparent)
    ax.xaxis.label.set_color(transparent)
    ax.yaxis.label.set_color(transparent)
    for tick_label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        tick_label.set_color(transparent)
    legend = ax.get_legend()
    if legend is not None:
        for legend_text in legend.get_texts():
            legend_text.set_color(transparent)
    for annotation in list(ax.texts):
        annotation.set_color(transparent)


def panel_confusion(ax: plt.Axes, metrics: dict[str, Any]) -> None:
    matrix = np.asarray(
        [
            [metrics["true_negative"], metrics["false_positive"]],
            [metrics["false_negative"], metrics["true_positive"]],
        ],
        dtype=int,
    )
    fractions = matrix / matrix.sum(axis=1, keepdims=True)
    cmap = LinearSegmentedColormap.from_list("paper_blue", ["#F4F7F9", "#2D6F9E"])
    ax.imshow(fractions, cmap=cmap, vmin=0, vmax=1)
    for row in range(2):
        for col in range(2):
            color = "white" if fractions[row, col] > 0.55 else COLORS["dark"]
            ax.text(
                col,
                row,
                f"{matrix[row, col]}\n({100 * fractions[row, col]:.1f}%)",
                ha="center",
                va="center",
                fontsize=9.5,
                fontweight="bold",
                color=color,
            )
    ax.set_xticks([0, 1], ["Normal", "Alarm"])
    ax.set_yticks([0, 1], ["Normal", "Alarm"])
    ax.set_xlabel("Predicted state")
    ax.set_ylabel("Reference state")
    ax.set_title("200 mL alarm\nclassification", pad=4)
    ax.tick_params(length=0)
    ax.text(
        0.5,
        -0.23,
        f"Accuracy = {100 * metrics['alarm_accuracy']:.1f}%   F1 = {metrics['alarm_f1']:.3f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.2,
    )


def panel_latency(
    ax: plt.Axes,
    series: dict[str, list[np.ndarray]],
    metrics: dict[str, Any],
) -> None:
    scopes = ["preprocessing", "raw_rf_to_volume"]
    positions = [1.0, 2.0]
    pooled_values = [np.concatenate(series[scope]) for scope in scopes]
    box = ax.boxplot(
        pooled_values,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": COLORS["dark"], "linewidth": 1.2},
        whiskerprops={"color": COLORS["gray"], "linewidth": 0.8},
        capprops={"color": COLORS["gray"], "linewidth": 0.8},
        boxprops={"edgecolor": COLORS["gray"], "linewidth": 0.8},
    )
    for patch, color in zip(box["boxes"], ["#CFE2EF", "#CFE8DD"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.85)

    session_colors = ["#2F68A1", "#D4842C", "#2A9270"]
    offsets = [-0.16, 0.0, 0.16]
    rng = np.random.default_rng(20260817)
    for scope_index, scope in enumerate(scopes):
        for session_index, runs in enumerate(series[scope]):
            x = (
                positions[scope_index]
                + offsets[session_index]
                + rng.uniform(-0.025, 0.025, len(runs))
            )
            ax.scatter(
                x,
                runs,
                s=7,
                color=session_colors[session_index],
                alpha=0.24,
                linewidth=0,
                zorder=2,
            )
            ax.scatter(
                positions[scope_index] + offsets[session_index],
                np.median(runs),
                marker="D",
                s=24,
                color=session_colors[session_index],
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
                label=f"Session {session_index + 1}" if scope_index == 0 else None,
            )
        pooled = metrics["scopes"][scope]["pooled"]
        ax.text(
            positions[scope_index],
            46.0,
            f"Median {pooled['median']:.2f} ms\nP90 {pooled['p90_nearest_rank']:.2f} ms",
            ha="center",
            va="top",
            fontsize=6.8,
            bbox={"boxstyle": "square,pad=0.3", "fc": "white", "ec": "#B3B7BC", "lw": 0.6},
        )
    ax.set_xlim(0.5, 2.5)
    ax.set_ylim(0, 48)
    ax.set_xticks(positions, ["RF preprocessing", "Raw RF to volume"])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Standardized warm latency (n = 150 per scope)")
    ax.legend(frameon=False, loc="lower right", ncol=1)
    style_axes(ax)


def panel_parity(ax: plt.Axes, parity_arrays: dict[str, np.ndarray | list[str]]) -> None:
    names = [
        "Parse",
        "Scale",
        "TGC",
        "Band-pass",
        "Crop",
        "7-channel",
        "Gaussian",
        "Downsample",
        "Standardize",
        "Final tensor",
    ]
    values = np.asarray(parity_arrays["stage_maxima"], dtype=float)
    values_micro = values * 1e6
    mean_micro = float(np.mean(values_micro))
    colors = [COLORS["gray"] if value == 0 else COLORS["blue"] for value in values]
    ax.bar(np.arange(len(values)), values_micro, color=colors, width=0.7)
    ax.axhline(mean_micro, color=COLORS["red"], linestyle="--", linewidth=0.9)
    for index, value in enumerate(values):
        if value == 0:
            ax.text(index, 0.5, "0", ha="center", va="bottom", fontsize=6, color=COLORS["gray"])
    ax.text(
        0.98,
        mean_micro + 0.5,
        f"Mean = {mean_micro:.2f}",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=6.3,
        color=COLORS["red"],
    )
    ax.set_ylim(0, 40)
    ax.set_xticks(np.arange(len(names)), names, rotation=43, ha="right")
    ax.set_ylabel(r"Maximum absolute difference ($\times 10^{-6}$)")
    ax.set_title("Android-Python preprocessing parity")
    style_axes(ax, grid=True)


def panel_heldout_concordance(ax: plt.Axes, arrays: dict[str, np.ndarray]) -> None:
    target = arrays["target"]
    difference = arrays["tensor_max_difference"]
    difference_micro = difference * 1e6
    mean_micro = float(np.mean(difference_micro))
    for volume in (50, 150, 300):
        mask = target == volume
        indices = np.flatnonzero(mask)
        ax.scatter(
            indices + 1,
            difference_micro[mask],
            s=18,
            color=COLORS[volume],
            edgecolor="white",
            linewidth=0.35,
            label=f"{volume} mL",
        )
    ax.axhline(mean_micro, color=COLORS["red"], linestyle="--", linewidth=0.9)
    ax.set_ylim(0, 4)
    ax.set_xlabel("Held-out sample index")
    ax.set_ylabel(r"Tensor max. absolute difference ($\times 10^{-6}$)")
    ax.set_title("Frozen held-out raw-RF concordance")
    ax.legend(frameon=False, ncol=3, loc="lower right")
    ax.text(
        0.02,
        0.96,
        "Android-desktop prediction max. difference = 0 mL",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.7,
        bbox={"boxstyle": "square,pad=0.3", "fc": "white", "ec": "#B3B7BC", "lw": 0.6},
    )
    style_axes(ax)


def panel_initialization(ax: plt.Axes, initialization: dict[str, Any]) -> None:
    keys = [
        "environment_acquisition_ms",
        "model_asset_read_ms",
        "session_options_ms",
        "session_create_ms",
        "input_asset_decode_and_tensor_create_ms",
        "total_initialization_ms",
    ]
    labels = ["ORT env.", "Model read", "Options", "Session", "Input tensor", "Total"]
    values = [np.asarray(initialization["components"][key]["values_ms"]) for key in keys]
    box = ax.boxplot(
        values,
        patch_artist=True,
        showfliers=True,
        widths=0.58,
        medianprops={"color": COLORS["dark"], "linewidth": 1.1},
        boxprops={"edgecolor": COLORS["gray"], "linewidth": 0.8},
        whiskerprops={"color": COLORS["gray"], "linewidth": 0.8},
        capprops={"color": COLORS["gray"], "linewidth": 0.8},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": COLORS["red"], "markeredgewidth": 0},
    )
    for index, patch in enumerate(box["boxes"]):
        patch.set_facecolor("#DDE8EF" if index < 5 else "#E4EEDC")
    rng = np.random.default_rng(20260817)
    for index, series in enumerate(values, start=1):
        ax.scatter(
            index + rng.uniform(-0.08, 0.08, len(series)),
            series,
            s=9,
            color=COLORS["dark"],
            alpha=0.35,
            linewidth=0,
        )
    total_median = initialization["components"]["total_initialization_ms"]["median"]
    ax.text(6, 275, f"Total median\n{total_median:.1f} ms", ha="center", va="top", fontsize=6.8)
    ax.set_xticks(range(1, 7), labels, rotation=30, ha="right")
    ax.set_ylabel("Initialization time (ms)")
    ax.set_title("Repeated fresh session initialization (n = 10)")
    style_axes(ax)


def panel_memory(ax: plt.Axes, memory: dict[str, Any]) -> None:
    labels = ["Baseline", "ORT session", "Raw RF loaded", "Preprocessed", "100 runs", "Closed + GC"]
    values = np.asarray([item["total_pss_mib"] for item in memory["observations"]])
    x = np.arange(len(values))
    ax.plot(x, values, color=COLORS["green"], linewidth=1.4, marker="o", markersize=4.5)
    ax.fill_between(x, values, memory["baseline_total_pss_kb"] / 1024.0, color="#DCECE5", alpha=0.6)
    for index, value in enumerate(values):
        ax.text(index, value + 1.2, f"{value:.1f}", ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x, labels, rotation=32, ha="right")
    ax.set_ylabel("Total PSS point observation (MiB)")
    ax.set_ylim(0, max(values) + 9)
    ax.set_title("Process memory point observations (not peaks)")
    style_axes(ax)


def save_panel(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_figures(
    metrics: dict[str, Any],
    latency_series: dict[str, list[np.ndarray]],
    prediction_arrays: dict[str, np.ndarray],
    parity_arrays: dict[str, np.ndarray | list[str]],
) -> None:
    # Main figure: pipeline, regression, alarm confusion, and warm latency.
    fig = plt.figure(figsize=(7.2, 6.3))
    grid = fig.add_gridspec(2, 3, height_ratios=[0.9, 2.0], width_ratios=[1.15, 0.8, 1.15])
    ax_a = fig.add_subplot(grid[0, :])
    ax_b = fig.add_subplot(grid[1, 0])
    ax_c = fig.add_subplot(grid[1, 1])
    ax_d = fig.add_subplot(grid[1, 2])
    panel_pipeline(ax_a, metrics["local_demonstration"])
    panel_regression(ax_b, prediction_arrays, metrics["heldout_performance"])
    panel_confusion(ax_c, metrics["heldout_performance"])
    panel_latency(ax_d, latency_series, metrics["latency"])
    add_panel_label(ax_a, "a", x=-0.01, y=1.06)
    add_panel_label(ax_b, "b", x=-0.13, y=1.15)
    add_panel_label(ax_c, "c", x=-0.32, y=1.20)
    add_panel_label(ax_d, "d", x=-0.18, y=1.15)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.96, bottom=0.09, wspace=0.42, hspace=0.48)
    save_panel(fig, "figure_smartphone_raw_rf_validation_main")

    # Supplement: numerical parity, held-out concordance, initialization, and memory.
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2))
    panel_parity(axes[0, 0], parity_arrays)
    panel_heldout_concordance(axes[0, 1], prediction_arrays)
    panel_initialization(axes[1, 0], metrics["initialization"])
    panel_memory(axes[1, 1], metrics["memory"])
    for label, ax in zip(("a", "b", "c", "d"), axes.flat):
        add_panel_label(ax, label, x=-0.13, y=1.15)
    fig.subplots_adjust(left=0.09, right=0.985, top=0.95, bottom=0.12, wspace=0.34, hspace=0.48)
    save_panel(fig, "figure_smartphone_implementation_validation_supplement")

    panel_specs = [
        ("panel_a_smartphone_pipeline", panel_pipeline, (metrics["local_demonstration"],), (7.0, 2.1)),
        (
            "panel_b_heldout_raw_rf_regression",
            panel_regression,
            (prediction_arrays, metrics["heldout_performance"]),
            (3.3, 3.2),
        ),
        (
            "panel_b_heldout_raw_rf_regression_no_text",
            panel_regression_no_text,
            (prediction_arrays, metrics["heldout_performance"]),
            (3.3, 3.2),
        ),
        ("panel_c_alarm_confusion", panel_confusion, (metrics["heldout_performance"],), (3.0, 3.0)),
        ("panel_d_raw_pipeline_latency", panel_latency, (latency_series, metrics["latency"]), (3.4, 3.2)),
    ]
    for stem, function, arguments, size in panel_specs:
        fig, ax = plt.subplots(figsize=size)
        function(ax, *arguments)
        fig.tight_layout()
        save_panel(fig, stem)


def main() -> None:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    latency_metrics, latency_series = load_latency()
    heldout_metrics, prediction_arrays = load_predictions()
    initialization, memory = load_initialization_and_memory()
    parity_metrics, parity_arrays = load_parity()
    local = load_local_report()

    metrics = {
        "analysis_policy": {
            "latency_percentiles": "nearest-rank",
            "session_variability": "mean and sample SD of three session medians",
            "memory_boundary": "point observations; not continuously sampled peaks",
        },
        "device": {
            "manufacturer": "Samsung",
            "model": "Galaxy S8+ (SM-G955N)",
            "android": "9 (API 28)",
            "hardware": "Exynos 8895",
        },
        "latency": latency_metrics,
        "heldout_performance": heldout_metrics,
        "initialization": initialization,
        "memory": memory,
        "preprocessing_parity": parity_metrics,
        "local_demonstration": local,
    }
    (ANALYSIS / "smartphone_end_to_end_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    write_latency_csv(latency_series)
    generate_figures(metrics, latency_series, prediction_arrays, parity_arrays)
    print(json.dumps(metrics["latency"]["scopes"], indent=2))
    print(f"Wrote metrics to {ANALYSIS / 'smartphone_end_to_end_metrics.json'}")
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
