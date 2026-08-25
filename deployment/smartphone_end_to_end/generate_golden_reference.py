#!/usr/bin/env python3
"""Generate validation-only wearable RF preprocessing golden references."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import scipy
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deployment.dataset import device_partitions, load_preprocessed_partition  # noqa: E402


PHYSICAL_CHANNEL_ORDER = ("S1", "S2", "S3", "S4", "H_left", "H_right")
LOGICAL_CHANNEL_ORDER = (
    "H_left",
    "H_virtual_mean_S2_S3",
    "H_right",
    "S1",
    "S2",
    "S3",
    "S4",
)
GOLDEN_SAMPLE_IDS = ("50_30.txt", "150_30.txt", "300_30.txt")
EXPECTED_MODEL_HASHES = {
    "fp32": "23f46e08bc3c60ce91ce323ce7135bf54cbfd53a26c29d3ae2e7fed4fb662a59",
    "u8u8": "f473b9b8b4a745a9d7579896619af063d2e9cabfaad71fd699728837073a684e",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device-data-root",
        required=True,
        help="Path to the wearable RF data root (not distributed in this release)",
    )
    parser.add_argument(
        "--fp32-model",
        default=str(PROJECT_ROOT / "deployment/artifacts/seed3/model_fp32.onnx"),
    )
    parser.add_argument(
        "--u8u8-model",
        default=str(PROJECT_ROOT / "deployment/artifacts/seed3/model_u8u8_qdq.onnx"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "golden_reference"),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def strict_parse_six_channel_text(path: Path) -> np.ndarray:
    tokens = path.read_text(encoding="utf-8").replace("'", "").replace(" ", "").strip().split(",")
    if not tokens or any(token == "" for token in tokens):
        raise ValueError(f"empty token in raw RF file: {path}")
    try:
        values = np.asarray([float(token) for token in tokens], dtype=np.float32)
    except ValueError as error:
        raise ValueError(f"nonnumeric token in raw RF file: {path}") from error
    expected_values = 6 * 5120
    if values.size != expected_values:
        raise ValueError(f"expected {expected_values} values, found {values.size}: {path}")
    parsed = values.reshape(6, 5120)
    if not np.isfinite(parsed).all():
        raise ValueError(f"non-finite raw RF value: {path}")
    return parsed


def reference_stages(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    parsed = strict_parse_six_channel_text(path)

    scaled = np.empty_like(parsed)
    for channel in range(6):
        scaled[channel] = parsed[channel] / 1700.0

    sample_count = scaled.shape[1]
    sampling_hz = 20e6
    sound_speed_m_s = 1500.0
    distance_mm = (
        np.arange(sample_count, dtype=np.float32)
        / sampling_hz
        * sound_speed_m_s
        / 2.0
        * 1e3
    )
    midpoint = sample_count // 2
    tgc = np.concatenate(
        (
            np.exp(0.15 * distance_mm[:midpoint] / 10.0),
            np.exp(0.4 * distance_mm[midpoint:] / 10.0),
        )
    ).astype(np.float32)
    tgc_output = scaled.copy()
    tgc_output *= tgc[None, :]

    b, a = butter(N=7, Wn=[1.5e6, 4.5e6], btype="band", fs=sampling_hz)
    filtered = tgc_output.copy()
    for channel in range(6):
        filtered[channel] = filtfilt(b, a, filtered[channel]).astype(np.float32)

    clipped_cropped = np.clip(filtered, -0.1, 0.1)[:, 300:3000]
    virtual_horizontal = np.mean(clipped_cropped[1:3], axis=0, keepdims=True)
    logical = np.concatenate(
        (
            clipped_cropped[4:5],
            virtual_horizontal,
            clipped_cropped[5:6],
            clipped_cropped[:4],
        ),
        axis=0,
    ).astype(np.float32)

    # Preserve the released behavior: axis 0 is filtered separately inside the
    # horizontal (3-channel) and sagittal (4-channel) groups.
    smoothed = np.concatenate(
        (
            gaussian_filter1d(logical[:3], sigma=1, axis=0),
            gaussian_filter1d(logical[3:], sigma=1, axis=0),
        ),
        axis=0,
    ).astype(np.float32)
    downsampled = smoothed[:, ::10][:, :220]

    standardized = np.zeros_like(downsampled)
    for channel in range(7):
        mean = downsampled[channel].mean()
        standard_deviation = downsampled[channel].std() + 1e-8
        standardized[channel] = (downsampled[channel] - mean) / standard_deviation

    tail_mean = standardized[:, -30:].mean(axis=1, keepdims=True)
    padded = np.concatenate(
        (standardized, np.repeat(tail_mean, 400 - standardized.shape[1], axis=1)),
        axis=1,
    ).astype(np.float32, copy=False)

    stages = {
        "01_parsed_six_channel_raw": parsed,
        "02_amplitude_scaled": scaled,
        "03_tgc_output": tgc_output,
        "04_bandpass_filtered": filtered,
        "05_clipped_cropped": clipped_cropped,
        "06_logical_seven_channel": logical,
        "07_gaussian_smoothed": smoothed,
        "08_downsampled_220": downsampled,
        "09_standardized": standardized,
        "10_padded_7x400": padded,
    }
    return stages, b, a


def create_session(model_path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )


def predict(session: ort.InferenceSession, model_input: np.ndarray) -> float:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: model_input[None].astype(np.float32)})[0]
    return float(np.asarray(output).reshape(-1)[0])


def save_stage(sample_dir: Path, name: str, values: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(values, dtype="<f4")
    path = sample_dir / f"{name}.f32le.bin"
    array.tofile(path)
    return {
        "file": path.name,
        "shape": list(array.shape),
        "dtype": "float32-little-endian",
        "values": int(array.size),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "mean": float(np.mean(array, dtype=np.float64)),
        "standard_deviation_ddof0": float(np.std(array, dtype=np.float64)),
        "nonfinite_count": int(array.size - np.count_nonzero(np.isfinite(array))),
        "sha256": sha256_file(path),
    }


def prepare_output(path: Path, force: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not force:
            raise FileExistsError(f"output directory is not empty; use --force: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    data_root = Path(args.device_data_root).resolve()
    models = {"fp32": Path(args.fp32_model).resolve(), "u8u8": Path(args.u8u8_model).resolve()}
    prepare_output(output_dir, args.force)

    for name, path in models.items():
        actual_hash = sha256_file(path)
        if actual_hash != EXPECTED_MODEL_HASHES[name]:
            raise RuntimeError(f"{name} model SHA mismatch: {actual_hash}")

    partitions = device_partitions(data_root)
    validation = partitions["validation"].reset_index(drop=True)
    validation_ids = [Path(path).name for path in validation["txt_path"]]
    if any(sample_id not in validation_ids for sample_id in GOLDEN_SAMPLE_IDS):
        raise RuntimeError("predeclared golden sample is not in the validation partition")

    authoritative_inputs, authoritative_labels, authoritative_ids = load_preprocessed_partition(
        data_root, "validation"
    )
    reconstructed = []
    for raw_path in validation["txt_path"]:
        stages, _, _ = reference_stages(Path(raw_path))
        reconstructed.append(stages["10_padded_7x400"])
    reconstructed_inputs = np.stack(reconstructed).astype(np.float32, copy=False)
    if authoritative_ids != validation_ids:
        raise RuntimeError("validation ID order changed between partition and loader")
    validation_difference = np.abs(reconstructed_inputs - authoritative_inputs)
    if not np.array_equal(reconstructed_inputs, authoritative_inputs):
        raise RuntimeError(
            "independent reconstruction is not bitwise equal to the authoritative loader; "
            f"max difference={validation_difference.max()}"
        )

    sessions = {name: create_session(path) for name, path in models.items()}
    sample_manifests = []
    b_reference = None
    a_reference = None
    try:
        for sample_id in GOLDEN_SAMPLE_IDS:
            validation_index = validation_ids.index(sample_id)
            row = validation.iloc[validation_index]
            source_path = Path(row["txt_path"])
            stages, b, a = reference_stages(source_path)
            b_reference = b
            a_reference = a
            sample_dir = output_dir / source_path.stem
            sample_dir.mkdir()
            stage_metadata = {
                name: save_stage(sample_dir, name, values) for name, values in stages.items()
            }
            model_input = stages["10_padded_7x400"]
            predictions = {
                name: predict(session, model_input) for name, session in sessions.items()
            }
            manifest = {
                "sample_id": sample_id,
                "partition": "wearable validation",
                "validation_index": validation_index,
                "target_ml": float(authoritative_labels[validation_index]),
                "raw_source": str(source_path),
                "raw_source_sha256": sha256_file(source_path),
                "physical_channel_order": list(PHYSICAL_CHANNEL_ORDER),
                "logical_channel_order": list(LOGICAL_CHANNEL_ORDER),
                "stages": stage_metadata,
                "predictions_ml": predictions,
            }
            write_json(sample_dir / "sample_manifest.json", manifest)
            sample_manifests.append(manifest)
    finally:
        for session in sessions.values():
            del session

    if b_reference is None or a_reference is None:
        raise RuntimeError("no golden samples were generated")

    coefficient_metadata = {
        "scipy_version": scipy.__version__,
        "design": "butter(N=7, Wn=[1.5e6, 4.5e6], btype='band', fs=20e6)",
        "representation": "ba",
        "filtfilt": {
            "axis": -1,
            "padtype": "odd",
            "padlen_argument": None,
            "effective_default_padlen": 45,
            "method": "pad",
            "irlen": None,
        },
        "b_float64": b_reference.tolist(),
        "a_float64": a_reference.tolist(),
        "gaussian_filter1d": {
            "sigma": 1.0,
            "axis": 0,
            "separate_groups": [3, 4],
            "order": 0,
            "mode": "reflect",
            "cval": 0.0,
            "truncate": 4.0,
            "radius": 4,
        },
    }
    write_json(output_dir / "filter_coefficients.json", coefficient_metadata)

    top_manifest = {
        "format_version": 1,
        "partition": "wearable validation only",
        "held_out_test_used": False,
        "sample_ids": list(GOLDEN_SAMPLE_IDS),
        "validation_partition_size": len(validation),
        "full_validation_reconstruction": {
            "compared_samples": len(validation),
            "bitwise_equal": True,
            "max_absolute_difference": float(validation_difference.max()),
        },
        "models": {
            name: {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for name, path in models.items()
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "onnxruntime": ort.__version__,
            "providers": ort.get_available_providers(),
        },
        "samples": [
            {
                "sample_id": manifest["sample_id"],
                "target_ml": manifest["target_ml"],
                "manifest": f"{Path(manifest['sample_id']).stem}/sample_manifest.json",
                "predictions_ml": manifest["predictions_ml"],
            }
            for manifest in sample_manifests
        ],
    }
    write_json(output_dir / "golden_reference_manifest.json", top_manifest)

    hashes = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            hashes.append(f"{sha256_file(path)}  {path.relative_to(output_dir)}")
    (output_dir / "SHA256SUMS.txt").write_text("\n".join(hashes) + "\n", encoding="ascii")

    print(f"Generated validation-only golden references: {output_dir}")
    print("Validation reconstruction: 48/48 bitwise equal")
    for manifest in sample_manifests:
        prediction = manifest["predictions_ml"]
        print(
            f"{manifest['sample_id']}: target={manifest['target_ml']:.0f} mL, "
            f"FP32={prediction['fp32']:.8f} mL, U8U8={prediction['u8u8']:.8f} mL"
        )


if __name__ == "__main__":
    main()
