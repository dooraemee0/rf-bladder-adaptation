#!/usr/bin/env python3
"""Package the frozen held-out raw RF set after Android validation parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

from generate_golden_reference import create_session, predict, reference_stages


ROOT = Path(__file__).resolve().parent
ANDROID = ROOT / "android_mobile_benchmark"
FREEZE = ROOT / "ANDROID_IMPLEMENTATION_FREEZE.json"
TEST_MANIFEST = ANDROID / "app/src/main/assets/test_manifest.json"
TEST_INPUTS = ANDROID / "app/src/main/assets/test_inputs.bin"
MODEL = ANDROID / "app/src/main/assets/model_8bit.onnx"
TEST_ASSETS = ANDROID / "app/src/androidTest/assets"
RAW_DESTINATION = TEST_ASSETS / "raw_rf_heldout"
EXPECTED_INPUTS_DESTINATION = TEST_ASSETS / "heldout_expected_inputs.f32le.bin"
OUTPUT_MANIFEST = TEST_ASSETS / "heldout_raw_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        required=True,
        help="Path to the wearable RF data root (not distributed in this release)",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_production() -> dict:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    aggregate = hashlib.sha256()
    for row in freeze["files"]:
        relative = row["path"]
        path = ANDROID / relative
        observed = sha256(path)
        if observed != row["sha256"]:
            raise RuntimeError(f"frozen production file changed: {relative}: {observed}")
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(observed.encode("ascii"))
        aggregate.update(b"\n")
    observed_manifest = aggregate.hexdigest()
    if observed_manifest != freeze["production_manifest_sha256"]:
        raise RuntimeError("frozen production aggregate hash mismatch")
    return freeze


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    freeze = verify_frozen_production()
    manifest = json.loads(TEST_MANIFEST.read_text(encoding="utf-8"))
    samples = manifest["samples"]
    if len(samples) != 60:
        raise RuntimeError(f"expected 60 held-out samples, found {len(samples)}")
    sample_ids = [row["sample_id"] for row in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("duplicate held-out sample ID")

    expected_inputs = np.fromfile(TEST_INPUTS, dtype="<f4").reshape(60, 7, 400)
    if sha256(TEST_INPUTS) != manifest["inputs_sha256"]:
        raise RuntimeError("existing frozen test input hash mismatch")
    session = create_session(MODEL)

    if RAW_DESTINATION.exists():
        shutil.rmtree(RAW_DESTINATION)
    RAW_DESTINATION.mkdir(parents=True)
    shutil.copy2(TEST_INPUTS, EXPECTED_INPUTS_DESTINATION)

    output_samples = []
    prediction_differences = []
    for row in samples:
        index = int(row["index"])
        target_ml = float(row["target_ml"])
        sample_id = row["sample_id"]
        source = raw_root / str(int(target_ml)) / sample_id
        if not source.is_file():
            raise FileNotFoundError(source)
        stages, _, _ = reference_stages(source)
        raw = np.asarray(stages["01_parsed_six_channel_raw"], dtype="<f4")
        model_input = np.asarray(stages["10_padded_7x400"], dtype=np.float32)
        if not np.array_equal(model_input, expected_inputs[index]):
            difference = float(np.max(np.abs(model_input - expected_inputs[index])))
            raise RuntimeError(f"held-out tensor mismatch for {sample_id}: {difference}")

        observed_prediction = predict(session, model_input)
        expected_prediction = float(row["desktop_8bit_ml"])
        prediction_difference = abs(observed_prediction - expected_prediction)
        prediction_differences.append(prediction_difference)
        if prediction_difference > 1e-5:
            raise RuntimeError(
                f"held-out desktop prediction mismatch for {sample_id}: {prediction_difference} mL"
            )

        asset_name = f"{Path(sample_id).stem}.f32le.bin"
        destination = RAW_DESTINATION / asset_name
        raw.tofile(destination)
        output_samples.append(
            {
                "index": index,
                "sample_id": sample_id,
                "partition": "held-out wearable test",
                "target_ml": target_ml,
                "asset": f"raw_rf_heldout/{asset_name}",
                "shape": [6, 5120],
                "dtype": "float32-little-endian",
                "raw_sha256": sha256(destination),
                "expected_tensor_sha256": hashlib.sha256(
                    expected_inputs[index].astype("<f4", copy=False).tobytes()
                ).hexdigest(),
                "expected_desktop_u8u8_ml": expected_prediction,
            }
        )

    output = {
        "format_version": 1,
        "partition": "held-out wearable test",
        "sample_count": 60,
        "one_time_evaluation_after_validation_freeze": True,
        "implementation_freeze_manifest_sha256": freeze["production_manifest_sha256"],
        "checkpoint_sha256": freeze["checkpoint_sha256"],
        "model_sha256": freeze["u8u8_onnx_sha256"],
        "expected_inputs_asset": EXPECTED_INPUTS_DESTINATION.name,
        "expected_inputs_sha256": sha256(EXPECTED_INPUTS_DESTINATION),
        "physical_channel_order": ["S1", "S2", "S3", "S4", "H_left", "H_right"],
        "logical_channel_order": [
            "H_left",
            "H_virtual_mean_S2_S3",
            "H_right",
            "S1",
            "S2",
            "S3",
            "S4",
        ],
        "desktop_recheck_prediction_max_abs_difference_ml": max(prediction_differences),
        "samples": output_samples,
    }
    OUTPUT_MANIFEST.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_MANIFEST}")
    print(f"Samples: {len(output_samples)}")
    print(f"Raw bytes: {sum((RAW_DESTINATION / Path(x['asset']).name).stat().st_size for x in output_samples)}")
    print(f"Desktop prediction recheck max difference: {max(prediction_differences):.9g} mL")


if __name__ == "__main__":
    main()
