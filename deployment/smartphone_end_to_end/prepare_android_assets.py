#!/usr/bin/env python3
"""Package validation-only raw RF and golden stages for the Android workspace."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GOLDEN = ROOT / "golden_reference"
ANDROID = ROOT / "android_mobile_benchmark"
SAMPLE_IDS = ("50_30", "150_30", "300_30")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    golden_manifest = json.loads((GOLDEN / "golden_reference_manifest.json").read_text())
    indexed = {Path(sample["sample_id"]).stem: sample for sample in golden_manifest["samples"]}

    app_assets = ANDROID / "app/src/main/assets/raw_rf_validation"
    benchmark_assets = ANDROID / "benchmark/src/androidTest/assets/raw_rf_validation"
    parity_assets = ANDROID / "app/src/androidTest/assets/golden_validation"
    for directory in (app_assets, benchmark_assets, parity_assets):
        directory.mkdir(parents=True, exist_ok=True)

    manifest_samples = []
    for sample_id in SAMPLE_IDS:
        source_dir = GOLDEN / sample_id
        source_manifest = json.loads((source_dir / "sample_manifest.json").read_text())
        parsed_name = "01_parsed_six_channel_raw.f32le.bin"
        for destination in (app_assets, benchmark_assets):
            shutil.copy2(source_dir / parsed_name, destination / f"{sample_id}.f32le.bin")

        parity_sample_dir = parity_assets / sample_id
        parity_sample_dir.mkdir(parents=True, exist_ok=True)
        for stage in source_manifest["stages"].values():
            shutil.copy2(source_dir / stage["file"], parity_sample_dir / stage["file"])
        shutil.copy2(source_dir / "sample_manifest.json", parity_sample_dir / "sample_manifest.json")

        top_sample = indexed[sample_id]
        packaged_file = app_assets / f"{sample_id}.f32le.bin"
        manifest_samples.append(
            {
                "sample_id": f"{sample_id}.txt",
                "partition": "wearable validation",
                "target_ml": top_sample["target_ml"],
                "asset": f"raw_rf_validation/{sample_id}.f32le.bin",
                "shape": [6, 5120],
                "dtype": "float32-little-endian",
                "sha256": sha256_file(packaged_file),
                "expected_u8u8_ml": top_sample["predictions_ml"]["u8u8"],
            }
        )

    manifest = {
        "format_version": 1,
        "partition": "wearable validation only",
        "held_out_test_used": False,
        "physical_channel_order": ["S1", "S2", "S3", "S4", "H_left", "H_right"],
        "shape_per_sample": [6, 5120],
        "dtype": "float32-little-endian",
        "samples": manifest_samples,
    }
    write_json(ANDROID / "app/src/main/assets/local_raw_manifest.json", manifest)
    write_json(ANDROID / "benchmark/src/androidTest/assets/local_raw_manifest.json", manifest)
    shutil.copy2(
        GOLDEN / "golden_reference_manifest.json",
        parity_assets / "golden_reference_manifest.json",
    )
    print(f"Packaged {len(manifest_samples)} validation-only raw RF samples")


if __name__ == "__main__":
    main()
