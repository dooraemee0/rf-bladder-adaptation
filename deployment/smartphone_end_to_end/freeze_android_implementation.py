#!/usr/bin/env python3
"""Record hashes of the validation-approved production Android implementation."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ANDROID = ROOT / "android_mobile_benchmark"
OUTPUT = ROOT / "ANDROID_IMPLEMENTATION_FREEZE.json"

PRODUCTION_PATHS = [
    "settings.gradle.kts",
    "build.gradle.kts",
    "gradle.properties",
    "app/build.gradle.kts",
    "app/src/main/AndroidManifest.xml",
    "app/src/main/assets/model_8bit.onnx",
    "app/src/main/assets/local_raw_manifest.json",
    "app/src/main/java/org/dgist/bladderbenchmark/LocalInferencePipeline.kt",
    "app/src/main/java/org/dgist/bladderbenchmark/LocalRawRfAssetLoader.kt",
    "app/src/main/java/org/dgist/bladderbenchmark/MainActivity.kt",
    "app/src/main/java/org/dgist/bladderbenchmark/OrtModelRunner.kt",
    "pipeline/build.gradle.kts",
    "pipeline/src/main/AndroidManifest.xml",
    "pipeline/src/main/java/org/dgist/bladderbenchmark/preprocessing/ButterworthFilter.kt",
    "pipeline/src/main/java/org/dgist/bladderbenchmark/preprocessing/ChannelConstructor.kt",
    "pipeline/src/main/java/org/dgist/bladderbenchmark/preprocessing/ChannelStandardizer.kt",
    "pipeline/src/main/java/org/dgist/bladderbenchmark/preprocessing/GaussianChannelSmoother.kt",
    "pipeline/src/main/java/org/dgist/bladderbenchmark/preprocessing/PreprocessingMetadata.kt",
    "pipeline/src/main/java/org/dgist/bladderbenchmark/preprocessing/TgcProcessor.kt",
    "pipeline/src/main/java/org/dgist/bladderbenchmark/preprocessing/WearableRfPreprocessor.kt",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    files = []
    aggregate = hashlib.sha256()
    for relative in PRODUCTION_PATHS:
        path = ANDROID / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = sha256(path)
        size = path.stat().st_size
        files.append({"path": relative, "sha256": digest, "size_bytes": size})
        aggregate.update(relative.encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(digest.encode("ascii"))
        aggregate.update(b"\n")

    payload = {
        "status": "frozen_after_validation_only_android_python_parity",
        "freeze_date": date.today().isoformat(),
        "held_out_raw_rf_used_before_freeze": False,
        "validation_samples": ["50_30.txt", "150_30.txt", "300_30.txt"],
        "validation_final_prediction_max_abs_difference_ml": 0.0,
        "validation_final_tensor_max_abs_difference": 2.771615982055664e-6,
        "checkpoint_sha256": "ba9e8bd59b8eeb14ea09cbb2fbd613fad4490699064c25e8d882681d04ce2572",
        "u8u8_onnx_sha256": "f473b9b8b4a745a9d7579896619af063d2e9cabfaad71fd699728837073a684e",
        "production_manifest_sha256": aggregate.hexdigest(),
        "files": files,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Production manifest SHA-256: {payload['production_manifest_sha256']}")


if __name__ == "__main__":
    main()
