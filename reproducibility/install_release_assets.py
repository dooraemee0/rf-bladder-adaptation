#!/usr/bin/env python3
"""Hash-verify and install the Android ONNX and processed test-input assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


MODEL_NAME = "derpp_seed3_u8u8_qdq.onnx"
TEST_INPUT_NAME = "mobile_test_inputs.bin"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify(path: Path, expected: dict) -> None:
    if not path.is_file():
        raise SystemExit(f"missing release asset: {path}")
    if path.stat().st_size != expected["bytes"]:
        raise SystemExit(
            f"release asset size mismatch for {path}: "
            f"{path.stat().st_size} != {expected['bytes']}"
        )
    observed_hash = sha256(path)
    if observed_hash != expected["sha256"]:
        raise SystemExit(
            f"release asset SHA-256 mismatch for {path}: "
            f"{observed_hash} != {expected['sha256']}"
        )


def install(source: Path, destinations: list[Path], repo: Path) -> None:
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() == destination.resolve():
            print(f"verified existing {destination.relative_to(repo)}")
            continue
        shutil.copy2(source, destination)
        print(f"installed {destination.relative_to(repo)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument(
        "--test-inputs",
        type=Path,
        help="Optional processed 60-sample tensor asset for model-only validation.",
    )
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    expected_list = json.loads((repo / "reproducibility/RELEASE_ARTIFACTS.json").read_text())
    expected = {item["file"]: item for item in expected_list}

    model = args.onnx.resolve()
    verify(model, expected[MODEL_NAME])
    install(
        model,
        [
            repo / "android_mobile_benchmark/app/src/main/assets/model_8bit.onnx",
            repo / "android_mobile_benchmark/benchmark/src/androidTest/assets/model_8bit.onnx",
        ],
        repo,
    )

    if args.test_inputs:
        test_inputs = args.test_inputs.resolve()
        verify(test_inputs, expected[TEST_INPUT_NAME])
        install(
            test_inputs,
            [
                repo / "android_mobile_benchmark/app/src/main/assets/test_inputs.bin",
                repo / "android_mobile_benchmark/benchmark/src/androidTest/assets/test_inputs.bin",
                repo / "android_mobile_benchmark/app/src/androidTest/assets/heldout_expected_inputs.f32le.bin",
            ],
            repo,
        )

    print("all supplied Android release assets verified")


if __name__ == "__main__":
    main()
