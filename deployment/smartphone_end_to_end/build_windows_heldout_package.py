#!/usr/bin/env python3
"""Build the post-freeze, one-time held-out Android evaluation package."""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE_ANDROID = ROOT / "android_mobile_benchmark"
WINDOWS_PACKAGE = ROOT / "windows_package"
DESTINATION_ANDROID = WINDOWS_PACKAGE / "android_mobile_benchmark"
FREEZE = ROOT / "ANDROID_IMPLEMENTATION_FREEZE.json"
OUTPUT_ZIP = ROOT / "smartphone_end_to_end_heldout_windows_package.zip"


def ignored(_directory: str, names: list[str]) -> set[str]:
    excluded = {"build", ".gradle", ".idea", "local.properties"}
    return {name for name in names if name in excluded}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if DESTINATION_ANDROID.exists():
        shutil.rmtree(DESTINATION_ANDROID)
    shutil.copytree(SOURCE_ANDROID, DESTINATION_ANDROID, ignore=ignored)
    shutil.copy2(FREEZE, WINDOWS_PACKAGE / FREEZE.name)

    model = DESTINATION_ANDROID / "app/src/main/assets/model_8bit.onnx"
    expected_model_hash = "f473b9b8b4a745a9d7579896619af063d2e9cabfaad71fd699728837073a684e"
    if sha256(model) != expected_model_hash:
        raise RuntimeError("frozen U8U8 model hash mismatch")
    heldout_manifest = DESTINATION_ANDROID / "app/src/androidTest/assets/heldout_raw_manifest.json"
    if not heldout_manifest.is_file():
        raise RuntimeError("held-out raw manifest was not generated")

    manifest_path = WINDOWS_PACKAGE / "hashes.sha256"
    files = sorted(
        path for path in WINDOWS_PACKAGE.rglob("*")
        if path.is_file() and path != manifest_path
    )
    manifest_path.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(WINDOWS_PACKAGE).as_posix()}\n" for path in files),
        encoding="utf-8",
    )

    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(WINDOWS_PACKAGE.rglob("*")):
            if path.is_file():
                archive.write(path, Path("windows_package") / path.relative_to(WINDOWS_PACKAGE))

    print(f"Created: {OUTPUT_ZIP}")
    print(f"Size: {OUTPUT_ZIP.stat().st_size} bytes")
    print(f"SHA-256: {sha256(OUTPUT_ZIP)}")


if __name__ == "__main__":
    main()
