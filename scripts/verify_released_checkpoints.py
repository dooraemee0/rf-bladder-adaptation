#!/usr/bin/env python3
"""Verify the v1.0.0 checkpoint ZIP or an extracted checkpoint directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import BinaryIO, Dict


REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "release" / "checkpoint_manifest.json"


def stream_sha256(stream: BinaryIO) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return stream_sha256(stream)


def verify_zip(path: Path, manifest: Dict[str, object]) -> None:
    release = manifest["release"]
    observed_size = path.stat().st_size
    observed_hash = file_sha256(path)
    if observed_size != release["asset_size_bytes"]:
        raise RuntimeError(
            f"ZIP size mismatch: {observed_size} != {release['asset_size_bytes']}"
        )
    if observed_hash != release["asset_sha256"]:
        raise RuntimeError(
            f"ZIP SHA-256 mismatch: {observed_hash} != {release['asset_sha256']}"
        )

    with zipfile.ZipFile(path) as archive:
        archive_files = {info.filename: info for info in archive.infolist() if not info.is_dir()}
        expected_files = manifest["checkpoints"]
        unexpected = sorted(set(archive_files) - set(expected_files))
        missing = sorted(set(expected_files) - set(archive_files))
        if unexpected or missing:
            raise RuntimeError(
                f"ZIP content mismatch; missing={missing}, unexpected={unexpected}"
            )
        for relative_path, expected in expected_files.items():
            info = archive_files[relative_path]
            if info.file_size != expected["size_bytes"]:
                raise RuntimeError(f"Size mismatch for {relative_path}")
            with archive.open(info) as stream:
                observed = stream_sha256(stream)
            if observed != expected["sha256"]:
                raise RuntimeError(f"SHA-256 mismatch for {relative_path}")


def verify_directory(path: Path, manifest: Dict[str, object]) -> None:
    expected_files = manifest["checkpoints"]
    for relative_path, expected in expected_files.items():
        relative = Path(relative_path)
        candidates = [path / relative, path / relative.relative_to("checkpoints/final")]
        checkpoint = next((candidate for candidate in candidates if candidate.is_file()), None)
        if checkpoint is None:
            raise FileNotFoundError(
                f"Missing {relative_path}; searched under {path.resolve()}"
            )
        if checkpoint.stat().st_size != expected["size_bytes"]:
            raise RuntimeError(f"Size mismatch for {checkpoint}")
        observed = file_sha256(checkpoint)
        if observed != expected["sha256"]:
            raise RuntimeError(f"SHA-256 mismatch for {checkpoint}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--zip", type=Path, help="Downloaded GitHub Release ZIP")
    source.add_argument(
        "--checkpoint-root",
        type=Path,
        help="Repository root, checkpoints/final, or an equivalent extracted tree",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if args.zip:
        verify_zip(args.zip, manifest)
        checked = args.zip
    else:
        verify_directory(args.checkpoint_root, manifest)
        checked = args.checkpoint_root
    print(f"Released checkpoint verification: PASS ({checked})")
    print(f"Verified {len(manifest['checkpoints'])} model state dictionaries")
    print(f"Manifest SHA-256: {file_sha256(MANIFEST_PATH)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Released checkpoint verification: FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
