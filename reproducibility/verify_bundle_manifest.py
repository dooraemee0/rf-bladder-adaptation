#!/usr/bin/env python3
"""Verify every payload row in FILE_MANIFEST_SHA256.csv."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_root", type=Path)
    args = parser.parse_args()
    root = args.bundle_root.resolve()
    manifest = root / "FILE_MANIFEST_SHA256.csv"
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    failures = []
    for row in rows:
        path = root / row["relative_path"]
        if not path.is_file():
            failures.append(f"missing: {row['relative_path']}")
        elif path.stat().st_size != int(row["byte_size"]):
            failures.append(f"size: {row['relative_path']}")
        elif sha256(path) != row["sha256"]:
            failures.append(f"sha256: {row['relative_path']}")
    print(f"verified payload files: {len(rows)}")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
