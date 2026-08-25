from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from common import regression_metrics, write_json


def desktop_prediction(row: dict) -> float:
    for key in ("desktop_8bit_ml", "desktop_int8_ml"):
        if key in row:
            return float(row[key])
    raise KeyError("manifest sample has no desktop 8-bit prediction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Android and desktop static 8-bit predictions")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--android-csv", required=True)
    parser.add_argument("--output", default="deployment/artifacts/android_comparison.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    expected = {row["sample_id"]: row for row in manifest["samples"]}

    actual = {}
    with Path(args.android_csv).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            actual[row["sample_id"]] = float(row["prediction_ml"])

    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if missing or extra:
        raise ValueError(f"sample mismatch: missing={missing[:5]}, extra={extra[:5]}")

    sample_ids = [row["sample_id"] for row in manifest["samples"]]
    target = np.asarray([expected[key]["target_ml"] for key in sample_ids])
    desktop = np.asarray([desktop_prediction(expected[key]) for key in sample_ids])
    android = np.asarray([actual[key] for key in sample_ids])
    difference = android - desktop
    result = {
        "n": len(sample_ids),
        "android_vs_desktop": {
            "max_absolute_output_difference_ml": float(np.max(np.abs(difference))),
            "mean_absolute_output_difference_ml": float(np.mean(np.abs(difference))),
        },
        "desktop_8bit": regression_metrics(target, desktop),
        "android_8bit": regression_metrics(target, android),
        "sample_concordance": [
            {
                "sample_id": key,
                "target_ml": float(y),
                "desktop_8bit_ml": float(d),
                "android_8bit_ml": float(a),
                "difference_ml": float(a - d),
            }
            for key, y, d, a in zip(sample_ids, target, desktop, android)
        ],
    }
    write_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "sample_concordance"}, indent=2))
    print(f"saved: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
