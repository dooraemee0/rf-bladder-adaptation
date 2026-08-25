#!/usr/bin/env python3
"""Verify frozen Android result files without generating new performance results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


MODEL_SHA = "f473b9b8b4a745a9d7579896619af063d2e9cabfaad71fd699728837073a684e"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    results = args.repo_root.resolve() / "deployment/public_results"

    parity = read_json(results / "preprocessing/preprocessing_parity_android.json")
    assert parity["partition"] == "wearable validation only"
    assert parity["held_out_test_used"] is False
    assert len(parity["samples"]) == 3
    final_stage_max = max(sample["stages"][-1]["maximum_absolute_difference"] for sample in parity["samples"])
    assert final_stage_max <= 2.771615982055664e-6

    heldout = read_json(results / "heldout/heldout_raw_rf_concordance_android.json")
    assert heldout["partition"] == "held-out wearable test"
    assert heldout["sample_count"] == 60
    assert heldout["model_sha256"] == MODEL_SHA
    assert heldout["missing_sample_count"] == 0
    assert heldout["duplicate_sample_count"] == 0
    assert heldout["nonfinite_tensor_count"] == 0
    assert heldout["nonfinite_prediction_count"] == 0
    assert heldout["prediction_maximum_absolute_difference_ml"] == 0

    with (results / "heldout/heldout_raw_rf_predictions_android.csv").open(newline="", encoding="utf-8-sig") as handle:
        predictions = list(csv.DictReader(handle))
    assert len(predictions) == 60

    preprocessing_medians = []
    raw_to_volume_medians = []
    for session_index in (1, 2, 3):
        report = read_json(results / f"raw_rf_pipeline/session{session_index}_benchmarkData.json")
        by_name = {entry["name"]: entry for entry in report["benchmarks"]}
        preprocessing = by_name["sixChannelRawRfToModelTensor"]
        raw_to_volume = by_name["sixChannelRawRfToScalarVolume"]
        assert len(preprocessing["metrics"]["timeNs"]["runs"]) == 50
        assert len(raw_to_volume["metrics"]["timeNs"]["runs"]) == 50
        assert preprocessing["thermalThrottleSleepSeconds"] == 0
        assert raw_to_volume["thermalThrottleSleepSeconds"] == 0
        preprocessing_medians.append(preprocessing["metrics"]["timeNs"]["median"] / 1e6)
        raw_to_volume_medians.append(raw_to_volume["metrics"]["timeNs"]["median"] / 1e6)

    print(f"PASS validation preprocessing parity: final-stage max={final_stage_max:.9g}")
    print("PASS held-out Android/desktop concordance: n=60, prediction max difference=0 mL")
    print(f"PASS three raw-RF benchmark sessions: median-of-session-medians preprocessing={statistics.median(preprocessing_medians):.3f} ms")
    print(f"PASS three raw-RF benchmark sessions: median-of-session-medians raw-RF-to-volume={statistics.median(raw_to_volume_medians):.3f} ms")


if __name__ == "__main__":
    main()
