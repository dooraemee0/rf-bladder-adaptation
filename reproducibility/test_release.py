from __future__ import annotations

import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class PublicReleaseTests(unittest.TestCase):
    def test_release_assets_are_unique_and_declared(self) -> None:
        assets = json.loads((ROOT / "reproducibility/RELEASE_ARTIFACTS.json").read_text())
        names = [item["file"] for item in assets]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names.count("derpp_seed3_u8u8_qdq.onnx"), 1)

    def test_heldout_manifest_is_phantom_only(self) -> None:
        manifest = json.loads(
            (ROOT / "android_mobile_benchmark/app/src/androidTest/assets/heldout_raw_manifest.json").read_text()
        )
        samples = manifest["samples"]
        self.assertEqual(len(samples), 60)
        self.assertEqual({float(sample["target_ml"]) for sample in samples}, {50.0, 150.0, 300.0})
        text = json.dumps(manifest).lower()
        for forbidden in ("patient", "hospital", "mrn", "subject_id"):
            self.assertNotIn(forbidden, text)

    def test_android_preprocessing_contract_is_frozen(self) -> None:
        metadata = (
            ROOT
            / "android_mobile_benchmark/pipeline/src/main/java/org/dgist/bladderbenchmark/preprocessing/PreprocessingMetadata.kt"
        ).read_text()
        self.assertRegex(metadata, r"RETAINED_SAMPLES\s*=\s*220")
        self.assertRegex(metadata, r"MODEL_SAMPLES\s*=\s*400")
        self.assertRegex(metadata, r"TAIL_MEAN_SAMPLES\s*=\s*30")

    def test_recorded_heldout_result_has_no_missing_samples(self) -> None:
        result = json.loads(
            (ROOT / "deployment/public_results/heldout/heldout_raw_rf_concordance_android.json").read_text()
        )
        self.assertEqual(result["sample_count"], 60)
        self.assertEqual(result["missing_sample_count"], 0)
        self.assertEqual(result["duplicate_sample_count"], 0)
        self.assertEqual(result["prediction_maximum_absolute_difference_ml"], 0)


if __name__ == "__main__":
    unittest.main()
