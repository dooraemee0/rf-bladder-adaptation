from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch


SEEDS = (1, 2, 3, 42)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SELECTION_RULE = (
    "Maximize wearable validation R2 + clinical-retention validation R2. "
    "Break an exact tie by the lowest sum of wearable and clinical validation MAE. "
    "Held-out test predictions and metrics are not loaded."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select one deployment seed using validation only")
    parser.add_argument("--device-root", required=True)
    parser.add_argument("--clinical-excel", required=True)
    parser.add_argument("--clinical-rf-root", required=True)
    parser.add_argument("--test-index", required=True, help="used only to reconstruct train/validation membership")
    parser.add_argument("--checkpoint-root", default="checkpoints/final/der++")
    parser.add_argument("--output", default="deployment/DEPLOYMENT_MODEL_SELECTION.md")
    parser.add_argument("--json-output", default="deployment/artifacts/deployment_model_selection.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["ABL_DEVICE_BASE"] = str(Path(args.device_root).resolve())
    os.environ["ABL_SNUH_EXCEL"] = str(Path(args.clinical_excel).resolve())
    os.environ["ABL_SNUH_DATA"] = str(Path(args.clinical_rf_root).resolve())
    os.environ["ABL_NUM_WORKERS"] = "0"
    os.environ["ABL_EVAL_NUM_WORKERS"] = "0"
    os.environ["ABL_USE_CACHE"] = "0"

    from ablation_common import evaluate, fresh_model, load_all_data, set_seed
    from deployment.common import runtime_metadata, sha256_file, write_json

    test_index = torch.load(args.test_index, map_location="cpu", weights_only=False)
    if hasattr(test_index, "tolist"):
        test_index = test_index.tolist()
    data = load_all_data(list(test_index))
    checkpoint_root = Path(args.checkpoint_root).resolve()
    rows = []
    for seed in SEEDS:
        checkpoint = checkpoint_root / f"seed{seed}" / "model.pt"
        model = fresh_model()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        model.eval()
        set_seed(seed)
        wearable = evaluate(model, data["dev_va"])
        clinical = evaluate(model, data["clin_va"])
        rows.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "wearable_validation_r2": wearable["r2"],
                "wearable_validation_mae_ml": wearable["mae"],
                "clinical_validation_r2": clinical["r2"],
                "clinical_validation_mae_ml": clinical["mae"],
                "combined_validation_r2": wearable["r2"] + clinical["r2"],
                "combined_validation_mae_ml": wearable["mae"] + clinical["mae"],
            }
        )

    ranked = sorted(
        rows,
        key=lambda row: (
            -row["combined_validation_r2"],
            row["combined_validation_mae_ml"],
        ),
    )
    selected = ranked[0]
    payload = {
        "selection_rule": SELECTION_RULE,
        "held_out_test_used_for_selection": False,
        "selected_seed": selected["seed"],
        "selected_checkpoint": selected["checkpoint"],
        "validation_rows": rows,
        "split_sizes": {
            "wearable_train": len(data["dev_tr"]),
            "wearable_validation": len(data["dev_va"]),
            "clinical_train": len(data["clin_tr"]),
            "clinical_validation": len(data["clin_va"]),
        },
        "runtime": runtime_metadata(),
    }
    write_json(args.json_output, payload)
    Path(args.output).write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"saved: {Path(args.output).resolve()}")


def render_markdown(payload: dict) -> str:
    lines = [
        "# Representative deployment checkpoint selection",
        "",
        f"Selection rule: {payload['selection_rule']}",
        "",
        "The held-out test partition was not evaluated or used by this selection script.",
        "The clinical test index was read only to reconstruct the original non-test train/validation pool.",
        "",
        "| Seed | Wearable val R2 | Clinical val R2 | Combined R2 | Wearable val MAE | Clinical val MAE |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["validation_rows"]:
        lines.append(
            f"| {row['seed']} | {row['wearable_validation_r2']:.6f} | "
            f"{row['clinical_validation_r2']:.6f} | {row['combined_validation_r2']:.6f} | "
            f"{row['wearable_validation_mae_ml']:.3f} mL | "
            f"{row['clinical_validation_mae_ml']:.3f} mL |"
        )
    selected = next(
        row for row in payload["validation_rows"] if row["seed"] == payload["selected_seed"]
    )
    lines.extend(
        [
            "",
            f"Selected seed: **{payload['selected_seed']}**.",
            f"Selected checkpoint SHA-256: `{selected['checkpoint_sha256']}`.",
            "",
            "This choice applies only to the single artifact required for physical-device deployment. "
            "Manuscript accuracy remains a mean +/- seed SD result across all four seeds.",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
