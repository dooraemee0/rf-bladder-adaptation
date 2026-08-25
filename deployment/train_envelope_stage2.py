from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deployment.envelope_ablation_utils import (  # noqa: E402
    SELECTION_RULE,
    extended_metrics,
    git_state,
    preprocessing_configuration,
    runtime_metadata,
    save_predictions,
    sha256_file,
    split_manifest,
    write_json,
)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train common-envelope DER++ Stage 2")
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--test-index", required=True)
    parser.add_argument("--device-root", required=True)
    parser.add_argument("--clinical-excel", required=True)
    parser.add_argument("--clinical-rf-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--der-alpha", type=float, default=0.5)
    parser.add_argument("--der-beta", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    os.environ["ABL_DEVICE_BASE"] = str(Path(args.device_root).resolve())
    os.environ["ABL_SNUH_EXCEL"] = str(Path(args.clinical_excel).resolve())
    os.environ["ABL_SNUH_DATA"] = str(Path(args.clinical_rf_root).resolve())
    os.environ["ABL_DEV_VAL_RATIO"] = "0.2"
    os.environ["ABL_CLIN_VAL_RATIO"] = "0.2"
    os.environ["ABL_NUM_WORKERS"] = str(args.num_workers)
    os.environ["ABL_EVAL_NUM_WORKERS"] = "0"
    os.environ["ABL_PIN_MEMORY"] = "1"
    os.environ["ABL_USE_CACHE"] = "1"
    os.environ["ABL_CACHE_DIR"] = str(Path(args.cache_dir).resolve())
    os.environ["ABL_WEARABLE_ENVELOPE"] = "1"


def main() -> None:
    args = parse_args()
    configure_environment(args)
    seeds = [int(value) for value in args.seeds.split(",") if value]
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    stage1 = Path(args.stage1_checkpoint).resolve()

    # Imports must follow environment configuration because ablation_common
    # intentionally snapshots its data/preprocessing configuration at import.
    from ablation_common import (  # noqa: E402
        count_trainable,
        evaluate,
        load_all_data,
        load_clinical_model,
        set_seed,
    )
    from strategies import train_der_plus_plus  # noqa: E402

    selection_record = {
        "selection_rule_recorded_before_training_and_test_evaluation": SELECTION_RULE,
        "held_out_test_used_for_training_or_selection": False,
        "command": [sys.executable, *sys.argv],
        "runtime": runtime_metadata(),
        "git": git_state(PROJECT_ROOT),
        "stage1_checkpoint": str(stage1),
        "stage1_checkpoint_sha256": sha256_file(stage1),
        "test_index": str(Path(args.test_index).resolve()),
        "test_index_sha256": sha256_file(args.test_index),
        "preprocessing": preprocessing_configuration(wearable_envelope=True),
        "training": {
            "strategy": "der++",
            "epochs": args.epochs,
            "optimizer": "AdamW",
            "learning_rate": args.lr,
            "weight_decay": 1e-3,
            "scheduler": "CosineAnnealingLR",
            "batch_size": 16,
            "der_alpha": args.der_alpha,
            "der_beta": args.der_beta,
            "validation_frequency_epochs": 30,
            "seeds": seeds,
        },
    }
    worker_name = "_".join(str(seed) for seed in seeds)
    write_json(output_root / f"worker_{worker_name}_preselection.json", selection_record)

    test_index = torch.load(args.test_index, map_location="cpu", weights_only=False)
    if hasattr(test_index, "tolist"):
        test_index = test_index.tolist()
    data = load_all_data(list(test_index))
    manifest, manifest_hash = split_manifest(data)
    write_json(
        output_root / f"worker_{worker_name}_split_manifest.json",
        {"dataset_split_sha256": manifest_hash, "splits": manifest},
    )

    train_data = {
        key: data[key] for key in ("dev_tr", "dev_va", "clin_tr", "clin_va")
    }
    for seed in seeds:
        seed_dir = output_root / "common_envelope" / f"seed{seed}"
        model_path = seed_dir / "model.pt"
        metrics_path = seed_dir / "validation_metrics.json"
        if model_path.exists() and metrics_path.exists() and args.resume:
            print(f"[resume] seed {seed} already complete: {seed_dir}")
            continue
        if seed_dir.exists() and any(seed_dir.iterdir()):
            raise FileExistsError(
                f"partial or existing seed directory will not be overwritten: {seed_dir}"
            )
        seed_dir.mkdir(parents=True, exist_ok=False)
        log_path = seed_dir / "training.log"
        with log_path.open("x", encoding="utf-8") as log_handle:
            tee = Tee(sys.stdout, log_handle)
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                started = time.time()
                print(f"selection_rule={SELECTION_RULE}")
                print("held_out_test_used_for_training_or_selection=False")
                print(f"condition=common_envelope seed={seed}")
                print(f"runtime={json.dumps(runtime_metadata(), sort_keys=True)}")
                print(f"dataset_split_sha256={manifest_hash}")
                print(f"stage1_checkpoint_sha256={sha256_file(stage1)}")
                set_seed(seed)
                model = load_clinical_model(stage1)
                cfg = argparse.Namespace(
                    seed=seed,
                    epochs=args.epochs,
                    lr=args.lr,
                    der_alpha=args.der_alpha,
                    der_beta=args.der_beta,
                )
                model = train_der_plus_plus(model, train_data, cfg)
                wearable_validation = evaluate(
                    model, data["dev_va"], wearable_envelope=True
                )
                clinical_validation = evaluate(
                    model, data["clin_va"], wearable_envelope=True
                )
                save_predictions(
                    seed_dir / "predictions_wearable_validation.csv",
                    data["dev_va"],
                    wearable_validation,
                )
                save_predictions(
                    seed_dir / "predictions_clinical_validation.csv",
                    data["clin_va"],
                    clinical_validation,
                )
                temporary = seed_dir / "model.pt.tmp"
                torch.save(model.state_dict(), temporary)
                temporary.replace(model_path)
                trainable, total = count_trainable(model)
                metrics = {
                    "condition": "common_envelope",
                    "seed": seed,
                    "held_out_test_evaluated": False,
                    "selection_rule": SELECTION_RULE,
                    "selection": getattr(model, "_validation_selection", None),
                    "wearable_validation": extended_metrics(
                        wearable_validation, wearable=True
                    ),
                    "clinical_retention_validation": extended_metrics(
                        clinical_validation, wearable=False
                    ),
                    "combined_validation_r2": float(
                        wearable_validation["r2"] + clinical_validation["r2"]
                    ),
                    "combined_validation_mae_ml": float(
                        wearable_validation["mae"] + clinical_validation["mae"]
                    ),
                    "trainable_parameters": int(trainable),
                    "total_parameters": int(total),
                    "dataset_split_sha256": manifest_hash,
                    "stage1_checkpoint_sha256": sha256_file(stage1),
                    "stage2_checkpoint_sha256": sha256_file(model_path),
                    "runtime_seconds": time.time() - started,
                    "runtime": runtime_metadata(),
                    "preprocessing": preprocessing_configuration(True),
                }
                write_json(metrics_path, metrics)
                print(f"stage2_checkpoint_sha256={metrics['stage2_checkpoint_sha256']}")
                print(f"runtime_seconds={metrics['runtime_seconds']:.3f}")
                print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
