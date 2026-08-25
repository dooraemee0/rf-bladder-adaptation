from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch


SEEDS = (1, 2, 3, 42)
SELECTION_RULE = (
    "Compare preprocessing conditions by the four-seed mean of wearable-validation "
    "R2 plus clinical-retention-validation R2. Break an exact tie by the lower "
    "four-seed mean of wearable-validation MAE plus clinical-retention-validation "
    "MAE. Held-out test metrics are not used for preprocessing selection."
)
CHANNEL_ORDER = (
    "H_left",
    "H_virtual=mean(S2,S3)",
    "H_right",
    "S1",
    "S2",
    "S3",
    "S4",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def runtime_metadata() -> dict:
    metadata = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        metadata["cuda_device"] = torch.cuda.get_device_name(torch.cuda.current_device())
    return metadata


def git_state(project_root: str | Path) -> dict:
    root = Path(project_root)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"], cwd=root, text=True, capture_output=True, check=True
    ).stdout.splitlines()
    return {"commit": commit, "working_tree_status": status}


def preprocessing_configuration(wearable_envelope: bool) -> dict:
    return {
        "physical_channels": 6,
        "amplitude_divisor": 1700.0,
        "tgc_coefficients": [0.15, 0.4],
        "sample_rate_hz": 20_000_000,
        "butterworth_order": 7,
        "bandpass_hz": [1_500_000, 4_500_000],
        "clip": [-0.1, 0.1],
        "raw_crop": [300, 3000],
        "virtual_channel": "mean(S2,S3) before optional envelope detection",
        "channel_order": list(CHANNEL_ORDER),
        "wearable_hilbert_envelope": bool(wearable_envelope),
        "hilbert_axis": 1 if wearable_envelope else None,
        "gaussian_sigma": 1,
        "gaussian_axis": 0,
        "downsample_factor": 10,
        "retained_samples": 220,
        "standardization": "per channel over 220 samples",
        "padding": "channel final-30-sample mean from 220 to 400",
    }


def dataframe_ids(dataframe: pd.DataFrame) -> list[str]:
    ids = []
    for _, row in dataframe.reset_index(drop=True).iterrows():
        if row["domain"] == "device":
            ids.append(f"device:{Path(row['txt_path']).name}:{float(row['volume']):g}")
        else:
            ids.append(
                f"clinical:{row.get('original_index', '')}:{row.get('Upright_H', '')}:"
                f"{row.get('Upright_S', '')}:{float(row['volume']):g}"
            )
    return ids


def split_manifest(data: dict[str, pd.DataFrame]) -> tuple[dict, str]:
    manifest = {name: dataframe_ids(frame) for name, frame in sorted(data.items())}
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return manifest, hashlib.sha256(encoded).hexdigest()


def extended_metrics(result: dict, wearable: bool) -> dict:
    pred = np.asarray(result["pred"], dtype=np.float64)
    gt = np.asarray(result["gt"], dtype=np.float64)
    output = {
        "n": int(len(gt)),
        "mae_ml": float(result["mae"]),
        "rmse_ml": float(result["rmse"]),
        "r2": float(result["r2"]),
        "accuracy_within_50ml_pct": float(result["acc50"]),
    }
    if wearable:
        truth = gt >= 200.0
        positive = pred >= 200.0
        tp = int(np.sum(truth & positive))
        tn = int(np.sum(~truth & ~positive))
        fp = int(np.sum(~truth & positive))
        fn = int(np.sum(truth & ~positive))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        output.update(
            {
                "alarm_200ml_accuracy_pct": 100.0 * (tp + tn) / len(gt),
                "alarm_200ml_precision": precision,
                "alarm_200ml_recall": recall,
                "alarm_200ml_f1": f1,
                "alarm_confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
            }
        )
    return output


def per_volume_predictions(result: dict) -> dict:
    pred = np.asarray(result["pred"])
    gt = np.asarray(result["gt"])
    return {
        str(int(volume)): {
            "n": int(np.sum(gt == volume)),
            "prediction_mean_ml": float(pred[gt == volume].mean()),
            "prediction_sd_ml": float(pred[gt == volume].std(ddof=1)),
        }
        for volume in sorted(np.unique(gt))
    }


def save_predictions(path: str | Path, dataframe: pd.DataFrame, result: dict) -> None:
    output = pd.DataFrame(
        {
            "sample_id": dataframe_ids(dataframe),
            "target_ml": result["gt"],
            "prediction_ml": result["pred"],
        }
    )
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(path, index=False)


def write_json(path: str | Path, value: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
