from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model import RFNet  # noqa: E402


INPUT_NAME = "rf"
OUTPUT_NAME = "volume"
INPUT_SHAPE = (1, 7, 400)
CHANNEL_ORDER = (
    "H_physical_left",
    "H_virtual_mean_S2_S3",
    "H_physical_right",
    "S1",
    "S2",
    "S3",
    "S4",
)


class DeployModel(nn.Module):
    """Expose RFNet as a fixed, single-tensor deployment graph."""

    def __init__(self, model: RFNet):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rf_h = [x[:, i : i + 1, :] for i in range(3)]
        rf_s = [x[:, i : i + 1, :] for i in range(3, 7)]
        volume, _, _ = self.model(rf_h, rf_s)
        return volume.unsqueeze(-1)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_deploy_model(checkpoint: str | Path) -> DeployModel:
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"expected state_dict checkpoint, got {type(state).__name__}")

    model = RFNet(num_h_rf=3, num_s_rf=4, encoder_out_dim=512, dropout_p=0.5)
    model.load_state_dict(state, strict=True)
    model.eval()
    wrapper = DeployModel(model)
    wrapper.eval()
    return wrapper


def predict_torch(model: nn.Module, inputs: np.ndarray) -> np.ndarray:
    outputs = []
    with torch.inference_mode():
        for sample in inputs:
            tensor = torch.from_numpy(np.asarray(sample, dtype=np.float32)).unsqueeze(0)
            outputs.append(float(model(tensor).reshape(-1)[0]))
    return np.asarray(outputs, dtype=np.float32)


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if target.shape != prediction.shape:
        raise ValueError(f"shape mismatch: target={target.shape}, prediction={prediction.shape}")
    error = prediction - target
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    r2 = float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    true_alarm = target >= 200.0
    pred_alarm = prediction >= 200.0
    return {
        "n": int(target.size),
        "mae_ml": float(np.mean(np.abs(error))),
        "rmse_ml": float(np.sqrt(np.mean(error**2))),
        "r2": r2,
        "accuracy_within_50ml_pct": float(np.mean(np.abs(error) <= 50.0) * 100.0),
        "alarm_200ml_accuracy_pct": float(np.mean(true_alarm == pred_alarm) * 100.0),
    }


def runtime_metadata() -> dict[str, Any]:
    import onnx
    import onnxruntime as ort

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
    }


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
