from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, filtfilt, hilbert

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader import FolderDatasetUnified, PreprocessTransform
from deployment.envelope_ablation_utils import CHANNEL_ORDER, runtime_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Numerical tests for wearable envelope ablation")
    parser.add_argument("--device-data-root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def manual_transform(data: np.ndarray, wearable_envelope: bool) -> np.ndarray:
    source = np.abs(hilbert(data, axis=1)) if wearable_envelope else data
    processed = gaussian_filter1d(source, sigma=1, axis=0)[:, ::10][:, :220]
    mean = processed.mean(axis=1, keepdims=True)
    std = processed.std(axis=1, keepdims=True) + 1e-8
    return ((processed - mean) / std).astype(np.float32)


def manual_device_arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    raw = path.read_text().replace("'", "").replace(" ", "").strip().split(",")
    values = np.asarray([float(value) for value in raw if value], dtype=np.float32)
    length = len(values) // 6
    data = values[: 6 * length].reshape(6, length) / 1700.0
    fs = 20e6
    distance = np.arange(length, dtype=np.float32) / fs * 1500.0 / 2.0 * 1e3
    tgc = np.concatenate(
        [
            np.exp(0.15 * distance[: length // 2] / 10.0),
            np.exp(0.4 * distance[length // 2 :] / 10.0),
        ]
    ).astype(np.float32)
    data *= tgc[None]
    b, a = butter(N=7, Wn=[1.5e6, 4.5e6], btype="band", fs=fs)
    for channel in range(6):
        data[channel] = filtfilt(b, a, data[channel]).astype(np.float32)
    data = np.clip(data, -0.1, 0.1)[:, 300:3000]
    expected_h = np.stack((data[4], np.mean(data[1:3], axis=0), data[5])).astype(np.float32)
    expected_s = data[:4].astype(np.float32)
    return expected_h, expected_s


def main() -> None:
    args = parse_args()
    rng = np.random.RandomState(20260808)
    synthetic = rng.normal(size=(3, 2700)).astype(np.float32)

    legacy = PreprocessTransform(num_select=3, wearable_envelope=False)
    common_envelope = PreprocessTransform(num_select=3, wearable_envelope=True)
    legacy_output = legacy(synthetic, domain="device")
    envelope_output = common_envelope(synthetic, domain="device")
    legacy_reference = manual_transform(synthetic, wearable_envelope=False)
    envelope_reference = manual_transform(synthetic, wearable_envelope=True)

    false_max_difference = float(np.max(np.abs(legacy_output - legacy_reference)))
    envelope_max_difference = float(np.max(np.abs(envelope_output - envelope_reference)))
    wrong_axis = np.abs(hilbert(synthetic, axis=0))
    wrong_axis_output = manual_transform(wrong_axis, wearable_envelope=False)
    axis_discrimination = float(np.max(np.abs(envelope_output - wrong_axis_output)))
    assert false_max_difference == 0.0
    assert envelope_max_difference == 0.0
    assert axis_discrimination > 1e-3

    clinical = rng.normal(size=(12, 4000)).astype(np.float32)
    clinical_default = PreprocessTransform(num_select=3, seed_offset=0)(
        clinical, domain="SNUH_uncut", seed=42
    )
    clinical_option = PreprocessTransform(
        num_select=3, seed_offset=0, wearable_envelope=True
    )(clinical, domain="SNUH_uncut", seed=42)
    clinical_max_difference = float(np.max(np.abs(clinical_default - clinical_option)))
    assert clinical_max_difference == 0.0

    device_files = sorted(Path(args.device_data_root).glob("50/*.txt"))
    if not device_files:
        raise FileNotFoundError("no device file found for channel-order test")
    frame = pd.DataFrame(
        [
            {
                "data_path": str(Path(args.device_data_root)),
                "txt_path": str(device_files[0]),
                "domain": "device",
                "volume": 50.0,
                "Upright_H": "",
                "Upright_S": "",
            }
        ]
    )
    dataset = FolderDatasetUnified(frame)
    actual_h, actual_s = dataset._load_device(frame.iloc[0], item_seed=0)
    expected_h, expected_s = manual_device_arrays(device_files[0])
    channel_order_difference = max(
        float(np.max(np.abs(actual_h - expected_h))),
        float(np.max(np.abs(actual_s - expected_s))),
    )
    assert channel_order_difference == 0.0

    result = {
        "passed": True,
        "wearable_envelope_false_vs_legacy_max_abs_difference": false_max_difference,
        "wearable_envelope_true_vs_manual_time_axis_max_abs_difference": envelope_max_difference,
        "time_axis_vs_wrong_channel_axis_max_abs_difference": axis_discrimination,
        "clinical_pipeline_option_invariance_max_abs_difference": clinical_max_difference,
        "channel_order_max_abs_difference": channel_order_difference,
        "channel_order": list(CHANNEL_ORDER),
        "real_sample": str(device_files[0]),
        "runtime": runtime_metadata(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
