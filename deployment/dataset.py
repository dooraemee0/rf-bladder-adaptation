from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataloader import FolderDatasetUnified, PreprocessTransform  # noqa: E402


SPLIT_SEED = 42
MODEL_LENGTH = 400


class DeviceAxisPreprocessTransform(PreprocessTransform):
    """Reproduce the device branch while making the Gaussian axis explicit."""

    def __init__(self, num_select: int, gaussian_axis: int):
        super().__init__(num_select=num_select)
        if gaussian_axis not in (0, 1):
            raise ValueError(f"gaussian_axis must be 0 or 1, got {gaussian_axis}")
        self.gaussian_axis = gaussian_axis

    def __call__(self, data, domain, seed=42):
        if domain != "device":
            return super().__call__(data, domain, seed)

        smoothed = gaussian_filter1d(
            data, sigma=self.sigma, axis=self.gaussian_axis
        )
        downsampled = smoothed[:, :: self.downsample_rate]
        target_len = self.default_max_len
        if downsampled.shape[1] >= target_len:
            processed = downsampled[:, :target_len]
        else:
            pad = target_len - downsampled.shape[1]
            processed = np.pad(downsampled, ((0, 0), (0, pad)), mode="edge")

        channel_mean = processed.mean(axis=1, keepdims=True)
        channel_std = processed.std(axis=1, keepdims=True) + 1e-8
        return ((processed - channel_mean) / channel_std).astype(np.float32)


def build_device_dataframe(device_root: str | Path) -> pd.DataFrame:
    root = Path(device_root)
    rows = []
    for volume in (50, 150, 300):
        for path in sorted(glob.glob(str(root / str(volume) / "*.txt"))):
            rows.append(
                {
                    "data_path": str(root),
                    "txt_path": path,
                    "domain": "device",
                    "volume": float(volume),
                    "Upright_H": "",
                    "Upright_S": "",
                }
            )
    if not rows:
        raise FileNotFoundError(
            f"no wearable files found below {root}; expected 50/, 150/, and 300/"
        )
    return pd.DataFrame(rows)


def _stratified_split(
    dataframe: pd.DataFrame, test_ratio: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    test_parts = []
    for volume in sorted(dataframe["volume"].unique()):
        group = dataframe[dataframe["volume"] == volume]
        train_idx, test_idx = train_test_split(
            group.index,
            test_size=test_ratio,
            random_state=seed,
            shuffle=True,
        )
        train_parts.append(dataframe.loc[train_idx])
        test_parts.append(dataframe.loc[test_idx])
    return (
        pd.concat(train_parts).reset_index(drop=True),
        pd.concat(test_parts).reset_index(drop=True),
    )


def device_partitions(device_root: str | Path) -> dict[str, pd.DataFrame]:
    all_data = build_device_dataframe(device_root)
    train_full, test = _stratified_split(all_data, test_ratio=0.2, seed=SPLIT_SEED)
    train, validation = _stratified_split(train_full, test_ratio=0.2, seed=SPLIT_SEED)
    return {"train": train, "validation": validation, "test": test}


def pad_tail_mean_to_400(signal: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal, dtype=np.float32)
    channels, length = signal.shape
    if length == MODEL_LENGTH:
        return signal
    if length > MODEL_LENGTH:
        return signal[:, :MODEL_LENGTH]
    tail = min(30, length)
    fill = signal[:, -tail:].mean(axis=1, keepdims=True)
    padding = np.repeat(fill, MODEL_LENGTH - length, axis=1)
    return np.concatenate((signal, padding), axis=1).astype(np.float32, copy=False)


def load_preprocessed_partition(
    device_root: str | Path,
    partition: str,
    gaussian_axis: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    partitions = device_partitions(device_root)
    if partition not in partitions:
        raise ValueError(f"partition must be one of {sorted(partitions)}, got {partition}")
    dataframe = partitions[partition].reset_index(drop=True)

    transform_h = DeviceAxisPreprocessTransform(num_select=3, gaussian_axis=gaussian_axis)
    transform_s = DeviceAxisPreprocessTransform(num_select=4, gaussian_axis=gaussian_axis)
    dataset = FolderDatasetUnified(dataframe, transform_h, transform_s)

    inputs = []
    labels = []
    sample_ids = []
    for index in range(len(dataset)):
        item = dataset[index]
        horizontal = item["horizon"].numpy()
        sagittal = item["sagittal"].numpy()
        model_input = pad_tail_mean_to_400(np.concatenate((horizontal, sagittal), axis=0))
        if model_input.shape != (7, MODEL_LENGTH):
            raise ValueError(f"unexpected input shape at index {index}: {model_input.shape}")
        inputs.append(model_input)
        labels.append(float(item["volume_gt"]))
        sample_ids.append(Path(dataframe.iloc[index]["txt_path"]).name)

    return (
        np.stack(inputs).astype(np.float32, copy=False),
        np.asarray(labels, dtype=np.float32),
        sample_ids,
    )


def select_balanced_calibration(
    inputs: np.ndarray,
    labels: np.ndarray,
    max_per_volume: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    selected = []
    for volume in (50.0, 150.0, 300.0):
        indices = np.flatnonzero(labels == volume)
        if max_per_volume is not None:
            indices = indices[:max_per_volume]
        selected.extend(indices.tolist())
    selected_array = np.asarray(selected, dtype=np.int64)
    return inputs[selected_array], labels[selected_array]
