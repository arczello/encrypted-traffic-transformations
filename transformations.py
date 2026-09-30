"""Data transformation schemes for 1D-to-2D encrypted traffic classification.

This module implements nine 2D data transformation schemes categorized into:
1. Mathematical time-series encodings: GASF, GADF, MTF, RP.
2. Statistical and structural feature-time representations:
   Direct, Differential, Z-Score, Feature-Grouped, and Multi-Scale.
"""

from typing import List, Tuple
import numpy as np
from pyts.image import (
    GramianAngularField,
    MarkovTransitionField,
    RecurrencePlot,
)
from sklearn.preprocessing import StandardScaler

SCHEMES: List[str] = [
    "GASF",
    "GADF",
    "MTF",
    "RP",
    "Direct",
    "Differential",
    "Z-Score",
    "Feature-Grouped",
    "Multi-Scale",
]


def compute_basic_features(window: np.ndarray) -> np.ndarray:
    """Compute 5 basic statistical features for a 1D window.

    Order: [Mean, Max, Min, Std, Median].
    """
    return np.array(
        [
            np.mean(window),
            np.max(window),
            np.min(window),
            np.std(window),
            np.median(window),
        ],
        dtype=np.float32,
    )


def extract_feature_time_matrix(
    series: np.ndarray, window_size: int = 10, stride: int = 5
) -> np.ndarray:
    """Transform a single 1D traffic intensity series into a 5 x W matrix."""
    length = len(series)
    if length < window_size:
        raise ValueError(
            f"Series length {length} must be >= window size {window_size}."
        )
    num_windows = (length - window_size) // stride + 1
    matrix = np.zeros((5, num_windows), dtype=np.float32)
    for idx in range(num_windows):
        start = idx * stride
        window = series[start : start + window_size]
        matrix[:, idx] = compute_basic_features(window)
    return matrix


def extract_feature_time_matrices_batch(
    x_raw: np.ndarray, window_size: int = 10, stride: int = 5
) -> np.ndarray:
    """Extract 5 x W feature-time matrices for a batch of 1D series.

    Args:
        x_raw: Array of shape (N, T).
        window_size: Width of sliding window.
        stride: Temporal step size.

    Returns:
        Array of shape (N, 5, num_windows).
    """
    samples, time_len = x_raw.shape
    num_windows = (time_len - window_size) // stride + 1
    out = np.zeros((samples, 5, num_windows), dtype=np.float32)
    for idx in range(num_windows):
        start = idx * stride
        window = x_raw[:, start : start + window_size]
        out[:, 0, idx] = np.mean(window, axis=1)
        out[:, 1, idx] = np.max(window, axis=1)
        out[:, 2, idx] = np.min(window, axis=1)
        out[:, 3, idx] = np.std(window, axis=1)
        out[:, 4, idx] = np.median(window, axis=1)
    return out


def apply_transformation_batch(
    x_raw: np.ndarray,
    scheme: str,
    window_size: int = 10,
    stride: int = 5,
) -> np.ndarray:
    """Apply one of the 9 data transformation schemes to a batch of flows.

    Args:
        x_raw: Raw 1D intensity sequences of shape (N, T).
        scheme: Name of the transformation scheme.
        window_size: Window size for statistical feature extraction.
        stride: Stride for sliding window.

    Returns:
        Transformed 2D representation batch with shape (N, H, W).
    """
    if scheme == "GASF":
        transformer = GramianAngularField(method="summation")
        return transformer.fit_transform(x_raw)

    if scheme == "GADF":
        transformer = GramianAngularField(method="difference")
        return transformer.fit_transform(x_raw)

    if scheme == "MTF":
        transformer = MarkovTransitionField(n_bins=8)
        return transformer.fit_transform(x_raw)

    if scheme == "RP":
        transformer = RecurrencePlot(threshold="point", percentage=20)
        return transformer.fit_transform(x_raw)

    f_batch = extract_feature_time_matrices_batch(
        x_raw, window_size=window_size, stride=stride
    )

    if scheme == "Direct":
        return f_batch

    if scheme == "Differential":
        t_diff = np.diff(f_batch, axis=2)
        pad = np.zeros(
            (f_batch.shape[0], f_batch.shape[1], 1), dtype=np.float32
        )
        return np.concatenate([pad, t_diff], axis=2)

    if scheme == "Z-Score":
        samples, feats, windows = f_batch.shape
        reshaped = f_batch.reshape(-1, windows)
        normed = StandardScaler().fit_transform(reshaped.T).T
        return normed.reshape(samples, feats, windows)

    if scheme == "Feature-Grouped":
        # Permute rows to [Mean, Median, Std, Max, Min]
        # Aligns central tendency and dynamic range adjacently
        return f_batch[:, [0, 4, 3, 1, 2], :]

    if scheme == "Multi-Scale":
        t_len = x_raw.shape[1]
        f1 = extract_feature_time_matrices_batch(
            x_raw, window_size=window_size, stride=stride
        )
        f2 = extract_feature_time_matrices_batch(
            x_raw, window_size=min(window_size * 2, t_len - 1), stride=stride
        )
        min_w = min(f1.shape[2], f2.shape[2])
        return np.concatenate([f1[:, :, :min_w], f2[:, :, :min_w]], axis=1)

    raise ValueError(f"Unknown transformation scheme: {scheme}")
