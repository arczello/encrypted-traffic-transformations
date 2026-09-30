"""Explainability pipeline: Grad-CAM and Attention-to-Burst (AtB) analysis.

Computes class activation heatmaps via Grad-CAM, performs spatial
upsampling to input matrix resolution, and quantifies attention allocation
on active download bursts versus idle periods (AtB metric).
"""

from typing import Optional, Tuple
import numpy as np
from scipy.ndimage import zoom
import tensorflow as tf
from tensorflow.keras import models


def compute_gradcam_heatmap(
    model: models.Model,
    input_sample: np.ndarray,
    last_conv_layer_name: str = "last_conv",
    class_idx: Optional[int] = None,
) -> np.ndarray:
    """Compute normalized 2D Grad-CAM saliency heatmap for an input sample.

    Args:
        model: Trained Keras model with named convolutional layer.
        input_sample: Input tensor with shape (1, H, W, 1).
        last_conv_layer_name: Name of target convolutional layer.
        class_idx: Target class index. If None, uses top predicted class.

    Returns:
        Normalized 2D heatmap of shape (H_conv, W_conv).
    """
    if input_sample.ndim != 4:
        raise ValueError(
            f"input_sample must be 4D (1, H, W, C), got {input_sample.shape}."
        )

    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([input_sample])
        if class_idx is None:
            class_idx = int(tf.argmax(predictions[0]))
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU rectification and normalization
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    return heatmap.numpy()


def upsample_heatmap(
    heatmap: np.ndarray, target_shape: Tuple[int, int]
) -> np.ndarray:
    """Upsample activation heatmap to match original input matrix dimensions."""
    zoom_factors = (
        target_shape[0] / heatmap.shape[0],
        target_shape[1] / heatmap.shape[1],
    )
    return zoom(heatmap, zoom_factors, order=1)


def compute_atb_ratio(
    raw_intensity: np.ndarray,
    temporal_attention: np.ndarray,
    noise_gate: float = 1e-5,
) -> float:
    """Compute the Attention-to-Burst (AtB) ratio (%).

    Calculates the proportion of Grad-CAM attention directed at active
    transmission bursts versus idle periods:
        AtB = sum(attention[active]) / sum(attention[all]) * 100

    Args:
        raw_intensity: 1D raw throughput/intensity series.
        temporal_attention: 1D temporal attention profile.
        noise_gate: Minimum intensity threshold for active burst.

    Returns:
        Percentage of attention allocated to active bursts (0-100%).
    """
    if raw_intensity.ndim != 1:
        raise ValueError(
            f"raw_intensity must be 1D, got shape {raw_intensity.shape}."
        )
    if temporal_attention.ndim != 1:
        raise ValueError(
            f"temporal_attention must be 1D, got {temporal_attention.shape}."
        )

    # Interpolate temporal attention if lengths differ
    if len(temporal_attention) != len(raw_intensity):
        x_att = np.linspace(0, 1, len(temporal_attention))
        x_raw = np.linspace(0, 1, len(raw_intensity))
        temporal_attention = np.interp(x_raw, x_att, temporal_attention)

    active_mask = raw_intensity > noise_gate
    total_mass = np.sum(temporal_attention)
    if total_mass == 0:
        return 0.0

    burst_mass = np.sum(temporal_attention[active_mask])
    return float((burst_mass / total_mass) * 100.0)
