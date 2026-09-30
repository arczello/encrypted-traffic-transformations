"""Evaluation pipeline for 2D encrypted traffic classification schemes.

Provides training and evaluation routines for 1D-to-2D data transformation
schemes using standardized CNN and benchmark Large Vision Models
(ResNet-50, ViT-Base/16).
"""

import time
from typing import Dict
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import tensorflow as tf

from models import adapt_input_for_lvm, build_cnn, build_resnet50, build_vit
from transformations import apply_transformation_batch


def set_seed(seed: int = 42) -> None:
    """Set global random seeds for deterministic reproduction."""
    np.random.seed(seed)
    tf.random.set_seed(seed)


def evaluate_scheme(
    raw_x: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    scheme: str,
    model_type: str = "cnn",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    patience: int = 10,
    seed: int = 42,
) -> Dict[str, float]:
    """Transform, train, and evaluate a single representation scheme.

    Args:
        raw_x: Pre-extracted 1D intensity sequences of shape (N, T).
        labels: Class label indices of shape (N,).
        num_classes: Number of distinct classification classes.
        scheme: Name of transformation scheme from transformations.SCHEMES.
        model_type: Model architecture ('cnn', 'resnet50', or 'vit').
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        lr: Learning rate for Adam optimizer.
        weight_decay: Weight decay factor.
        patience: Early stopping patience (validation loss).
        seed: Random state seed.

    Returns:
        Dictionary containing Macro F1 scores, runtime, and parameter counts.
    """
    if raw_x.ndim != 2:
        raise ValueError(
            f"raw_x must have shape (N, T), received shape {raw_x.shape}."
        )
    if labels.ndim != 1:
        raise ValueError(
            f"labels must have shape (N,), received shape {labels.shape}."
        )
    if len(raw_x) != len(labels):
        raise ValueError("Length mismatch between raw_x and labels.")

    t_start = time.time()
    x_transformed = apply_transformation_batch(raw_x, scheme)[..., np.newaxis]
    trans_time = time.time() - t_start

    if model_type in ["resnet50", "vit"]:
        x_transformed = adapt_input_for_lvm(x_transformed)

    # Stratified partition: 70% train, 15% validation, 15% test
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x_transformed,
        labels,
        test_size=0.15,
        stratify=labels,
        random_state=seed,
    )
    val_ratio = 0.15 / 0.85
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val,
        y_train_val,
        test_size=val_ratio,
        stratify=y_train_val,
        random_state=seed,
    )

    input_shape = x_train.shape[1:]
    if model_type == "cnn":
        model = build_cnn(input_shape, num_classes=num_classes)
    elif model_type == "resnet50":
        model = build_resnet50(input_shape, num_classes=num_classes)
    elif model_type == "vit":
        model = build_vit(input_shape, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr,
        weight_decay=weight_decay if weight_decay > 0 else None,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
    )
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        min_delta=1e-3,
        restore_best_weights=True,
        verbose=0,
    )

    t_train_start = time.time()
    model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0,
    )
    train_time = time.time() - t_train_start

    # Model weights evaluated are from best validation checkpoint
    preds = np.argmax(model.predict(x_test, verbose=0), axis=1)

    global_f1 = f1_score(y_test, preds, average="macro", zero_division=0)

    # Inter-bin: Video (0, 1, 2) vs Non-Video (3, 4, 5)
    y_test_bin = np.isin(y_test, [0, 1, 2]).astype(int)
    preds_bin = np.isin(preds, [0, 1, 2]).astype(int)
    inter_bin_f1 = f1_score(
        y_test_bin, preds_bin, average="macro", zero_division=0
    )

    # Intra-video F1 (classes 0, 1, 2)
    v_mask = np.isin(y_test, [0, 1, 2])
    intra_video_f1 = (
        f1_score(
            y_test[v_mask], preds[v_mask], average="macro", zero_division=0
        )
        if np.any(v_mask)
        else 0.0
    )

    # Intra-non-video F1 (classes 3, 4, 5)
    nv_mask = np.isin(y_test, [3, 4, 5])
    intra_nv_f1 = (
        f1_score(
            y_test[nv_mask], preds[nv_mask], average="macro", zero_division=0
        )
        if np.any(nv_mask)
        else 0.0
    )

    return {
        "scheme": scheme,
        "global_f1": float(global_f1),
        "inter_bin_f1": float(inter_bin_f1),
        "intra_video_f1": float(intra_video_f1),
        "intra_nv_f1": float(intra_nv_f1),
        "trans_time": float(trans_time),
        "train_time": float(train_time),
        "model_params": int(model.count_params()),
    }
