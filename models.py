"""Model architecture definitions for 2D encrypted traffic classification.

Implements the standardized baseline 2D CNN and benchmark Large Vision Models
(ResNet-50 and Vision Transformer ViT-Base/16).
"""

from typing import Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, ops


def build_cnn(
    input_shape: Tuple[int, ...], num_classes: int = 6
) -> models.Model:
    """Build the standardized baseline 2D CNN model.

    Architecture:
      - Conv2D(32, 3x3, padding='same', ReLU)
      - MaxPooling2D(2x2)
      - Conv2D(64, 3x3, padding='same', ReLU, name='last_conv')
      - MaxPooling2D(2x2)
      - Flatten
      - Dense(128, ReLU)
      - Dropout(0.5)
      - Dense(num_classes, Softmax)
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(
        32, kernel_size=(3, 3), activation="relu", padding="same"
    )(inputs)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    conv_last = layers.Conv2D(
        64,
        kernel_size=(3, 3),
        activation="relu",
        padding="same",
        name="last_conv",
    )(x)
    if conv_last.shape[1] is not None and conv_last.shape[1] >= 2:
        x = layers.MaxPooling2D(pool_size=(2, 2))(conv_last)
    else:
        x = conv_last

    flat = layers.Flatten()(x)
    dense1 = layers.Dense(128, activation="relu")(flat)
    drop = layers.Dropout(0.5)(dense1)
    outputs = layers.Dense(num_classes, activation="softmax")(drop)

    return models.Model(inputs=inputs, outputs=outputs)


def build_resnet50(
    input_shape: Tuple[int, ...] = (224, 224, 3), num_classes: int = 6
) -> models.Model:
    """Build ResNet-50 benchmark model initialized from scratch."""
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        pooling="avg",
    )
    outputs = layers.Dense(num_classes, activation="softmax")(base.output)
    return models.Model(inputs=base.input, outputs=outputs)


class ViTPatchEmbedding(layers.Layer):
    """Patch extraction and linear projection layer for Vision Transformer."""

    def __init__(
        self, patch_size: int = 16, embed_dim: int = 768, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = layers.Conv2D(
            embed_dim,
            kernel_size=patch_size,
            strides=patch_size,
            padding="valid",
        )

    def build(self, input_shape: Tuple[int, ...]) -> None:
        h, w = input_shape[1], input_shape[2]
        num_patches = (h // self.patch_size) * (w // self.patch_size)
        self.cls_token = self.add_weight(
            shape=(1, 1, self.embed_dim),
            initializer="zeros",
            trainable=True,
            name="cls_token",
        )
        self.pos_embed = self.add_weight(
            shape=(1, num_patches + 1, self.embed_dim),
            initializer="zeros",
            trainable=True,
            name="pos_embed",
        )

    def call(self, x: tf.Tensor) -> tf.Tensor:
        batch_size = ops.shape(x)[0]
        patches = self.proj(x)
        num_patches = ops.shape(patches)[1] * ops.shape(patches)[2]
        patches = ops.reshape(
            patches, (batch_size, num_patches, self.embed_dim)
        )
        cls_tokens = ops.broadcast_to(
            self.cls_token, (batch_size, 1, self.embed_dim)
        )
        tokens = ops.concatenate([cls_tokens, patches], axis=1)
        return tokens + self.pos_embed


def build_vit(
    input_shape: Tuple[int, ...] = (224, 224, 3),
    num_classes: int = 6,
    patch_size: int = 16,
    embed_dim: int = 768,
    num_heads: int = 12,
    num_layers: int = 12,
    mlp_dim: int = 3072,
    dropout: float = 0.1,
) -> models.Model:
    """Build ViT-Base/16 benchmark model initialized from scratch."""
    inputs = layers.Input(shape=input_shape)
    x = ViTPatchEmbedding(patch_size=patch_size, embed_dim=embed_dim)(inputs)
    x = layers.Dropout(dropout)(x)

    for _ in range(num_layers):
        norm1 = layers.LayerNormalization(epsilon=1e-6)(x)
        attn = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads,
            dropout=dropout,
        )(norm1, norm1)
        x = layers.Add()([x, attn])

        norm2 = layers.LayerNormalization(epsilon=1e-6)(x)
        mlp = layers.Dense(mlp_dim, activation="gelu")(norm2)
        mlp = layers.Dropout(dropout)(mlp)
        mlp = layers.Dense(embed_dim)(mlp)
        mlp = layers.Dropout(dropout)(mlp)
        x = layers.Add()([x, mlp])

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    cls_token = layers.Lambda(lambda t: t[:, 0])(x)
    outputs = layers.Dense(num_classes, activation="softmax")(cls_token)
    return models.Model(inputs=inputs, outputs=outputs)


def adapt_input_for_lvm(x: np.ndarray) -> np.ndarray:
    """Resize single-channel 2D matrices to (224, 224, 3) for LVMs."""
    resized = tf.image.resize(x, (224, 224), method="bilinear").numpy()
    replicated = np.repeat(resized, 3, axis=-1)
    min_val = replicated.min(axis=(1, 2, 3), keepdims=True)
    max_val = replicated.max(axis=(1, 2, 3), keepdims=True)
    denom = np.where(max_val - min_val > 1e-8, max_val - min_val, 1.0)
    return (replicated - min_val) / denom
