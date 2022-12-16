"""
Blocks based on https://github.com/google-research/maxim/blob/main/maxim/models/maxim.py
"""

import functools

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers

from ..layers import Resizing

Conv1x1 = functools.partial(layers.Conv2D, kernel_size=(1, 1), padding="same")


def MlpBlock(
    mlp_dim: int,
    dropout_rate: float = 0.0,
    use_bias: bool = True,
    name: str = "mlp_block",
):
    """A 1-hidden-layer MLP block, applied over the last dimension."""

    def apply(x):
        d = K.int_shape(x)[-1]
        x = layers.Dense(mlp_dim, use_bias=use_bias, name=f"{name}_Dense_0")(x)
        x = tf.nn.gelu(x, approximate=True)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(d, use_bias=use_bias, name=f"{name}_Dense_1")(x)
        return x

    return apply


def UpSampleRatio(
    num_channels: int, ratio: float, use_bias: bool = True, name: str = "upsample"
):
    """Upsample features given a ratio > 0."""

    def apply(x):
        # Following `jax.image.resize()`
        x = Resizing(
            ratio=1 / ratio,
            method="bilinear",
            antialias=True,
            name=f"{name}_resizing_{K.get_uid('Resizing')}",
        )(x)

        x = Conv1x1(filters=num_channels, use_bias=use_bias, name=f"{name}_Conv_0")(x)
        return x

    return apply
