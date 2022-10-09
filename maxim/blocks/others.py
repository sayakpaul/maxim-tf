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
        n, h, w, c = (
            K.int_shape(x)[0],
            K.int_shape(x)[1],
            K.int_shape(x)[2],
            K.int_shape(x)[3],
        )

        # Following `jax.image.resize()`
        x = Resizing(
            height=tf.cast(h * ratio, tf.int32),
            width=tf.cast(w * ratio, tf.int32),
            method="bilinear",
            antialias=True,
        )(x)

        x = Conv1x1(filters=num_channels, use_bias=use_bias, name=f"{name}_Conv_0")(x)
        return x

    return apply
