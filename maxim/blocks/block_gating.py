"""
Blocks based on https://github.com/google-research/maxim/blob/main/maxim/models/maxim.py
"""

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers

from ..layers import SwapAxes, TFBlockImages, TFUnblockImages


def BlockGatingUnit(use_bias: bool = True, name: str = "block_gating_unit"):
    """A SpatialGatingUnit as defined in the gMLP paper.

    The 'spatial' dim is defined as the **second last**.
    If applied on other dims, you should swapaxes first.
    """

    def apply(x):
        u, v = tf.split(x, 2, axis=-1)
        v = layers.LayerNormalization(
            epsilon=1e-06, name=f"{name}_intermediate_layernorm"
        )(v)
        n = K.int_shape(x)[-2]  # get spatial dim
        v = SwapAxes()(v, -1, -2)
        v = layers.Dense(n, use_bias=use_bias, name=f"{name}_Dense_0")(v)
        v = SwapAxes()(v, -1, -2)
        return u * (v + 1.0)

    return apply


def BlockGmlpLayer(
    block_size,
    use_bias: bool = True,
    factor: int = 2,
    dropout_rate: float = 0.0,
    name: str = "block_gmlp",
):
    """Block gMLP layer that performs local mixing of tokens."""

    def apply(x):
        n, h, w, num_channels = (
            K.int_shape(x)[0],
            K.int_shape(x)[1],
            K.int_shape(x)[2],
            K.int_shape(x)[3],
        )
        fh, fw = block_size
        x, gh, gw = TFBlockImages()(x, patch_size=(fh, fw))
        # MLP2: Local (block) mixing part, provides within-block communication.
        y = layers.LayerNormalization(epsilon=1e-06, name=f"{name}_LayerNorm")(x)
        y = layers.Dense(
            num_channels * factor,
            use_bias=use_bias,
            name=f"{name}_in_project",
        )(y)
        y = tf.nn.gelu(y, approximate=True)
        y = BlockGatingUnit(use_bias=use_bias, name=f"{name}_BlockGatingUnit")(y)
        y = layers.Dense(
            num_channels,
            use_bias=use_bias,
            name=f"{name}_out_project",
        )(y)
        y = layers.Dropout(dropout_rate)(y)
        x = x + y
        x = TFUnblockImages()(x, patch_size=(fh, fw), grid_size=(gh, gw))
        return x

    return apply
