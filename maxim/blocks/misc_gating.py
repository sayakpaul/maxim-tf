"""
Blocks based on https://github.com/google-research/maxim/blob/main/maxim/models/maxim.py
"""

import functools

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers

from ..layers import SwapAxes, TFBlockImages, TFBlockImagesByGrid, TFUnblockImages
from .block_gating import BlockGmlpLayer
from .grid_gating import GridGmlpLayer

Conv1x1 = functools.partial(layers.Conv2D, kernel_size=(1, 1), padding="same")
Conv3x3 = functools.partial(layers.Conv2D, kernel_size=(3, 3), padding="same")
ConvT_up = functools.partial(
    layers.Conv2DTranspose, kernel_size=(2, 2), strides=(2, 2), padding="same"
)
Conv_down = functools.partial(
    layers.Conv2D, kernel_size=(4, 4), strides=(2, 2), padding="same"
)


def ResidualSplitHeadMultiAxisGmlpLayer(
    block_size,
    grid_size,
    block_gmlp_factor: int = 2,
    grid_gmlp_factor: int = 2,
    input_proj_factor: int = 2,
    use_bias: bool = True,
    dropout_rate: float = 0.0,
    name: str = "residual_split_head_maxim",
):
    """The multi-axis gated MLP block."""

    def apply(x):
        shortcut = x
        n, h, w, num_channels = (
            K.int_shape(x)[0],
            K.int_shape(x)[1],
            K.int_shape(x)[2],
            K.int_shape(x)[3],
        )
        x = layers.LayerNormalization(epsilon=1e-06, name=f"{name}_LayerNorm_in")(x)

        x = layers.Dense(
            int(num_channels) * input_proj_factor,
            use_bias=use_bias,
            name=f"{name}_in_project",
        )(x)
        x = tf.nn.gelu(x, approximate=True)

        u, v = tf.split(x, 2, axis=-1)

        # GridGMLPLayer
        u = GridGmlpLayer(
            grid_size=grid_size,
            factor=grid_gmlp_factor,
            use_bias=use_bias,
            dropout_rate=dropout_rate,
            name=f"{name}_GridGmlpLayer",
        )(u)

        # BlockGMLPLayer
        v = BlockGmlpLayer(
            block_size=block_size,
            factor=block_gmlp_factor,
            use_bias=use_bias,
            dropout_rate=dropout_rate,
            name=f"{name}_BlockGmlpLayer",
        )(v)

        x = tf.concat([u, v], axis=-1)

        x = layers.Dense(
            num_channels,
            use_bias=use_bias,
            name=f"{name}_out_project",
        )(x)
        x = layers.Dropout(dropout_rate)(x)
        x = x + shortcut
        return x

    return apply


def GetSpatialGatingWeights(
    features: int,
    block_size,
    grid_size,
    input_proj_factor: int = 2,
    dropout_rate: float = 0.0,
    use_bias: bool = True,
    name: str = "spatial_gating",
):

    """Get gating weights for cross-gating MLP block."""

    def apply(x):
        n, h, w, num_channels = (
            K.int_shape(x)[0],
            K.int_shape(x)[1],
            K.int_shape(x)[2],
            K.int_shape(x)[3],
        )

        # input projection
        x = layers.LayerNormalization(epsilon=1e-06, name=f"{name}_LayerNorm_in")(x)
        x = layers.Dense(
            num_channels * input_proj_factor,
            use_bias=use_bias,
            name=f"{name}_in_project",
        )(x)
        x = tf.nn.gelu(x, approximate=True)
        u, v = tf.split(x, 2, axis=-1)

        # Get grid MLP weights
        gh, gw = grid_size
        u, phu, pwu = TFBlockImagesByGrid()(u, grid_size=(gh, gw))
        dim_u = gh * gw
        u = SwapAxes()(u, -1, -3)
        u = layers.Dense(dim_u, use_bias=use_bias, name=f"{name}_Dense_0")(u)
        u = SwapAxes()(u, -1, -3)
        u = TFUnblockImages()(u, grid_size=(gh, gw), patch_size=(phu, pwu))

        # Get Block MLP weights
        fh, fw = block_size
        v, gh, gw = TFBlockImages()(v, patch_size=(fh, fw))
        dim_v = fh * fw
        v = SwapAxes()(v, -1, -2)
        v = layers.Dense(dim_v, use_bias=use_bias, name=f"{name}_Dense_1")(v)
        v = SwapAxes()(v, -1, -2)
        v = TFUnblockImages()(v, patch_size=(fh, fw), grid_size=(gh, gw))

        x = tf.concat([u, v], axis=-1)
        x = layers.Dense(num_channels, use_bias=use_bias, name=f"{name}_out_project")(x)
        x = layers.Dropout(dropout_rate)(x)
        return x

    return apply


def CrossGatingBlock(
    features: int,
    block_size,
    grid_size,
    dropout_rate: float = 0.0,
    input_proj_factor: int = 2,
    upsample_y: bool = True,
    use_bias: bool = True,
    name: str = "cross_gating",
):

    """Cross-gating MLP block."""

    def apply(x, y):
        # Upscale Y signal, y is the gating signal.
        if upsample_y:
            y = ConvT_up(
                filters=features, use_bias=use_bias, name=f"{name}_ConvTranspose_0"
            )(y)

        x = Conv1x1(filters=features, use_bias=use_bias, name=f"{name}_Conv_0")(x)
        n, h, w, num_channels = (
            K.int_shape(x)[0],
            K.int_shape(x)[1],
            K.int_shape(x)[2],
            K.int_shape(x)[3],
        )

        y = Conv1x1(filters=num_channels, use_bias=use_bias, name=f"{name}_Conv_1")(y)

        shortcut_x = x
        shortcut_y = y

        # Get gating weights from X
        x = layers.LayerNormalization(epsilon=1e-06, name=f"{name}_LayerNorm_x")(x)
        x = layers.Dense(num_channels, use_bias=use_bias, name=f"{name}_in_project_x")(
            x
        )
        x = tf.nn.gelu(x, approximate=True)
        gx = GetSpatialGatingWeights(
            features=num_channels,
            block_size=block_size,
            grid_size=grid_size,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            name=f"{name}_SplitHeadMultiAxisGating_x",
        )(x)

        # Get gating weights from Y
        y = layers.LayerNormalization(epsilon=1e-06, name=f"{name}_LayerNorm_y")(y)
        y = layers.Dense(num_channels, use_bias=use_bias, name=f"{name}_in_project_y")(
            y
        )
        y = tf.nn.gelu(y, approximate=True)
        gy = GetSpatialGatingWeights(
            features=num_channels,
            block_size=block_size,
            grid_size=grid_size,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            name=f"{name}_SplitHeadMultiAxisGating_y",
        )(y)

        # Apply cross gating: X = X * GY, Y = Y * GX
        y = y * gx
        y = layers.Dense(num_channels, use_bias=use_bias, name=f"{name}_out_project_y")(
            y
        )
        y = layers.Dropout(dropout_rate)(y)
        y = y + shortcut_y

        x = x * gy  # gating x using y
        x = layers.Dense(num_channels, use_bias=use_bias, name=f"{name}_out_project_x")(
            x
        )
        x = layers.Dropout(dropout_rate)(x)
        x = x + y + shortcut_x  # get all aggregated signals
        return x, y

    return apply
