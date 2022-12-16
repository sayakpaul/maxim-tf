"""
Blocks based on https://github.com/google-research/maxim/blob/main/maxim/models/maxim.py
"""

import functools

import tensorflow as tf
from tensorflow.keras import layers

from .others import MlpBlock

Conv3x3 = functools.partial(layers.Conv2D, kernel_size=(3, 3), padding="same")
Conv1x1 = functools.partial(layers.Conv2D, kernel_size=(1, 1), padding="same")


def CALayer(
    num_channels: int,
    reduction: int = 4,
    use_bias: bool = True,
    name: str = "channel_attention",
):
    """Squeeze-and-excitation block for channel attention.

    ref: https://arxiv.org/abs/1709.01507
    """

    def apply(x):
        # 2D global average pooling
        y = layers.GlobalAvgPool2D(keepdims=True)(x)
        # Squeeze (in Squeeze-Excitation)
        y = Conv1x1(
            filters=num_channels // reduction, use_bias=use_bias, name=f"{name}_Conv_0"
        )(y)
        y = tf.nn.relu(y)
        # Excitation (in Squeeze-Excitation)
        y = Conv1x1(filters=num_channels, use_bias=use_bias, name=f"{name}_Conv_1")(y)
        y = tf.nn.sigmoid(y)
        return x * y

    return apply


def RCAB(
    num_channels: int,
    reduction: int = 4,
    lrelu_slope: float = 0.2,
    use_bias: bool = True,
    name: str = "residual_ca",
):
    """Residual channel attention block. Contains LN,Conv,lRelu,Conv,SELayer."""

    def apply(x):
        shortcut = x
        x = layers.LayerNormalization(epsilon=1e-06, name=f"{name}_LayerNorm")(x)
        x = Conv3x3(filters=num_channels, use_bias=use_bias, name=f"{name}_conv1")(x)
        x = tf.nn.leaky_relu(x, alpha=lrelu_slope)
        x = Conv3x3(filters=num_channels, use_bias=use_bias, name=f"{name}_conv2")(x)
        x = CALayer(
            num_channels=num_channels,
            reduction=reduction,
            use_bias=use_bias,
            name=f"{name}_channel_attention",
        )(x)
        return x + shortcut

    return apply


def RDCAB(
    num_channels: int,
    reduction: int = 16,
    use_bias: bool = True,
    dropout_rate: float = 0.0,
    name: str = "rdcab",
):
    """Residual dense channel attention block. Used in Bottlenecks."""

    def apply(x):
        y = layers.LayerNormalization(epsilon=1e-06, name=f"{name}_LayerNorm")(x)
        y = MlpBlock(
            mlp_dim=num_channels,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            name=f"{name}_channel_mixing",
        )(y)
        y = CALayer(
            num_channels=num_channels,
            reduction=reduction,
            use_bias=use_bias,
            name=f"{name}_channel_attention",
        )(y)
        x = x + y
        return x

    return apply


def SAM(
    num_channels: int,
    output_channels: int = 3,
    use_bias: bool = True,
    name: str = "sam",
):

    """Supervised attention module for multi-stage training.

    Introduced by MPRNet [CVPR2021]: https://github.com/swz30/MPRNet
    """

    def apply(x, x_image):
        """Apply the SAM module to the input and num_channels.
        Args:
          x: the output num_channels from UNet decoder with shape (h, w, c)
          x_image: the input image with shape (h, w, 3)
        Returns:
          A tuple of tensors (x1, image) where (x1) is the sam num_channels used for the
            next stage, and (image) is the output restored image at current stage.
        """
        # Get num_channels
        x1 = Conv3x3(filters=num_channels, use_bias=use_bias, name=f"{name}_Conv_0")(x)

        # Output restored image X_s
        if output_channels == 3:
            image = (
                Conv3x3(
                    filters=output_channels, use_bias=use_bias, name=f"{name}_Conv_1"
                )(x)
                + x_image
            )
        else:
            image = Conv3x3(
                filters=output_channels, use_bias=use_bias, name=f"{name}_Conv_1"
            )(x)

        # Get attention maps for num_channels
        x2 = tf.nn.sigmoid(
            Conv3x3(filters=num_channels, use_bias=use_bias, name=f"{name}_Conv_2")(
                image
            )
        )

        # Get attended feature maps
        x1 = x1 * x2

        # Residual connection
        x1 = x1 + x
        return x1, image

    return apply
