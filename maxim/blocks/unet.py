import functools

import tensorflow as tf
from tensorflow.keras import layers

from .attentions import RCAB
from .misc_gating import CrossGatingBlock, ResidualSplitHeadMultiAxisGmlpLayer

Conv1x1 = functools.partial(layers.Conv2D, kernel_size=(1, 1), padding="same")
Conv3x3 = functools.partial(layers.Conv2D, kernel_size=(3, 3), padding="same")
ConvT_up = functools.partial(
    layers.Conv2DTranspose, kernel_size=(2, 2), strides=(2, 2), padding="same"
)
Conv_down = functools.partial(
    layers.Conv2D, kernel_size=(4, 4), strides=(2, 2), padding="same"
)


def UNetEncoderBlock(
    num_channels: int,
    block_size,
    grid_size,
    num_groups: int = 1,
    lrelu_slope: float = 0.2,
    block_gmlp_factor: int = 2,
    grid_gmlp_factor: int = 2,
    input_proj_factor: int = 2,
    channels_reduction: int = 4,
    dropout_rate: float = 0.0,
    downsample: bool = True,
    use_global_mlp: bool = True,
    use_bias: bool = True,
    use_cross_gating: bool = False,
    name: str = "unet_encoder",
):
    """Encoder block in MAXIM."""

    def apply(x, skip=None, enc=None, dec=None):
        if skip is not None:
            x = tf.concat([x, skip], axis=-1)

        # convolution-in
        x = Conv1x1(filters=num_channels, use_bias=use_bias)(x)
        shortcut_long = x

        for i in range(num_groups):
            if use_global_mlp:
                x = ResidualSplitHeadMultiAxisGmlpLayer(
                    grid_size=grid_size,
                    block_size=block_size,
                    grid_gmlp_factor=grid_gmlp_factor,
                    block_gmlp_factor=block_gmlp_factor,
                    input_proj_factor=input_proj_factor,
                    use_bias=use_bias,
                    dropout_rate=dropout_rate,
                    name=f"{name}_SplitHeadMultiAxisGmlpLayer_{i}",
                )(x)
            x = RCAB(
                num_channels=num_channels,
                reduction=channels_reduction,
                lrelu_slope=lrelu_slope,
                use_bias=use_bias,
                name=f"{name}_channel_attention_block_1{i}",
            )(x)

        x = x + shortcut_long

        if enc is not None and dec is not None:
            assert use_cross_gating
            x, _ = CrossGatingBlock(
                features=num_channels,
                block_size=block_size,
                grid_size=grid_size,
                dropout_rate=dropout_rate,
                input_proj_factor=input_proj_factor,
                upsample_y=False,
                use_bias=use_bias,
                name=f"{name}_cross_gating_block",
            )(x, enc + dec)

        if downsample:
            x_down = Conv_down(filters=num_channels, use_bias=use_bias)(x)
            return x_down, x
        else:
            return x

    return apply


def UNetDecoderBlock(
    num_channels: int,
    block_size,
    grid_size,
    num_groups: int = 1,
    lrelu_slope: float = 0.2,
    block_gmlp_factor: int = 2,
    grid_gmlp_factor: int = 2,
    input_proj_factor: int = 2,
    channels_reduction: int = 4,
    dropout_rate: float = 0.0,
    downsample: bool = True,
    use_global_mlp: bool = True,
    use_bias: bool = True,
    name: str = "unet_decoder",
):

    """Decoder block in MAXIM."""

    def apply(x, bridge=None):
        x = ConvT_up(filters=num_channels, use_bias=use_bias)(x)
        x = UNetEncoderBlock(
            num_channels=num_channels,
            num_groups=num_groups,
            lrelu_slope=lrelu_slope,
            block_size=block_size,
            grid_size=grid_size,
            block_gmlp_factor=block_gmlp_factor,
            grid_gmlp_factor=grid_gmlp_factor,
            channels_reduction=channels_reduction,
            use_global_mlp=use_global_mlp,
            dropout_rate=dropout_rate,
            downsample=False,
            use_bias=use_bias,
            name=f"{name}_unet_encoder",
        )(x, skip=bridge)

        return x

    return apply
