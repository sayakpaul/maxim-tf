import functools

from tensorflow.keras import layers

from .attentions import RDCAB
from .misc_gating import ResidualSplitHeadMultiAxisGmlpLayer

Conv1x1 = functools.partial(layers.Conv2D, kernel_size=(1, 1), padding="same")


def BottleneckBlock(
    features: int,
    block_size,
    grid_size,
    num_groups: int = 1,
    block_gmlp_factor: int = 2,
    grid_gmlp_factor: int = 2,
    input_proj_factor: int = 2,
    channels_reduction: int = 4,
    dropout_rate: float = 0.0,
    use_bias: bool = True,
    name: str = "bottleneck_block",
):
    """The bottleneck block consisting of multi-axis gMLP block and RDCAB."""

    def apply(x):
        # input projection
        x = Conv1x1(filters=features, use_bias=use_bias, name=f"{name}_input_proj")(x)
        shortcut_long = x

        for i in range(num_groups):
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
            # Channel-mixing part, which provides within-patch communication.
            x = RDCAB(
                num_channels=features,
                reduction=channels_reduction,
                use_bias=use_bias,
                name=f"{name}_channel_attention_block_1_{i}",
            )(x)

        # long skip-connect
        x = x + shortcut_long
        return x

    return apply
