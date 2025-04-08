"""
MAXIM based on https://github.com/google-research/maxim/blob/main/maxim/models/maxim.py
"""

import functools

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers

from .blocks.attentions import SAM
from .blocks.bottleneck import BottleneckBlock
from .blocks.misc_gating import CrossGatingBlock
from .blocks.others import UpSampleRatio
from .blocks.unet import UNetDecoderBlock, UNetEncoderBlock
from .layers import Resizing

Conv1x1 = functools.partial(layers.Conv2D, kernel_size=(1, 1), padding="same")
Conv3x3 = functools.partial(layers.Conv2D, kernel_size=(3, 3), padding="same")
ConvT_up = functools.partial(
    layers.Conv2DTranspose, kernel_size=(2, 2), strides=(2, 2), padding="same"
)
Conv_down = functools.partial(
    layers.Conv2D, kernel_size=(4, 4), strides=(2, 2), padding="same"
)


def MAXIM(
    features: int = 64,
    depth: int = 3,
    num_stages: int = 2,
    num_groups: int = 1,
    use_bias: bool = True,
    num_supervision_scales: int = 1,
    lrelu_slope: float = 0.2,
    use_global_mlp: bool = True,
    use_cross_gating: bool = True,
    high_res_stages: int = 2,
    block_size_hr=(16, 16),
    block_size_lr=(8, 8),
    grid_size_hr=(16, 16),
    grid_size_lr=(8, 8),
    num_bottleneck_blocks: int = 1,
    block_gmlp_factor: int = 2,
    grid_gmlp_factor: int = 2,
    input_proj_factor: int = 2,
    channels_reduction: int = 4,
    num_outputs: int = 3,
    dropout_rate: float = 0.0,
):
    """The MAXIM model function with multi-stage and multi-scale supervision.

    For more model details, please check the CVPR paper:
    MAXIM: MUlti-Axis MLP for Image Processing (https://arxiv.org/abs/2201.02973)

    Attributes:
      features: initial hidden dimension for the input resolution.
      depth: the number of downsampling depth for the model.
      num_stages: how many stages to use. It will also affects the output list.
      num_groups: how many blocks each stage contains.
      use_bias: whether to use bias in all the conv/mlp layers.
      num_supervision_scales: the number of desired supervision scales.
      lrelu_slope: the negative slope parameter in leaky_relu layers.
      use_global_mlp: whether to use the multi-axis gated MLP block (MAB) in each
        layer.
      use_cross_gating: whether to use the cross-gating MLP block (CGB) in the
        skip connections and multi-stage feature fusion layers.
      high_res_stages: how many stages are specificied as high-res stages. The
        rest (depth - high_res_stages) are called low_res_stages.
      block_size_hr: the block_size parameter for high-res stages.
      block_size_lr: the block_size parameter for low-res stages.
      grid_size_hr: the grid_size parameter for high-res stages.
      grid_size_lr: the grid_size parameter for low-res stages.
      num_bottleneck_blocks: how many bottleneck blocks.
      block_gmlp_factor: the input projection factor for block_gMLP layers.
      grid_gmlp_factor: the input projection factor for grid_gMLP layers.
      input_proj_factor: the input projection factor for the MAB block.
      channels_reduction: the channel reduction factor for SE layer.
      num_outputs: the output channels.
      dropout_rate: Dropout rate.

    Returns:
      The output contains a list of arrays consisting of multi-stage multi-scale
      outputs. For example, if num_stages = num_supervision_scales = 3 (the
      model used in the paper), the output specs are: outputs =
      [[output_stage1_scale1, output_stage1_scale2, output_stage1_scale3],
       [output_stage2_scale1, output_stage2_scale2, output_stage2_scale3],
       [output_stage3_scale1, output_stage3_scale2, output_stage3_scale3],]
      The final output can be retrieved by outputs[-1][-1].
    """

    def apply(x):
        n, h, w, c = (
            K.int_shape(x)[0],
            K.int_shape(x)[1],
            K.int_shape(x)[2],
            K.int_shape(x)[3],
        )  # input image shape

        shortcuts = []
        shortcuts.append(x)

        # Get multi-scale input images
        for i in range(1, num_supervision_scales):
            resizing_layer = Resizing(
                ratio=(2**i),
                method="nearest",
                antialias=True,  # Following `jax.image.resize()`.
                name=f"initial_resizing_{K.get_uid('Resizing')}",
            )
            shortcuts.append(resizing_layer(x))

        # store outputs from all stages and all scales
        # Eg, [[(64, 64, 3), (128, 128, 3), (256, 256, 3)],   # Stage-1 outputs
        #      [(64, 64, 3), (128, 128, 3), (256, 256, 3)],]  # Stage-2 outputs
        outputs_all = []
        sam_features, encs_prev, decs_prev = [], [], []

        for idx_stage in range(num_stages):
            # Input convolution, get multi-scale input features
            x_scales = []
            for i in range(num_supervision_scales):
                x_scale = Conv3x3(
                    filters=(2**i) * features,
                    use_bias=use_bias,
                    name=f"stage_{idx_stage}_input_conv_{i}",
                )(shortcuts[i])

                # If later stages, fuse input features with SAM features from prev stage
                if idx_stage > 0:
                    # use larger blocksize at high-res stages
                    if use_cross_gating:
                        block_size = (
                            block_size_hr if i < high_res_stages else block_size_lr
                        )
                        grid_size = (
                            grid_size_hr if i < high_res_stages else block_size_lr
                        )
                        x_scale, _ = CrossGatingBlock(
                            features=(2**i) * features,
                            block_size=block_size,
                            grid_size=grid_size,
                            dropout_rate=dropout_rate,
                            input_proj_factor=input_proj_factor,
                            upsample_y=False,
                            use_bias=use_bias,
                            name=f"stage_{idx_stage}_input_fuse_sam_{i}",
                        )(x_scale, sam_features.pop())
                    else:
                        x_scale = Conv1x1(
                            filters=(2**i) * features,
                            use_bias=use_bias,
                            name=f"stage_{idx_stage}_input_catconv_{i}",
                        )(tf.concat([x_scale, sam_features.pop()], axis=-1))

                x_scales.append(x_scale)

            # start encoder blocks
            encs = []
            x = x_scales[0]  # First full-scale input feature

            for i in range(depth):  # 0, 1, 2
                # use larger blocksize at high-res stages, vice versa.
                block_size = block_size_hr if i < high_res_stages else block_size_lr
                grid_size = grid_size_hr if i < high_res_stages else block_size_lr
                use_cross_gating_layer = True if idx_stage > 0 else False

                # Multi-scale input if multi-scale supervision
                x_scale = x_scales[i] if i < num_supervision_scales else None

                # UNet Encoder block
                enc_prev = encs_prev.pop() if idx_stage > 0 else None
                dec_prev = decs_prev.pop() if idx_stage > 0 else None

                x, bridge = UNetEncoderBlock(
                    num_channels=(2**i) * features,
                    num_groups=num_groups,
                    downsample=True,
                    lrelu_slope=lrelu_slope,
                    block_size=block_size,
                    grid_size=grid_size,
                    block_gmlp_factor=block_gmlp_factor,
                    grid_gmlp_factor=grid_gmlp_factor,
                    input_proj_factor=input_proj_factor,
                    channels_reduction=channels_reduction,
                    use_global_mlp=use_global_mlp,
                    dropout_rate=dropout_rate,
                    use_bias=use_bias,
                    use_cross_gating=use_cross_gating_layer,
                    name=f"stage_{idx_stage}_encoder_block_{i}",
                )(x, skip=x_scale, enc=enc_prev, dec=dec_prev)

                # Cache skip signals
                encs.append(bridge)

            # Global MLP bottleneck blocks
            for i in range(num_bottleneck_blocks):
                x = BottleneckBlock(
                    block_size=block_size_lr,
                    grid_size=block_size_lr,
                    features=(2 ** (depth - 1)) * features,
                    num_groups=num_groups,
                    block_gmlp_factor=block_gmlp_factor,
                    grid_gmlp_factor=grid_gmlp_factor,
                    input_proj_factor=input_proj_factor,
                    dropout_rate=dropout_rate,
                    use_bias=use_bias,
                    channels_reduction=channels_reduction,
                    name=f"stage_{idx_stage}_global_block_{i}",
                )(x)
            # cache global feature for cross-gating
            global_feature = x

            # start cross gating. Use multi-scale feature fusion
            skip_features = []
            for i in reversed(range(depth)):  # 2, 1, 0
                # use larger blocksize at high-res stages
                block_size = block_size_hr if i < high_res_stages else block_size_lr
                grid_size = grid_size_hr if i < high_res_stages else block_size_lr

                # get additional multi-scale signals
                signal = tf.concat(
                    [
                        UpSampleRatio(
                            num_channels=(2**i) * features,
                            ratio=2 ** (j - i),
                            use_bias=use_bias,
                            name=f"UpSampleRatio_{K.get_uid('UpSampleRatio')}",
                        )(enc)
                        for j, enc in enumerate(encs)
                    ],
                    axis=-1,
                )

                # Use cross-gating to cross modulate features
                if use_cross_gating:
                    skips, global_feature = CrossGatingBlock(
                        features=(2**i) * features,
                        block_size=block_size,
                        grid_size=grid_size,
                        input_proj_factor=input_proj_factor,
                        dropout_rate=dropout_rate,
                        upsample_y=True,
                        use_bias=use_bias,
                        name=f"stage_{idx_stage}_cross_gating_block_{i}",
                    )(signal, global_feature)
                else:
                    skips = Conv1x1(
                        filters=(2**i) * features, use_bias=use_bias, name="Conv_0"
                    )(signal)
                    skips = Conv3x3(
                        filters=(2**i) * features, use_bias=use_bias, name="Conv_1"
                    )(skips)

                skip_features.append(skips)

            # start decoder. Multi-scale feature fusion of cross-gated features
            outputs, decs, sam_features = [], [], []
            for i in reversed(range(depth)):
                # use larger blocksize at high-res stages
                block_size = block_size_hr if i < high_res_stages else block_size_lr
                grid_size = grid_size_hr if i < high_res_stages else block_size_lr

                # get multi-scale skip signals from cross-gating block
                signal = tf.concat(
                    [
                        UpSampleRatio(
                            num_channels=(2**i) * features,
                            ratio=2 ** (depth - j - 1 - i),
                            use_bias=use_bias,
                            name=f"UpSampleRatio_{K.get_uid('UpSampleRatio')}",
                        )(skip)
                        for j, skip in enumerate(skip_features)
                    ],
                    axis=-1,
                )

                # Decoder block
                x = UNetDecoderBlock(
                    num_channels=(2**i) * features,
                    num_groups=num_groups,
                    lrelu_slope=lrelu_slope,
                    block_size=block_size,
                    grid_size=grid_size,
                    block_gmlp_factor=block_gmlp_factor,
                    grid_gmlp_factor=grid_gmlp_factor,
                    input_proj_factor=input_proj_factor,
                    channels_reduction=channels_reduction,
                    use_global_mlp=use_global_mlp,
                    dropout_rate=dropout_rate,
                    use_bias=use_bias,
                    name=f"stage_{idx_stage}_decoder_block_{i}",
                )(x, bridge=signal)

                # Cache decoder features for later-stage's usage
                decs.append(x)

                # output conv, if not final stage, use supervised-attention-block.
                if i < num_supervision_scales:
                    if idx_stage < num_stages - 1:  # not last stage, apply SAM
                        sam, output = SAM(
                            num_channels=(2**i) * features,
                            output_channels=num_outputs,
                            use_bias=use_bias,
                            name=f"stage_{idx_stage}_supervised_attention_module_{i}",
                        )(x, shortcuts[i])
                        outputs.append(output)
                        sam_features.append(sam)
                    else:  # Last stage, apply output convolutions
                        output = Conv3x3(
                            num_outputs,
                            use_bias=use_bias,
                            name=f"stage_{idx_stage}_output_conv_{i}",
                        )(x)
                        output = output + shortcuts[i]
                        outputs.append(output)
            # Cache encoder and decoder features for later-stage's usage
            encs_prev = encs[::-1]
            decs_prev = decs

            # Store outputs
            outputs_all.append(outputs)
        return outputs_all

    return apply
