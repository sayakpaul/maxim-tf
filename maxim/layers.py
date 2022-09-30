from tensorflow.experimental import numpy as tnp
from tensorflow.keras import backend as K
from tensorflow.keras import layers


class BlockImages(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, patch_size):
        bs, h, w, num_channels = (
            K.int_shape(x)[0],
            K.int_shape(x)[1],
            K.int_shape(x)[2],
            K.int_shape(x)[3],
        )

        grid_height, grid_width = h // patch_size[0], w // patch_size[1]

        x = layers.Reshape(
            (grid_height * patch_size[0], grid_width * patch_size[1], num_channels)
        )(x)

        x = layers.Reshape(
            (-1, grid_height * grid_width, patch_size[0] * patch_size[1], num_channels)
        )(x)

        return x


class UnblockImages(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, grid_size, patch_size):
        num_channels = K.int_shape(x)[-1]

        x = layers.Reshape(
            (grid_size[0] * grid_size[1], patch_size[0] * patch_size[1], num_channels)
        )(x)

        x = layers.Reshape(
            (grid_size[0] * patch_size[0], grid_size[1] * patch_size[1], num_channels)
        )(x)

        return x


class SwapAxes(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, axis_one, axis_two):
        return tnp.swapaxes(x, axis_one, axis_two)
