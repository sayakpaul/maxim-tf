import einops
import tensorflow as tf
from tensorflow.experimental import numpy as tnp
from tensorflow.keras import backend as K
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable("maxim")
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

        x = einops.rearrange(
            x,
            "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c",
            gh=grid_height,
            gw=grid_width,
            fh=patch_size[0],
            fw=patch_size[1],
        )

        return x

    def get_config(self):
        config = super().get_config().copy()
        return config


@tf.keras.utils.register_keras_serializable("maxim")
class UnblockImages(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, grid_size, patch_size):
        x = einops.rearrange(
            x,
            "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
            gh=grid_size[0],
            gw=grid_size[1],
            fh=patch_size[0],
            fw=patch_size[1],
        )

        return x

    def get_config(self):
        config = super().get_config().copy()
        return config


@tf.keras.utils.register_keras_serializable("maxim")
class SwapAxes(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, axis_one, axis_two):
        return tnp.swapaxes(x, axis_one, axis_two)

    def get_config(self):
        config = super().get_config().copy()
        return config


@tf.keras.utils.register_keras_serializable("maxim")
class Resizing(layers.Layer):
    def __init__(self, height, width, antialias=True, method="bilinear", **kwargs):
        super().__init__(**kwargs)
        self.height = height
        self.width = width
        self.antialias = antialias
        self.method = method

    def call(self, x):
        return tf.image.resize(
            x,
            size=(self.height, self.width),
            antialias=self.antialias,
            method=self.method,
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "height": self.height,
                "width": self.width,
                "antialias": self.antialias,
                "method": self.method,
            }
        )
        return config
