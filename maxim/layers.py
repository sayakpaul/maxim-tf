"""
Layers based on https://github.com/google-research/maxim/blob/main/maxim/models/maxim.py
"""

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
class Resizing(tf.keras.layers.Layer):
    def __init__(self, ratio: int, method="bilinear", antialias=True, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.method = method
        self.antialias = antialias

    def __call__(self, img):
        n, h, w, c = tf.shape(img)
        x = tf.image.resize(
            img,
            (h // self.ratio, w // self.ratio),
            method=self.method,
            antialias=self.antialias,
        )
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "ratio": self.ratio,
                "antialias": self.antialias,
                "method": self.method,
            }
        )
        return config
