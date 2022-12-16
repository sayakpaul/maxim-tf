"""
Layers based on https://github.com/google-research/maxim/blob/main/maxim/models/maxim.py
and reworked to cope with variable image dimensions
"""

import tensorflow as tf
from tensorflow.experimental import numpy as tnp
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable("maxim")
class TFBlockImages(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, image, patch_size):
        bs, h, w, num_channels = (
            tf.shape(image)[0],
            tf.shape(image)[1],
            tf.shape(image)[2],
            tf.shape(image)[3],
        )
        ph, pw = patch_size
        gh = h // ph
        gw = w // pw
        pad = [[0, 0], [0, 0]]
        patches = tf.space_to_batch_nd(image, [ph, pw], pad)
        patches = tf.split(patches, ph * pw, axis=0)
        patches = tf.stack(patches, 3)  # (bs, h/p, h/p, p*p, 3)
        patches_dim = tf.shape(patches)
        patches = tf.reshape(
            patches, [patches_dim[0], patches_dim[1], patches_dim[2], -1]
        )
        patches = tf.reshape(
            patches,
            (patches_dim[0], patches_dim[1] * patches_dim[2], ph * pw, num_channels),
        )
        return [patches, gh, gw]

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable("maxim")
class TFBlockImagesByGrid(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, image, grid_size):
        bs, h, w, num_channels = (
            tf.shape(image)[0],
            tf.shape(image)[1],
            tf.shape(image)[2],
            tf.shape(image)[3],
        )
        gh, gw = grid_size
        ph = h // gh
        pw = w // gw
        pad = [[0, 0], [0, 0]]

        def block_single_image(img):
            pat = tf.expand_dims(img, 0)  # batch = 1
            pat = tf.space_to_batch_nd(pat, [ph, pw], pad)  # p*p*bs, g, g, c
            pat = tf.expand_dims(pat, 3)  # pxpxbs, g, g, 1, c
            pat = tf.transpose(pat, perm=[3, 1, 2, 0, 4])  # 1, g, g, pxp, c
            pat = tf.reshape(pat, [gh, gw, ph * pw, num_channels])
            return pat

        patches = image
        patches = tf.map_fn(fn=lambda x: block_single_image(x), elems=patches)
        patches_dim = tf.shape(patches)
        patches = tf.reshape(
            patches, [patches_dim[0], patches_dim[1], patches_dim[2], -1]
        )
        patches = tf.reshape(
            patches,
            (patches_dim[0], patches_dim[1] * patches_dim[2], ph * pw, num_channels),
        )
        return [patches, ph, pw]

    def get_config(self):
        return super().get_config()


@tf.keras.utils.register_keras_serializable("maxim")
class TFUnblockImages(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, x, patch_size, grid_size):
        bs, grid_sqrt, patch_sqrt, num_channels = (
            tf.shape(x)[0],
            tf.shape(x)[1],
            tf.shape(x)[2],
            tf.shape(x)[3],
        )
        ph, pw = patch_size
        gh, gw = grid_size

        pad = [[0, 0], [0, 0]]

        y = tf.reshape(x, (bs, gh, gw, -1, num_channels))  # (bs, gh, gw, ph*pw, 3)
        y = tf.expand_dims(y, 0)
        y = tf.transpose(y, perm=[4, 1, 2, 3, 0, 5])
        y = tf.reshape(y, [bs * ph * pw, gh, gw, num_channels])
        y = tf.batch_to_space(y, [ph, pw], pad)

        return y

    def get_config(self):
        return super().get_config()


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
    def __init__(self, ratio: float, method="bilinear", antialias=True, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.method = method
        self.antialias = antialias

    def call(self, img):
        shape = tf.shape(img)

        new_sh = tf.cast(shape[1:3], tf.float32) // self.ratio

        x = tf.image.resize(
            img,
            size=tf.cast(new_sh, tf.int32),
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
