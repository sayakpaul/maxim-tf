import random

import einops
import numpy as np
import tensorflow as tf
from maxim.layers import TFBlockImages, TFBlockImagesByGrid, TFUnblockImages
from tensorflow.keras import backend as K
from tensorflow.keras import layers

LOW_THRESHOLD = 1e-7


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


def test_patch_block_equivalence(random_image, window_size):
    patch_size = window_size
    patched_image_original = BlockImages()(random_image, patch_size=patch_size)
    patched_image_tf, _, _ = TFBlockImages()(random_image, patch_size=patch_size)
    difference = np.sum(
        (patched_image_original.numpy() - patched_image_tf.numpy()) ** 2
    )
    assert difference < LOW_THRESHOLD


def test_grid_block_equivalence(random_image, window_size):
    grid_size = window_size
    gh, gw = grid_size
    height, width = random_image.shape[1], random_image.shape[2]
    patch_size = (height // gh, width // gw)
    patched_image_original = BlockImages()(random_image, patch_size=patch_size)
    patched_image_tf, _, _ = TFBlockImagesByGrid()(random_image, grid_size=grid_size)
    difference = np.sum(
        (patched_image_original.numpy() - patched_image_tf.numpy()) ** 2
    )
    assert difference < LOW_THRESHOLD


def test_reconstruction_by_grid(random_image, window_size):
    grid_size = window_size
    height, width = random_image.shape[1], random_image.shape[2]
    p_h, p_w = height // grid_size[0], width // grid_size[1]

    # Block and Unblock with einops layers
    patched_image_original = BlockImages()(random_image, patch_size=(p_h, p_w))
    reconstructed_original = UnblockImages()(
        patched_image_original,
        grid_size=grid_size,
        patch_size=(p_h, p_w),
    )

    # Block and Unblock with TF layers
    patched_image_tf, ph, pw = TFBlockImagesByGrid()(random_image, grid_size=grid_size)
    reconstructed_image_tf = TFUnblockImages()(
        patched_image_tf, grid_size=grid_size, patch_size=(ph, pw)
    )

    # Compare implementation diff and reconstruction diff
    difference_between_implementations = np.sum(
        (reconstructed_original.numpy() - reconstructed_image_tf.numpy()) ** 2
    )
    assert difference_between_implementations < LOW_THRESHOLD
    difference_between_reconstruction = np.sum(
        (random_image.numpy() - reconstructed_image_tf.numpy()) ** 2
    )
    assert difference_between_reconstruction < LOW_THRESHOLD


def test_reconstruction(random_image, window_size):
    patch_size = window_size
    height, width = random_image.shape[1], random_image.shape[2]

    # Block and Unblock with einops layers
    patched_image_original = BlockImages()(random_image, patch_size=patch_size)
    reconstructed_original = UnblockImages()(
        patched_image_original,
        grid_size=(height // patch_size[0], width // patch_size[1]),
        patch_size=patch_size,
    )

    # Block and Unblock with TF layers
    patched_image_tf, gh, gw = TFBlockImages()(random_image, patch_size=patch_size)
    reconstructed_image_tf = TFUnblockImages()(
        patched_image_tf, patch_size=patch_size, grid_size=(gh, gw)
    )

    # Compare implementation diff and reconstruction diff
    difference_between_implementations = np.sum(
        (reconstructed_original.numpy() - reconstructed_image_tf.numpy()) ** 2
    )
    assert difference_between_implementations < LOW_THRESHOLD
    difference_between_reconstruction = np.sum(
        (random_image.numpy() - reconstructed_image_tf.numpy()) ** 2
    )
    assert difference_between_reconstruction < LOW_THRESHOLD
