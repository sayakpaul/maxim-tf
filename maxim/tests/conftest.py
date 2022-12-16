import gc
import random

import pytest
import tensorflow as tf
from maxim import maxim
from maxim.configs import MAXIM_CONFIGS
from tensorflow import keras


@pytest.fixture()
def fix_random():
    tf.random.set_seed(0)
    random.seed(0)


@pytest.fixture(params=[(16, 12), (12, 16), (16, 16)])
def window_size(request):
    return request.param


@pytest.fixture(params=[20, 30, 40])
def random_image(request, fix_random, window_size):
    h, w = window_size
    n_windows = request.param

    h_img = h * n_windows
    w_img = w * n_windows

    return tf.random.uniform(shape=(5, h_img, w_img, 3), dtype=tf.float32)


@pytest.fixture(params=[(10, 13), (14, 15), (20, 20)])
def random_image_multiple_of_64(request, fix_random, window_size):
    h, w = request.param
    h_img = h * 64
    w_img = w * 64

    return tf.random.uniform(shape=(1, h_img, w_img, 3), dtype=tf.float32)


##########################################################################
########################## Fixtures for model test #######################


def Model(variant=None, input_resolution=(None, None), **kw) -> keras.Model:
    """Factory function to easily create a Model variant like "S".

    Args:
      variant: UNet model variants. Options: 'S-1' | 'S-2' | 'S-3'
          | 'M-1' | 'M-2' | 'M-3'
      input_resolution: Size of the input images.
      **kw: Other UNet config dicts.

    Returns:
      The MAXIM model.
    """

    if variant is not None:
        config = MAXIM_CONFIGS[variant]
        for k, v in config.items():
            kw.setdefault(k, v)

    if "variant" in kw:
        _ = kw.pop("variant")
    if "input_resolution" in kw:
        _ = kw.pop("input_resolution")
    model_name = kw.pop("name")

    maxim_model = maxim.MAXIM(**kw)

    inputs = keras.Input((*input_resolution, 3))
    outputs = maxim_model(inputs)
    final_model = keras.Model(inputs, outputs, name=f"{model_name}_model")

    return final_model


# Scope = session means it should only be instantiated once per test session.
@pytest.fixture(scope="session", params=["S-2"])
def none_model(request):
    model = Model(variant=request.param, input_resolution=(None, None))
    yield model
    del model
    gc.collect()
