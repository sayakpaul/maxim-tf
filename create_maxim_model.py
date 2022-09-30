from tensorflow import keras

from maxim import maxim
from maxim.configs import MAXIM_CONFIGS


def Model(variant=None, input_resolution=256, **kw) -> keras.Model:
    """Factory function to easily create a Model variant like "S".
    Every model file should have this Model() function that returns the flax
    model function. The function name should be fixed.
    Args:
      variant: UNet model variants. Options: 'S-1' | 'S-2' | 'S-3'
          | 'M-1' | 'M-2' | 'M-3'
      input_resolution: Size of the input images.
      **kw: Other UNet config dicts.
    Returns:
      The MAXIM() model function
    """

    if variant is not None:
        config = MAXIM_CONFIGS[variant]
        for k, v in config.items():
            if k != "name":
                kw.setdefault(k, v)

    inputs = keras.Input((input_resolution, input_resolution, 3))
    maxim_model = maxim.MAXIM(**kw)
    outputs = maxim_model(inputs)
    final_model = keras.Model(inputs, outputs, name=f'{config["name"]}_model')

    return final_model
