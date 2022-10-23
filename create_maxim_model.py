from tensorflow import keras

from maxim import maxim
from maxim.configs import MAXIM_CONFIGS


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
