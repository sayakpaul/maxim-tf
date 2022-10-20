"""Generates model documentation for MAXIM TF models.

Credits: Willi Gierke
"""

import os
from string import Template

import attr

template = Template(
    """# Module $HANDLE

MAXIM model pre-trained on the $DATASET_DESCRIPTION suitable for image $TASK.

<!-- asset-path: https://storage.googleapis.com/maxim-tf/tars/$ARCHIVE_NAME.tar.gz  -->
<!-- task: $TASK_METADATA -->
<!-- network-architecture: maxim -->
<!-- format: saved_model_2 -->
<!-- license: apache -->
<!-- colab: https://colab.research.google.com/github/sayakpaul/maxim-tf/blob/main/notebooks/inference.ipynb -->

## Overview

This model is based on the MAXIM backbone [1] pre-trained on the $DATASET_DESCRIPTION. You can use this
model for image $TASK. Please refer to the Colab Notebook linked on this page for more details.

MAXIM introduces a common backbone for different image processing tasks like
denoising, deblurring, dehazing, deraining, and enhancement. You can find the complete
collection of MAXIM models on TF-Hub on [this page](https://tfhub.dev/sayakpaul/collections/maxim/1).

## Notes

* The original model weights are provided in [2]. There were ported to Keras models
(`tf.keras.Model`) and then serialized as TensorFlow SavedModels. The porting
steps are available in [3].
* The format of the model handle is: `'maxim_{variant}_{task}_{dataset}'`.
* The model can be unrolled into a standard Keras model and you can inspect its topology.
To do so, first download the model from TF-Hub and then load it using `tf.keras.models.load_model`
providing the path to the downloaded model folder.

## References

[1] [MAXIM: Multi-Axis MLP for Image Processing Tu et al.](https://arxiv.org/abs/2201.02973)

[2] [MAXIM GitHub](https://github.com/google-research/maxim)

[3] [MAXIM TF GitHub](https://github.com/sayakpaul/maxim-tf)

## Acknowledgements

* [Gustavo Martins](https://twitter.com/gusthema?lang=en)
* [ML-GDE program](https://developers.google.com/programs/experts/)

"""
)


@attr.s
class Config:
    variant = attr.ib(type=str)
    dataset = attr.ib(type=str)
    task = attr.ib(type=str)
    task_metadata = attr.ib(type=str)

    def gcs_folder_name(self):
        return f"{self.variant}_{self.task}_{self.dataset}"

    def handle(self):
        return f"sayakpaul/maxim_{self.gcs_folder_name().lower()}/1"

    def rel_doc_file_path(self):
        """Relative to the tfhub.dev directory."""
        return f"assets/docs/{self.handle()}.md"


for c in [
    Config("S-2", "sots-indoor", "dehazing", "dehazing"),
    Config("S-2", "sots-outdoor", "dehazing", "dehazing"),
    Config("S-2", "rain13k", "deraining", "deraining"),
    Config("S-2", "raindrop", "deraining", "deraining"),
    Config("S-2", "fivek", "enhancement", "enhancement"),
    Config("S-2", "lol", "enhancement", "enhancement"),
    Config("S-3", "gopro", "deblurring", "deblurring"),
    Config("S-3", "realblur_j", "deblurring", "deblurring"),
    Config("S-3", "realblur_r", "deblurring", "deblurring"),
    Config("S-3", "reds", "deblurring", "deblurring"),
    Config("S-3", "sidd", "denoising", "denoising"),
]:
    save_path = os.path.join(
        "/Users/sayakpaul/Downloads/", "tfhub.dev", c.rel_doc_file_path()
    )
    model_folder = save_path.split("/")[-2]
    model_abs_path = "/".join(save_path.split("/")[:-1])

    if not os.path.exists(model_abs_path):
        os.makedirs(model_abs_path, exist_ok=True)

    with open(save_path, "w") as f:
        f.write(
            template.substitute(
                HANDLE=c.handle(),
                DATASET_DESCRIPTION=c.dataset,
                TASK=c.task,
                TASK_METADATA=c.task_metadata,
                ARCHIVE_NAME=c.gcs_folder_name(),
            )
        )
