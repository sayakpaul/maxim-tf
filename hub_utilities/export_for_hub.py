"""Generates .tar.gz archives from SavedModels and serializes them."""


import os
from typing import List

import tensorflow as tf

TF_MODEL_ROOT = "gs://maxim-tf"
TAR_ARCHIVES = os.path.join(TF_MODEL_ROOT, "tars/")


def prepare_archive(model_name: str) -> None:
    """Prepares a tar archive."""
    archive_name = f"{model_name}.tar.gz"
    print(f"Archiving to {archive_name}.")
    archive_command = f"cd {model_name} && tar -czvf ../{archive_name} *"
    os.system(archive_command)
    os.system(f"rm -rf {model_name}")


def save_to_gcs(model_paths: List[str]) -> None:
    """Prepares tar archives and saves them inside a GCS bucket."""
    for path in model_paths:
        print(f"Preparing model: {path}.")
        model_name = path.strip("/")
        abs_model_path = os.path.join(TF_MODEL_ROOT, model_name)

        print(f"Copying from {abs_model_path}.")
        os.system(f"gsutil cp -r {abs_model_path} .")
        prepare_archive(model_name)

    os.system(f"gsutil -m cp -r *.tar.gz {TAR_ARCHIVES}")
    os.system("rm -rf *.tar.gz")


model_paths = tf.io.gfile.listdir(TF_MODEL_ROOT)
print(f"Total models: {len(model_paths)}.")

print("Preparing archives for the classification and feature extractor models.")
save_to_gcs(model_paths)
tar_paths = tf.io.gfile.listdir(TAR_ARCHIVES)
print(f"Total tars: {len(tar_paths)}.")
