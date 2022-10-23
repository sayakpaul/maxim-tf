"""
Script to port the pre-trained JAX params of MAXIM to TF.

Usage:
    python convert_to_tf.py

The above will convert a MAXIM-3S model trained on the denoising task with the
SIDD dataset. You can find the tasks and checkpoints supported by MAXIM here:
https://github.com/google-research/maxim#results-and-pre-trained-models.

So, to convert the pre-trained JAX params (for deblurring on GoPro dataset, say) to TF,
you can run the following:
    
    python convert_to_tf.py \
        --task Deblurring \
        --ckpt_path gs://gresearch/maxim/ckpt/Deblurring/GoPro/checkpoint.npz

"""

import argparse
import collections
import io
import re
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from huggingface_hub import push_to_hub_keras

from create_maxim_model import Model
from maxim.configs import MAXIM_CONFIGS

_MODEL_VARIANT_DICT = {
    "Denoising": "S-3",
    "Deblurring": "S-3",
    "Deraining": "S-2",
    "Dehazing": "S-2",
    "Enhancement": "S-2",
}


# `recover_tree()` and `get_params()` come from here:
# https://github.com/google-research/maxim/blob/main/maxim/run_eval.py
def recover_tree(keys, values):
    """Recovers a tree as a nested dict from flat names and values.
    This function is useful to analyze checkpoints that are saved by our programs
    without need to access the exact source code of the experiment. In particular,
    it can be used to extract an reuse various subtrees of the scheckpoint, e.g.
    subtree of parameters.
    Args:
      keys: a list of keys, where '/' is used as separator between nodes.
      values: a list of leaf values.
    Returns:
      A nested tree-like dict.
    """
    tree = {}
    sub_trees = collections.defaultdict(list)
    for k, v in zip(keys, values):
        if "/" not in k:
            tree[k] = v
        else:
            k_left, k_right = k.split("/", 1)
            sub_trees[k_left].append((k_right, v))
    for k, kv_pairs in sub_trees.items():
        k_subtree, v_subtree = zip(*kv_pairs)
        tree[k] = recover_tree(k_subtree, v_subtree)
    return tree


def get_params(ckpt_path):
    """Get params checkpoint."""
    with tf.io.gfile.GFile(ckpt_path, "rb") as f:
        data = f.read()
    values = np.load(io.BytesIO(data))
    params = recover_tree(*zip(*values.items()))
    params = params["opt"]["target"]
    return params


# From https://stackoverflow.com/questions/5491913/sorting-list-in-python
def sort_nicely(l):
    """Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def modify_upsample(jax_params):
    modified_jax_params = collections.OrderedDict()

    jax_keys = list(jax_params.keys())
    keys_upsampling = []
    for k in range(len(jax_keys)):
        if "UpSample" in jax_keys[k]:
            keys_upsampling.append(jax_keys[k])
    sorted_keys_upsampling = sort_nicely(keys_upsampling)

    i = 1
    for k in sorted_keys_upsampling:
        k_t = k.split("_")[0] + "_" + str(i)
        i += 1
        for j in jax_params[k]:
            for l in jax_params[k][j]:
                modified_param_name = f"{k_t}_{j}/{l}:0"
                params = jax_params[k][j][l]
                modified_jax_params.update({modified_param_name: params})

    return modified_jax_params


def modify_jax_params(jax_params):
    modified_jax_params = collections.OrderedDict()

    for k in jax_params:
        if "UpSample" not in k:
            params = jax_params[k]

            if ("ConvTranspose" in k) and ("bias" not in k):
                params = params.transpose(0, 1, 3, 2)

            split_names = k.split("_")
            modified_param_name = (
                "_".join(split_names[0:-1]) + "/" + split_names[-1] + ":0"
            )

            if "layernorm" in modified_param_name.lower():
                if "scale" in modified_param_name:
                    modified_param_name = modified_param_name.replace("scale", "gamma")
                elif "bias" in modified_param_name:
                    modified_param_name = modified_param_name.replace("bias", "beta")

            modified_jax_params.update({modified_param_name: params})

    return modified_jax_params


def port_jax_params(configs: dict, ckpt_path: str) -> Tuple[dict, tf.keras.Model]:
    # Initialize TF Model.
    print("Initializing model.")
    tf_model = Model(**configs)

    # Obtain a mapping of the TF variable names and their values.
    tf_model_variables = tf_model.variables
    tf_model_variables_dict = {}
    for v in tf_model_variables:
        tf_model_variables_dict[v.name] = v

    # Obtain the JAX pre-trained variables.
    jax_params = get_params(ckpt_path)
    [flat_jax_dict] = pd.json_normalize(jax_params, sep="_").to_dict(orient="records")

    # Amend the JAX variables to match the names of the TF variables.
    modified_jax_params = modify_jax_params(flat_jax_dict)
    modified_jax_params.update(modify_upsample(jax_params))

    # Porting.
    tf_weights = []
    i = 0

    for k in modified_jax_params:
        param = modified_jax_params[k]
        tf_weights.append((tf_model_variables_dict[k], param))
        i += 1

    assert i == len(modified_jax_params) == len(tf_model_variables_dict)

    tf.keras.backend.batch_set_value(tf_weights)

    return modified_jax_params, tf_model


def main(args):
    task = args.task
    task_from_ckpt = args.ckpt_path.split("/")[-3]

    assert task == task_from_ckpt, "Provided task and provided checkpoints differ."
    f" Task provided: {task}, task dervived from checkpoints: {task_from_ckpt}."

    # From https://github.com/google-research/maxim/blob/main/maxim/run_eval.py#L55
    variant = _MODEL_VARIANT_DICT[task]
    configs = MAXIM_CONFIGS.get(variant)
    configs.update(
        {
            "variant": variant,
            "dropout_rate": 0.0,
            "num_outputs": 3,
            "use_bias": True,
            "num_supervision_scales": 3,
        }
    )

    _, tf_model = port_jax_params(configs, args.ckpt_path)
    print("Model porting successful.")

    dataset_name = args.ckpt_path.split("/")[-2].lower()
    tf_params_path = f"{variant}_{task.lower()}_{dataset_name}.h5"

    # tf_model.save_weights(tf_params_path)
    # print(f"Model params serialized to {tf_params_path}.")
    saved_model_path = tf_params_path.replace(".h5", "")
    tf_model.save(saved_model_path)
    print(f"SavedModel serialized to {saved_model_path}.")
    # push_to_hub_keras(tf_model, repo_path_or_name=f"sayakpaul/{saved_model_path}")
    # print("Model pushed to Hugging Face Hub.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Conversion of the JAX pre-trained MAXIM weights to TensorFlow."
    )
    parser.add_argument(
        "-t",
        "--task",
        default="Denoising",
        type=str,
        choices=[
            "Denoising",
            "Deblurring",
            "Deraining",
            "Dehazing",
            "Enhancement",
        ],
        help="Name of the task on which the corresponding checkpoints were derived.",
    )
    parser.add_argument(
        "-c",
        "--ckpt_path",
        default="gs://gresearch/maxim/ckpt/Denoising/SIDD/checkpoint.npz",
        type=str,
        help="Checkpoint to port.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
