import os

GCS_ROOT = "gs://gresearch/maxim/ckpt"

# From https://github.com/google-research/maxim#results-and-pre-trained-models
DS_TASKS_MAP = {
    "Denoising": ["SIDD"],
    "Deblurring": [
        "GoPro",
        "REDS",
        "RealBlur_R",
        "RealBlur_J",
    ],
    "Deraining": ["Rain13k", "Raindrop"],
    "Dehazing": [
        "SOTS-Indoor",
        "SOTS-Outdoor",
    ],
    "Enhancement": ["LOL", "FiveK"],
}


def main():
    for task in DS_TASKS_MAP:
        datasets = DS_TASKS_MAP[task]

        for dataset in datasets:
            command = f"python ../convert_to_tf.py -t {task} -c {GCS_ROOT}/{task}/{dataset}/checkpoint.npz"
            print(f"Converting for task: {task} and dataset: {dataset}.")
            os.system(command)


if __name__ == "__main__":
    main()
