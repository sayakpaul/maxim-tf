import timeit

import numpy as np
import tensorflow as tf

from create_maxim_model import Model

INPUT_RESOLUTION = 256

MAXIM_S1 = Model("S-1")
DUMMY_INPUTS = tf.ones((1, INPUT_RESOLUTION, INPUT_RESOLUTION, 3))


def benchmark_regular_model():
    # Warmup
    print("Benchmarking TF model...")
    for _ in range(2):
        _ = MAXIM_S1(DUMMY_INPUTS, training=False)

    # Timing
    tf_runtimes = timeit.repeat(
        lambda: MAXIM_S1(DUMMY_INPUTS, training=False), number=1, repeat=10
    )
    print(f"Average latency (seconds): {np.mean(tf_runtimes)}.")


@tf.function(jit_compile=True)
def infer():
    return MAXIM_S1(DUMMY_INPUTS, training=False)


def benchmark_xla_model():
    # Warmup
    print("Benchmarking Jit-compiled TF model...")
    for _ in range(2):
        _ = infer()

    # Timing
    tf_runtimes = timeit.repeat(lambda: infer(), number=1, repeat=10)
    print(f"Average latency (seconds): {np.mean(tf_runtimes)}.")


if __name__ == "__main__":
    benchmark_regular_model()
    benchmark_xla_model()
