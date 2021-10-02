import time
import logging

import numpy as np
import tensorflow as tf


logger = tf.get_logger()
logger.setLevel(logging.ERROR)
SEED = 42


def generate_random_batch(batch: int = 128, height: int = 100, width: int = 100, depth: int = 3):
    
    images = tf.random.normal((batch, height, width, depth), seed=SEED)

    return images


def use_cpu(images):
    
    with tf.device('/cpu:0'):
        conv = tf.keras.layers.Conv2D(filters=32, kernel_size=7)
        net_cpu = conv(images)
        
        return tf.math.reduce_sum(net_cpu)


def use_gpu(images):
    
    with tf.device('/gpu:0'):
        conv = tf.keras.layers.Conv2D(filters=32, kernel_size=7)
        net_gpu = conv(images)
        
        return tf.math.reduce_sum(net_gpu)


def timeit(n, func, **kwargs):
    
    t0 = time.perf_counter()
    for _ in range(n):
        func(**kwargs)

    print(f"Elapsed time for {func.__name__}: {time.perf_counter() - t0}")


if __name__ == '__main__':
    
    batch = generate_random_batch()

    # warm up
    use_cpu(batch)
    timeit(n=10, func=use_cpu, images=batch)

    if tf.test.is_gpu_available():
        use_gpu(batch)
        timeit(n=10, func=use_gpu, images=batch)
    else:
        print("GPU isn't available. List of available devices below")
        print(tf.config.list_physical_devices())
