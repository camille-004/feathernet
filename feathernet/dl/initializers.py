import numpy as np


def xavier_initializer(shape: tuple[int]) -> np.ndarray:
    stddev = np.sqrt(2 / (shape[0] + shape[1]))
    return np.random.randn(*shape) * stddev


def he_initializer(shape: tuple[int]) -> np.ndarray:
    stddev = np.sqrt(2 / (shape[0]))
    return np.random.randn(*shape) * stddev
