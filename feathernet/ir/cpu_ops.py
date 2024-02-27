import numpy as np


def cpu_add(operands: list) -> np.ndarray:
    return np.add(*operands)


def cpu_matmul(operands: list) -> np.ndarray:
    return np.matmul(*operands)
