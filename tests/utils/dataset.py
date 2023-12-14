import numpy as np


def make_dataset(
    samples: int = 100, features: int = 10
) -> tuple[np.ndarray, np.ndarray]:
    np.random.seed(42)
    X = np.random.randn(samples, features)
    y = np.random.randint(0, 2, (samples, 1))
    return X, y
