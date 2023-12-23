from typing import Iterator

import numpy as np


class DataLoader:
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
    ) -> None:
        self.X, self.y = X, y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = self.X.shape[0]
        self.indices = np.arange(self.num_samples)

    def __iter__(self) -> Iterator:
        if self.shuffle:
            np.random.shuffle(self.indices)
        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = min(start_idx + self.batch_size, self.num_samples)
            batch_indices = self.indices[start_idx:end_idx]
            yield self.X[batch_indices], self.y[batch_indices]
