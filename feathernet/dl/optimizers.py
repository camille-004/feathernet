from abc import ABC, abstractmethod

import numpy as np


class Optimizer(ABC):
    @abstractmethod
    def update(self, params: np.ndarray, grads: np.ndarray) -> None:
        pass


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate

    def update(self, params: np.ndarray, grads: np.ndarray) -> None:
        params -= self.learning_rate * grads
