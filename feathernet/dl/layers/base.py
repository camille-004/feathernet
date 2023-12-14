from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class BaseLayer(ABC):
    def __init__(self, initializer: Callable = None):
        self.initializer = (
            initializer
            if initializer is not None
            else lambda s: np.random.randn(*s)
        )

    @abstractmethod
    def forward(self, _input: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError
