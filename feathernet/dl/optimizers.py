from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Optimizer(ABC):
    @abstractmethod
    def update(self, params: np.ndarray, grads: np.ndarray) -> None:
        pass

    def serialize(self) -> dict[str, Any]:
        return {"type": self.__class__.__name__}


class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate

    def update(self, params: np.ndarray, grads: np.ndarray) -> None:
        params -= self.learning_rate * grads

    def serialize(self) -> dict[str, Any]:
        serialized_data = super().serialize()
        serialized_data["learning_rate"] = self.learning_rate
        return serialized_data
