from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np


class BaseLayer(ABC):
    def __init__(self, initializer: Callable = None):
        self.initializer = (
            initializer
            if initializer is not None
            else lambda s: np.random.randn(*s)
        )
        self.layer_type = self.__class__.__name__

    @abstractmethod
    def forward(self, _input: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def serialize(self) -> dict[str, Any]:
        serialized_data = {"type": self.layer_type}

        if hasattr(self, "weights"):
            serialized_data["weights"] = self.weights
        if hasattr(self, "bias"):
            serialized_data["bias"] = self.bias

        return serialized_data

    def _get_initializer_name(self) -> str:
        if hasattr(self.initializer, "__name__"):
            if self.initializer.__name__ == "<lambda>":
                return "random_initializer"
            return self.initializer.__name__
        else:
            return "none"
