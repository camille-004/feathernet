from typing import Callable

import numpy as np

from feathernet.dl.layers.base import BaseLayer


class Dense(BaseLayer):
    def __init__(
        self, input_dim: int, output_dim: int, initializer: Callable = None
    ) -> None:
        super().__init__(initializer)
        self.inputs = None
        self.weights = self.initializer((input_dim, output_dim))
        self.bias = np.zeros(output_dim)
        self.weights_grad = np.zeros_like(self.weights)
        self.bias_grad = np.zeros_like(self.bias)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.bias

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        self.weights_grad = np.dot(self.inputs.T, output_grad)
        self.bias_grad = np.sum(output_grad, axis=0)

        return np.dot(output_grad, self.weights.T)
