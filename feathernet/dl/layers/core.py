from typing import Any, Callable

import numpy as np

from feathernet.dl.layers.base import BaseLayer


class Dense(BaseLayer):
    def __init__(
        self, input_dim: int, output_dim: int, initializer: Callable = None
    ) -> None:
        super(Dense, self).__init__(initializer)
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

    def serialize(self) -> dict[str, Any]:
        serialized_data = super().serialize()
        serialized_data.update(
            {
                "input_dim": self.weights.shape[0],
                "output_dim": self.weights.shape[1],
                "initializer": self._get_initializer_name(),
            }
        )
        return serialized_data


class Dropout(BaseLayer):
    def __init__(self, rate) -> None:
        super(Dropout, self).__init__()
        self.rate = rate
        self.mask = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=inputs.shape)
            return inputs * self.mask
        else:
            return inputs * (1 - self.rate)

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad * self.mask

    def serialize(self) -> dict[str, Any]:
        serialized_data = super().serialize()
        serialized_data.update({"rate": self.rate})
        return serialized_data


class BatchNorm(BaseLayer):
    def __init__(self, epsilon: float = 1e-5) -> None:
        super(BatchNorm, self).__init__()
        self.epsilon = epsilon
        self.mean = None
        self.variance = None

        self.last_input = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        self.last_input = inputs

        if training:
            self.mean = np.mean(inputs, axis=0)
            self.variance = np.var(inputs, axis=0)
            normalized = (inputs - self.mean) / np.sqrt(
                self.variance + self.epsilon
            )
            return normalized
        else:
            return (inputs - self.mean) / np.sqrt(self.variance + self.epsilon)

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        N, D = self.last_input.shape
        inv_std = 1.0 / np.sqrt(self.variance + self.epsilon)
        xmu = self.last_input - self.mean
        dx_normalized = output_grad * inv_std
        d_var = np.sum(output_grad * xmu * -0.5 * inv_std**3, axis=0)
        d_mu = np.sum(dx_normalized * -inv_std, axis=0) + d_var * np.mean(
            -2.0 * xmu, axis=0
        )
        dx = (dx_normalized * inv_std) + (d_var * 2 * xmu / N) + (d_mu / N)
        return dx

    def serialize(self) -> dict[str, Any]:
        serialized_data = super().serialize()
        serialized_data.update(
            {
                "epsilon": self.epsilon,
                "mean": self.mean,
                "variance": self.variance,
            }
        )
        return serialized_data
