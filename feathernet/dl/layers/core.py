from typing import Any, Callable

import numpy as np

from feathernet.dl.layers.base import BaseLayer

__all__ = ["Dense", "BatchNorm", "Dropout"]


class Dense(BaseLayer):
    def __init__(
        self, input_dim: int, output_dim: int, initializer: Callable = None
    ) -> None:
        super(Dense, self).__init__()
        self.inputs = None

        if initializer is not None:
            self.initializer = initializer

        self.weights = self.initializer((input_dim, output_dim))
        self.bias = np.zeros(output_dim)

        self.weights_grad = np.zeros_like(self.weights)
        self.bias_grad = np.zeros_like(self.bias)

        self.original_input_shape = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.ndim > 2:  # Flatten
            self.original_input_shape = inputs.shape
            inputs = inputs.reshape(inputs.shape[0], -1)

        self.inputs = inputs
        return np.dot(inputs, self.weights) + self.bias

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        self.weights_grad = np.dot(self.inputs.T, output_grad)
        self.bias_grad = np.sum(output_grad, axis=0)

        if hasattr(self, "original_input_shape"):
            return np.dot(output_grad, self.weights.T).reshape(
                self.original_input_shape
            )
        else:
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
        self.mean = 0.0
        self.variance = 0.0

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
        if self.last_input is None:
            raise ValueError("No forward pass data stored in BatchNorm layer.")

        N, D = self.last_input.shape[0], np.prod(self.last_input.shape[1:])
        reshaped_inputs = self.last_input.reshape(N, D)

        reshaped_mean = self.mean.reshape(1, D)
        reshaped_variance = self.variance.reshape(1, D)

        # Compute gradients.
        inv_std = 1.0 / np.sqrt(reshaped_variance + self.epsilon)
        xmu = reshaped_inputs - reshaped_mean

        dx_normalized = output_grad.reshape(N, D) * inv_std
        d_var = np.sum(dx_normalized * xmu * -0.5 * inv_std**3, axis=0)
        d_mu = np.sum(dx_normalized * -inv_std, axis=0) + d_var * np.mean(
            -2.0 * xmu, axis=0
        )
        dx = (dx_normalized * inv_std) + (d_var * 2 * xmu / N) + (d_mu / N)
        return dx.reshape(self.last_input.shape)

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
