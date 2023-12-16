from abc import abstractmethod
from typing import Any

import numpy as np

from feathernet.dl.layers.base import BaseLayer

__all__ = ["Activation", "ReLU", "Sigmoid", "Softmax"]


class Activation(BaseLayer):
    def __init__(self):
        super(Activation, self).__init__()

    @abstractmethod
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        pass

    def serialize(self) -> dict[str, Any]:
        return super().serialize()


class ReLU(Activation):
    def __init__(self):
        super(ReLU, self).__init__()
        self.inputs = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return np.maximum(0, inputs)

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        if (
            self.inputs.ndim > 2
            and self.inputs.shape[0] == output_grad.shape[0]
        ):
            reshaped_inputs = self.inputs.reshape(output_grad.shape)
            return output_grad * (reshaped_inputs > 0)

        return output_grad * (self.inputs > 0)


class Sigmoid(Activation):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.output = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.output = 1 / (1 + np.exp(-inputs))
        return self.output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        if (
            self.output.ndim > 2
            and self.output.shape[0] == output_grad.shape[0]
        ):
            reshaped_output = self.output.reshape(output_grad.shape)
            return output_grad * (reshaped_output * (1 - reshaped_output))

        return output_grad * (self.output * (1 - self.output))


class Softmax(Activation):
    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        return output_grad
