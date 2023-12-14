from typing import Any, Callable

import numpy as np

from feathernet.dl.layers.base import BaseLayer


class Conv2D(BaseLayer):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        initializer: Callable = None,
    ) -> None:
        super(Conv2D, self).__init__(initializer)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.inputs = None
        self.bias_grad = None
        self.weights_grad = None
        self.output_width = None
        self.output_height = None

        self.weights = self.initializer((input_dim, output_dim))
        self.bias = np.zeros(output_dim)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if self.padding > 0:
            self.inputs = np.pad(
                inputs,
                (
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ),
                mode="constant",
                constant_values=0,
            )
        else:
            self.inputs = inputs

        self.output_height = (
            self.inputs.shape[2] - self.kernel_size
        ) // self.stride + 1
        self.output_width = (
            self.inputs.shape[3] - self.kernel_size
        ) // self.stride + 1

        output = np.zeros(
            (
                inputs.shape[0],
                self.output_dim,
                self.output_height,
                self.output_width,
            )
        )

        for i in range(self.output_height):
            for j in range(self.output_width):
                input_slice = self.inputs[
                    :,
                    :,
                    i * self.stride : i * self.stride + self.kernel_size,
                    j * self.stride : j * self.stride + self.kernel_size,
                ]
                output[:, :, i, j] = (
                    np.tensordot(
                        input_slice, self.weights, axes=([1, 2, 3], [1, 2, 3])
                    )
                    + self.bias
                )

        return output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        # Gradient w.r.t. input
        input_grad = np.zeros_like(self.inputs)

        # Gradient w.r.t. weights, biases
        self.weights_grad = np.zeros_like(self.weights)
        self.bias_grad = np.zeros_like(self.bias)

        for i in range(self.output_height):
            for j in range(self.output_width):
                input_slice = self.inputs[
                    :,
                    :,
                    i * self.stride : i * self.stride + self.kernel_size,
                    j * self.stride : j * self.stride + self.kernel_size,
                ]
                for k in range(self.output_dim):
                    self.weights_grad[k, :, :, :] += np.sum(
                        input_slice
                        * output_grad[:, k, i, j][:, None, None, None],
                        axis=0,
                    )
                    input_grad[
                        :,
                        :,
                        i * self.stride : i * self.stride + self.kernel_size,
                        j * self.stride : j * self.stride + self.kernel_size,
                    ] += (
                        self.weights[k, :, :, :]
                        * output_grad[:, k, i, j][:, None, None, None]
                    )

        return input_grad

    def serialize(self) -> dict[str, Any]:
        serialized_data = super().serialize()
        serialized_data.update(
            {
                "input_dim": self.input_dim,
                "output_dim": self.output_dim,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "padding": self.padding,
                "initializer": self._get_initializer_name(),
            }
        )
        return serialized_data
