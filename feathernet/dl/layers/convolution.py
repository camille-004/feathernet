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

        if initializer is not None:
            self.initializer = initializer

        self.weights = self.initializer(
            (output_dim, input_dim, kernel_size, kernel_size)
        )
        self.bias = np.zeros(output_dim)

        self.inputs = None
        self.inputs_col = None
        self.weights_grad = None
        self.bias_grad = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if inputs.ndim != 4:
            raise ValueError("Expected input to Conv2D to be a 4D tensor.")
        if self.padding > 0:
            pad_width = (
                (0, 0),
                (0, 0),
                (self.padding, self.padding),
                (self.padding, self.padding),
            )
            self.inputs = np.pad(
                inputs, pad_width, mode="constant", constant_values=0
            )
        else:
            self.inputs = inputs

        # Dimensions
        N, C, H, W = inputs.shape
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        self.inputs_col = self.im2col(inputs, self.kernel_size, self.stride)
        weights_col = self.weights.reshape(self.output_dim, -1)

        output_col = np.dot(weights_col, self.inputs_col) + self.bias[:, None]
        output = output_col.reshape(
            self.output_dim, out_h, out_w, N
        ).transpose(3, 0, 1, 2)
        return output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        N, C, H, W = self.inputs.shape
        output_grad_col = output_grad.transpose(1, 2, 3, 0).reshape(
            self.output_dim, -1
        )

        self.weights_grad = np.dot(output_grad_col, self.inputs_col.T).reshape(
            self.weights.shape
        )
        self.bias_grad = np.sum(output_grad_col, axis=1)

        weights_col = self.weights.reshape(self.output_dim, -1)
        dX_col = np.dot(weights_col.T, output_grad_col)
        dX = self.col2im(dX_col, N, C, H, W, self.kernel_size, self.stride)

        return dX

    def im2col(
        self, inputs: np.ndarray, kernel_size: int, stride: int
    ) -> np.ndarray:
        N, C, H, W = inputs.shape
        out_h = (H + 2 * self.padding - kernel_size) // stride + 1
        out_w = (W + 2 * self.padding - kernel_size) // stride + 1

        i0 = np.repeat(np.arange(kernel_size), kernel_size)
        i0 = np.tile(i0, C)
        i1 = stride * np.repeat(np.arange(out_h), out_w)
        j0 = np.tile(np.arange(kernel_size), kernel_size * C)
        j1 = stride * np.tile(np.arange(out_w), out_h)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        k = np.repeat(np.arange(C), kernel_size * kernel_size).reshape(-1, 1)

        if self.padding > 0:
            inputs_padded = np.pad(
                inputs,
                (
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ),
                mode="constant",
                constant_values=0,
            )
        else:
            inputs_padded = inputs

        cols = inputs_padded[:, k, i, j]
        cols = cols.transpose(1, 2, 0).reshape(
            kernel_size * kernel_size * C, -1
        )
        return cols

    def col2im(
        self,
        cols: np.ndarray,
        N: int,
        C: int,
        H: int,
        W: int,
        kernel_size: int,
        stride: int,
    ) -> np.ndarray:
        H_padded, W_padded = H + 2 * self.padding, W + 2 * self.padding
        H_out = (H - kernel_size) // stride + 1
        W_out = (W - kernel_size) // stride + 1

        cols_reshaped = cols.reshape(C * kernel_size * kernel_size, -1, N)
        cols_reshaped = cols_reshaped.transpose(2, 0, 1)

        images = np.zeros((N, C, H_padded, W_padded))

        for i in range(kernel_size):
            for j in range(kernel_size):
                i_lim = i + stride * H_out
                j_lim = j + stride * W_out

                idx_start = i * kernel_size + j
                cols_slice = cols_reshaped[:, idx_start :: kernel_size**2, :]
                cols_slice_reshaped = cols_slice.reshape(N, -1, H_out, W_out)

                images[
                    :, :, i:i_lim:stride, j:j_lim:stride
                ] += cols_slice_reshaped

        return images[
            :,
            :,
            self.padding : H_padded - self.padding,
            self.padding : W_padded - self.padding,
        ]

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
