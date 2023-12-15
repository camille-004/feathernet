from typing import Any

import numpy as np

from feathernet.dl.layers.base import BaseLayer


class Pooling(BaseLayer):
    def __init__(
        self, strategy: str = "max", pool_size: int = 2, stride: int = 2
    ) -> None:
        super(Pooling, self).__init__()
        self.strategy = strategy
        self.pool_size = pool_size
        self.stride = stride

        self.last_input = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.last_input = inputs

        h, w = inputs.shape[2], inputs.shape[3]
        new_h = (h - self.pool_size) // self.stride + 1
        new_w = (w - self.pool_size) // self.stride + 1

        output = np.zeros((inputs.shape[0], inputs.shape[1], new_h, new_w))

        for image in range(inputs.shape[0]):
            for channel in range(inputs.shape[1]):
                for i in range(new_h):
                    for j in range(new_w):
                        h_start, h_end = (
                            i * self.stride,
                            i * self.stride + self.pool_size,
                        )
                        w_start, w_end = (
                            j * self.stride,
                            j * self.stride + self.pool_size,
                        )
                        pool_region = inputs[
                            image, channel, h_start:h_end, w_start:w_end
                        ]

                        if self.strategy == "max":
                            output[image, channel, i, j] = np.max(pool_region)
                        elif self.strategy == "average":
                            output[image, channel, i, j] = np.mean(pool_region)

        return output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        input_grad = np.zeros_like(self.last_input)

        for batch in range(output_grad.shape[0]):
            for channel in range(output_grad.shape[1]):
                for i in range(output_grad.shape[2]):
                    for j in range(output_grad.shape[3]):
                        h_start, h_end = (
                            i * self.stride,
                            i * self.stride + self.pool_size,
                        )
                        w_start, w_end = (
                            j * self.stride,
                            j * self.stride + self.pool_size,
                        )
                        pool_region = self.last_input[
                            batch, channel, h_start:h_end, w_start:w_end
                        ]

                        if pool_region.size > 0:
                            if self.strategy == "max":
                                max_val = np.max(pool_region)
                                mask = pool_region == max_val
                                input_grad[
                                    batch,
                                    channel,
                                    h_start:h_end,
                                    w_start:w_end,
                                ] += (
                                    mask * output_grad[batch, channel, i, j]
                                )
                            elif self.strategy == "average":
                                avg_grad = output_grad[
                                    batch, channel, i, j
                                ] / (self.pool_size**2)
                                input_grad[
                                    batch,
                                    channel,
                                    h_start:h_end,
                                    w_start:w_end,
                                ] += (
                                    np.ones_like(
                                        (self.pool_size, self.pool_size)
                                    )
                                    * avg_grad
                                )

        return input_grad

    def serialize(self) -> dict[str, Any]:
        serialized_data = super().serialize()
        serialized_data.update(
            {
                "pool_size": self.pool_size,
                "stride": self.stride,
                "strategy": self.strategy,
            }
        )
        return serialized_data
