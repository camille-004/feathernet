import unittest

import numpy as np

from feathernet.dl.layers.convolution import Conv2D


class TestConv2D(unittest.TestCase):
    def setUp(self) -> None:
        self.conv2d = Conv2D(
            input_dim=1, output_dim=2, kernel_size=3, stride=1, padding=0
        )
        self.conv2d.weights = np.ones((2, 1, 3, 3))
        self.conv2d.bias = np.zeros(2)

    def test_forward(self) -> None:
        inputs = np.random.rand(1, 1, 5, 5)
        output = self.conv2d.forward(inputs)
        self.assertEqual(output.shape, (1, 2, 3, 3))

    def test_backward(self) -> None:
        inputs = np.random.rand(1, 1, 5, 5)
        output = self.conv2d.forward(inputs)
        output_grad = np.ones_like(output)
        input_grad = self.conv2d.backward(output_grad)
        self.assertEqual(input_grad.shape, inputs.shape)


if __name__ == "__main__":
    unittest.main()
