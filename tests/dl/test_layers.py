import unittest

import numpy as np

from feathernet.dl.layers import BatchNorm, Conv2D, Dropout, Pooling


class TestBatchNorm(unittest.TestCase):
    def setUp(self) -> None:
        self.batch_norm = BatchNorm()

    def test_forward(self) -> None:
        inputs = np.random.rand(5, 10)
        output = self.batch_norm.forward(inputs)
        self.assertEqual(output.shape, inputs.shape)

    def test_backward(self) -> None:
        inputs = np.random.rand(5, 10)
        self.batch_norm.forward(inputs)
        output_grad = np.random.rand(5, 10)
        input_grad = self.batch_norm.backward(output_grad)
        self.assertEqual(input_grad.shape, inputs.shape)


class TestDropout(unittest.TestCase):
    def setUp(self) -> None:
        self.dropout = Dropout(rate=0.5)

    def test_forward_training(self) -> None:
        inputs = np.random.rand(5, 10)
        output = self.dropout.forward(inputs, training=True)
        self.assertEqual(output.shape, inputs.shape)

    def test_forward_inference(self):
        inputs = np.random.rand(5, 10)
        output = self.dropout.forward(inputs, training=False)
        np.testing.assert_array_almost_equal(
            output, inputs * (1 - self.dropout.rate)
        )

    def test_backward(self):
        inputs = np.random.rand(5, 10)
        self.dropout.forward(inputs, training=True)
        output_grad = np.random.rand(5, 10)
        input_grad = self.dropout.backward(output_grad)
        self.assertEqual(input_grad.shape, inputs.shape)


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


class TestPooling(unittest.TestCase):
    def setUp(self) -> None:
        self.max_pool = Pooling(pool_size=2, stride=2, strategy="max")
        self.avg_pool = Pooling(pool_size=2, stride=2, strategy="average")

    def test_forward_max_pool(self) -> None:
        inputs = np.random.rand(1, 1, 4, 4)
        output = self.max_pool.forward(inputs)
        self.assertEqual(output.shape, (1, 1, 2, 2))

    def test_forward_avg_pool(self) -> None:
        inputs = np.random.rand(1, 1, 4, 4)
        output = self.avg_pool.forward(inputs)
        self.assertEqual(output.shape, (1, 1, 2, 2))

    def test_backward_max_pool(self) -> None:
        inputs = np.random.rand(1, 1, 4, 4)
        self.max_pool.forward(inputs)
        output_grad = np.random.rand(1, 1, 2, 2)
        input_grad = self.max_pool.backward(output_grad)
        self.assertEqual(input_grad.shape, inputs.shape)

    def test_backward_avg_pool(self) -> None:
        inputs = np.random.rand(1, 1, 4, 4)
        self.avg_pool.forward(inputs)
        output_grad = np.random.rand(1, 1, 2, 2)
        input_grad = self.avg_pool.backward(output_grad)
        self.assertEqual(input_grad.shape, inputs.shape)


if __name__ == "__main__":
    unittest.main()
