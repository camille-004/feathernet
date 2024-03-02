import unittest

import numpy as np

from feathernet.tensor import Tensor


class TestTensor(unittest.TestCase):
    def test_tensor_init_with_scalar(self) -> None:
        tensor = Tensor(5)
        self.assertEqual(tensor.data, 5)
        self.assertEqual(tensor.shape, ())
        self.assertEqual(tensor.dtype, np.float32)
        self.assertEqual(tensor.device, "cpu")

    def test_tensor_init_with_array(self) -> None:
        array = np.array([1, 2, 3])
        tensor = Tensor(array)
        np.testing.assert_array_equal(tensor.data, array)
        self.assertEqual(tensor.shape, array.shape)
        self.assertEqual(tensor.dtype, array.dtype)
        self.assertEqual(tensor.device, "cpu")

    def test_tensor_shape(self) -> None:
        array = np.array([[1, 2], [3, 4]])
        tensor = Tensor(array)
        self.assertEqual(tensor.shape, (2, 2))

    def tensor_tensor_dtype(self) -> None:
        array = np.array([1.0, 2.0, 3.0])
        tensor = Tensor(array)
        self.assertEqual(tensor.dtype, np.float64)

    def test_tensor_device(self) -> None:
        tensor = Tensor(5, device="gpu")
        self.assertEqual(tensor.device, "gpu")

    def test_tensor_reshape(self) -> None:
        array = np.array([1, 2, 3, 4])
        tensor = Tensor(array)
        reshaped = tensor.reshape((2, 2))
        self.assertEqual(reshaped.shape, (2, 2))
        np.testing.assert_array_equal(reshaped.data, array.reshape((2, 2)))

    def test_tensor_to(self) -> None:
        array = np.array([1, 2, 3])
        tensor = Tensor(array, device="cpu")
        tensor_gpu = tensor.to("gpu")
        self.assertEqual(tensor_gpu.device, "gpu")
        np.testing.assert_array_equal(tensor_gpu.data, array)


if __name__ == "__main__":
    unittest.main()
