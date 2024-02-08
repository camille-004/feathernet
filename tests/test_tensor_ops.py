import unittest

import numpy as np

from feathernet.tensor import Tensor


class TestTensorOps(unittest.TestCase):
    def test_scalar_addition(self) -> None:
        tensor_a = Tensor(5, device="cpu")
        tensor_b = Tensor(3, device="cpu")
        result_cpu = tensor_a + tensor_b
        result_gpu = tensor_a.to("gpu") + tensor_b.to("gpu")
        self.assertEqual(result_cpu.data, result_gpu.data)
        self.assertEqual(result_cpu.data, 8)

    def test_vector_addition(self) -> None:
        tensor_a = Tensor([1, 2, 3], device="cpu")
        tensor_b = Tensor([4, 5, 6], device="cpu")
        result_cpu = tensor_a + tensor_b
        result_gpu = tensor_a.to("gpu") + tensor_b.to("gpu")
        np.testing.assert_array_equal(result_cpu.data, result_gpu.data)
        np.testing.assert_array_equal(result_cpu.data, np.array([5, 7, 9]))

    def test_matrix_addition(self) -> None:
        tensor_a = Tensor([[1, 2], [3, 4]], device="cpu")
        tensor_b = Tensor([[5, 6], [7, 8]], device="cpu")
        result_cpu = tensor_a + tensor_b
        result_gpu = tensor_a.to("gpu") + tensor_b.to("gpu")
        np.testing.assert_array_equal(result_cpu.data, result_gpu.data)
        np.testing.assert_array_equal(
            result_cpu.data, np.array([[6, 8], [10, 12]])
        )

    def test_mismatching_shapes(self) -> None:
        tensor_a = Tensor(5, device="cpu")
        tensor_b = Tensor([1, 2], device="cpu")
        with self.assertRaises(ValueError):
            result = tensor_a + tensor_b  # noqa

    def test_mismatching_device(self) -> None:
        tensor_a = Tensor([1, 2, 3], device="cpu")
        tensor_b = Tensor([4, 5, 6], device="gpu")
        result = tensor_a + tensor_b
        np.testing.assert_array_equal(result.data, np.array([5, 7, 9]))

    def test_non_tensor_op(self) -> None:
        tensor_a = Tensor([1, 2, 3], device="cpu")
        tensor_b = np.array([4, 5, 6])
        with self.assertRaises(ValueError):
            result = tensor_a + tensor_b  # noqa


if __name__ == "__main__":
    unittest.main()
