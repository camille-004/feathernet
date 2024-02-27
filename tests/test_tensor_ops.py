import unittest

import numpy as np

from feathernet.tensor import Tensor


class TestTensorAdd(unittest.TestCase):
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

    def test_scalar_addition_multiple_addends(self) -> None:
        tensor_a = Tensor(5, device="cpu")
        tensor_b = Tensor(3, device="cpu")
        tensor_c = Tensor(4, device="cpu")
        tensor_d = Tensor(2, device="cpu")

        result_cpu = tensor_a + tensor_b + tensor_c + tensor_d
        result_gpu = tensor_a.to("gpu") + tensor_b + tensor_c + tensor_d

        self.assertEqual(result_cpu.data, result_gpu.data)

        expected_result = 5 + 3 + 4 + 2
        self.assertEqual(result_cpu.data, expected_result)

    def test_mismatching_device(self) -> None:
        tensor_a = Tensor([1, 2, 3], device="cpu")
        tensor_b = Tensor([4, 5, 6], device="gpu")
        result = tensor_a + tensor_b
        np.testing.assert_array_equal(result.data, np.array([5, 7, 9]))


class TestTensorSub(unittest.TestCase):
    def test_scalar_subtraction(self) -> None:
        tensor_a = Tensor(5, device="cpu")
        tensor_b = Tensor(3, device="cpu")

        result_cpu = tensor_a - tensor_b
        result_gpu = tensor_a.to("gpu") - tensor_b.to("gpu")
        self.assertEqual(result_cpu.data, result_gpu.data)
        self.assertEqual(result_cpu.data, 2)

    def test_vector_subtraction(self) -> None:
        tensor_a = Tensor([1, 2, 3], device="cpu")
        tensor_b = Tensor([4, 5, 6], device="cpu")
        result_cpu = tensor_a - tensor_b
        result_gpu = tensor_a.to("gpu") - tensor_b.to("gpu")
        np.testing.assert_array_equal(result_cpu.data, result_gpu.data)
        np.testing.assert_array_equal(result_cpu.data, np.array([-3, -3, -3]))

    def test_matrix_subtraction(self) -> None:
        tensor_a = Tensor([[1, 2], [3, 4]], device="cpu")
        tensor_b = Tensor([[5, 6], [7, 8]], device="cpu")
        result_cpu = tensor_a - tensor_b
        result_gpu = tensor_a.to("gpu") - tensor_b.to("gpu")
        np.testing.assert_array_equal(result_cpu.data, result_gpu.data)
        np.testing.assert_array_equal(
            result_cpu.data, np.array([[-4, -4], [-4, -4]])
        )

    def test_scalar_subtraction_multiple_subtrahends(self) -> None:
        tensor_a = Tensor(5, device="cpu")
        tensor_b = Tensor(3, device="cpu")
        tensor_c = Tensor(4, device="cpu")
        tensor_d = Tensor(2, device="cpu")

        result_cpu = tensor_a - tensor_b - tensor_c - tensor_d
        result_gpu = tensor_a.to("gpu") - tensor_b - tensor_c - tensor_d

        self.assertEqual(result_cpu.data, result_gpu.data)

        expected_result = 5 - 3 - 4 - 2
        self.assertEqual(result_cpu.data, expected_result)

    def test_mismatching_device(self) -> None:
        tensor_a = Tensor([1, 2, 3], device="cpu")
        tensor_b = Tensor([4, 5, 6], device="gpu")
        result = tensor_a - tensor_b
        np.testing.assert_array_equal(result.data, np.array([-3, -3, -3]))

    def test_combined_addition_subtraction(self) -> None:
        tensor_a = Tensor([100, 200, 300], device="cpu")
        tensor_b = Tensor([10, 20, 30], device="cpu")
        tensor_c = Tensor([1, 2, 3], device="cpu")
        tensor_d = Tensor([5, 5, 5], device="cpu")

        result_cpu = tensor_a - tensor_b + tensor_c - tensor_d
        result_gpu = (
            tensor_a.to("gpu")
            - tensor_b.to("gpu")
            + tensor_c.to("gpu")
            - tensor_d.to("gpu")
        )

        expected_result = (
            np.array([100, 200, 300])
            - np.array([10, 20, 30])
            + np.array([1, 2, 3])
            - np.array([5, 5, 5])
        )
        np.testing.assert_array_equal(result_cpu.data, result_gpu.data)
        np.testing.assert_array_equal(result_cpu.data, expected_result)


class TestTensorMatMul(unittest.TestCase):
    def test_vector_matmul(self) -> None:
        tensor_a = Tensor([1, 2, 3], device="cpu")
        tensor_b = Tensor([[1], [2], [3]], device="cpu")
        result_cpu = tensor_a @ tensor_b
        result_gpu = tensor_a.to("gpu") @ tensor_b.to("gpu")
        self.assertEqual(result_cpu.data, result_gpu.data)
        self.assertEqual(result_cpu.data, 14)

    def test_matrix_vector_matmul(self) -> None:
        tensor_a = Tensor([[1, 2], [3, 4]], device="cpu")
        tensor_b = Tensor([1, 2], device="cpu")
        result_cpu = tensor_a @ tensor_b
        result_gpu = tensor_a.to("gpu") @ tensor_b.to("gpu")
        np.testing.assert_array_equal(result_cpu.data, result_gpu.data)
        np.testing.assert_array_equal(result_cpu.data, np.array([5, 11]))

    def test_matrix_matmul(self) -> None:
        tensor_a = Tensor([[1, 2], [3, 4]], device="cpu")
        tensor_b = Tensor([[5, 6], [7, 8]], device="cpu")
        result_cpu = tensor_a @ tensor_b
        result_gpu = tensor_a.to("gpu") @ tensor_b.to("gpu")
        np.testing.assert_array_equal(result_cpu.data, result_gpu.data)
        np.testing.assert_array_equal(
            result_cpu.data, np.array([[19, 22], [43, 50]])
        )

    def test_mismatching_device_matmul(self) -> None:
        tensor_a = Tensor([[1, 2], [3, 4]], device="cpu")
        tensor_b = Tensor([[5, 6], [7, 8]], device="gpu")
        result = tensor_a @ tensor_b
        np.testing.assert_array_equal(
            result.data, np.array([[19, 22], [43, 50]])
        )

    def test_gpu_matmul_multidim(self):
        a_data = np.random.rand(10, 5, 4).astype(np.float32)
        b_data = np.random.rand(10, 4, 6).astype(np.float32)
        tensor_a = Tensor(a_data, device="cpu")
        tensor_b = Tensor(b_data, device="cpu")

        expected_result = np.matmul(a_data, b_data)

        result_cpu = tensor_a @ tensor_b
        result_gpu = tensor_a.to("gpu") @ tensor_b.to("gpu")
        np.testing.assert_array_equal(result_cpu.data, result_gpu.data)
        np.testing.assert_allclose(result_cpu.data, expected_result, atol=1e-6)

    def test_chain_matmul(self) -> None:
        tensor_a = Tensor([[1, 2], [3, 4]], device="cpu")
        tensor_b = Tensor([[5, 6], [7, 8]], device="cpu")
        tensor_c = Tensor([[1, 2], [3, 4]], device="cpu")

        result_cpu = tensor_a @ tensor_b @ tensor_c
        result_gpu = (
            tensor_a.to("gpu") @ tensor_b.to("gpu") @ tensor_c.to("gpu")
        )
        expected_result = np.matmul(
            np.matmul(tensor_a.data, tensor_b.data), tensor_c.data
        )
        np.testing.assert_array_equal(result_cpu.data, result_gpu.data)
        np.testing.assert_array_almost_equal(
            result_cpu.data, expected_result, decimal=6
        )


class TestTensorComplexOps(unittest.TestCase):
    def test_combined_addition_matmul(self) -> None:
        tensor_a = Tensor([[1, 2], [3, 4]], device="cpu")
        tensor_b = Tensor([[5, 6], [7, 8]], device="cpu")
        tensor_c = Tensor([[1, 1]], device="cpu")
        tensor_d = Tensor([[1], [2]], device="cpu")

        result_cpu = (tensor_a @ tensor_b) + (tensor_c @ tensor_d)
        result_gpu = (tensor_a.to("gpu") @ tensor_b.to("gpu")) + (
            tensor_c.to("gpu") @ tensor_d.to("gpu")
        )

        expected_result = (np.dot(tensor_a.data, tensor_b.data)) + np.dot(
            tensor_c.data, tensor_d.data
        )

        np.testing.assert_array_equal(result_cpu.data, result_gpu.data)
        np.testing.assert_array_almost_equal(
            result_cpu.data, expected_result, decimal=6
        )


if __name__ == "__main__":
    unittest.main()
