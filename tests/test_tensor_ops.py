import unittest

import numpy as np

from feathernet.tensor import Tensor


class TestTensorOps(unittest.TestCase):
    def test_addition(self) -> None:
        tensor1 = Tensor(5)
        tensor2 = Tensor(3)
        result_tensor = tensor1 + tensor2
        self.assertEqual(result_tensor.data, 8, "Addition operation failed.")

    def test_subtraction(self) -> None:
        tensor1 = Tensor(5)
        tensor2 = Tensor(3)
        result_tensor = tensor1 - tensor2
        self.assertEqual(
            result_tensor.data, 2, "Subtraction operation failed."
        )

    def test_multiplication(self) -> None:
        tensor1 = Tensor(5)
        tensor2 = Tensor(3)
        result_tensor = tensor1 * tensor2
        self.assertEqual(
            result_tensor.data, 15, "Multiplication operation failed."
        )

    def test_division(self) -> None:
        tensor1 = Tensor(6)
        tensor2 = Tensor(3)
        result_tensor = tensor1 / tensor2
        self.assertEqual(result_tensor.data, 2, "Division operation failed.")

    def test_division_by_zero(self) -> None:
        tensor1 = Tensor(5)
        tensor2 = Tensor(0)
        result_tensor = tensor1 / tensor2
        self.assertTrue(
            np.isinf(result_tensor.data) or np.isnan(result_tensor.data),
            "Division by zero should result in inf or NaN.",
        )


if __name__ == "__main__":
    unittest.main()
