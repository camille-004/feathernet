import unittest

from feathernet.tensor import Tensor


class TestTensor(unittest.TestCase):
    def test_initialization(self) -> None:
        scalar_tesnor = Tensor(5)
        self.assertEqual(scalar_tesnor.data, 5)

        list_tensor = Tensor([1, 2, 3])
        self.assertEqual(list_tensor.data, [1, 2, 3])

    def test_addition(self):
        tensor1 = Tensor(5)
        tensor2 = Tensor(3)
        result = tensor1 + tensor2
        self.assertEqual(result.data, 8)


if __name__ == "__main__":
    unittest.main()
