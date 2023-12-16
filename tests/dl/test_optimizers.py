import unittest

import numpy as np

from feathernet.dl.optimizers import SGD


class TestOptimizers(unittest.TestCase):
    def test_sgd_update(self) -> None:
        learning_rate = 0.01
        sgd = SGD(learning_rate)

        params = np.array([1.0, 2.0, 3.0])
        grads = np.array([0.1, 0.2, 0.3])

        expected_params = params - learning_rate * grads

        sgd.update(params, grads)

        np.testing.assert_array_equal(params, expected_params)


if __name__ == "__main__":
    unittest.main()
