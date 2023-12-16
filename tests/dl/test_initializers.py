import unittest

import numpy as np

from feathernet.dl.initializers import xavier_initializer, he_initializer


class TestInitializers(unittest.TestCase):
    def test_xavier_initializer(self) -> None:
        shape = (10, 5)
        weights = xavier_initializer(shape)
        self.assertEqual(weights.shape, shape)

        expected_stddev = np.sqrt(2 / (shape[0] + shape[1]))
        self.assertAlmostEqual(weights.std(), expected_stddev, places=1)

    def test_he_initializer(self) -> None:
        shape = (10, 5)
        weights = he_initializer(shape)
        self.assertEqual(weights.shape, shape)

        expected_stddev = np.sqrt(2 / (shape[0]))
        self.assertAlmostEqual(weights.std(), expected_stddev, places=1)


if __name__ == "__main__":
    unittest.main()
