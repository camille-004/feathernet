import unittest

import numpy as np

from feathernet.dl.utils import prune_weights, quantize_weights


class TestPruningWeights(unittest.TestCase):
    def test_prune_weights(self) -> None:
        weights = np.array([[0.01, -0.02, 0.03, 0, 0.05] for _ in range(10)])
        pruned_weights = prune_weights(weights, threshold=0.03)

        for weight in pruned_weights.flatten():
            self.assertTrue(weight == 0 or abs(weight) >= 0.03)


class TestQuantizeWeights(unittest.TestCase):
    def test_quantize_weights(self) -> None:
        weights = np.random.randn(10, 5).astype(np.float32)
        quantized_weights = quantize_weights(weights, precision=np.int8)

        self.assertTrue(quantized_weights.dtype == np.int8)

        original_range = np.max(weights) - np.min(weights)
        quantized_range = np.max(quantized_weights) - np.min(quantized_weights)
        self.assertLess(quantized_range, original_range)


if __name__ == "__main__":
    unittest.main()
