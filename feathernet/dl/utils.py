import numpy as np


def prune_weights(weights: np.ndarray, threshold: float) -> np.ndarray:
    pruned_weights = weights.copy()
    pruned_weights[np.abs(pruned_weights) < threshold] = 0
    return pruned_weights


def quantize_weights(weights: np.ndarray, precision: type) -> np.ndarray:
    max_val = np.max(np.abs(weights))
    scale = max_val / 127  # 127 for 8-bit precision
    quantized_weights = np.round(weights / scale).astype(precision)
    return quantized_weights
