from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Loss(ABC):
    @abstractmethod
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        pass

    @abstractmethod
    def backward(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> np.ndarray:
        """Calculate the gradient of the loss w.r.t. the predictions."""
        pass

    def serialize(self) -> dict[str, Any]:
        return {"type": self.__class__.__name__}


class MeanSquaredError(Loss):
    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        return np.mean((predictions - targets) ** 2)

    def backward(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> np.ndarray:
        return 2 * (predictions - targets) / targets.size


class CrossEntropy(Loss):
    def forward(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> np.ndarray:
        epsilon = 1e-6
        predictions = np.clip(predictions, epsilon, 1 - epsilon)

        N = predictions.shape[0]
        ce_loss = -np.sum(targets * np.log(predictions)) / N
        return ce_loss

    def backward(
        self, predictions: np.ndarray, targets: np.ndarray
    ) -> np.ndarray:
        epsilon = 1e-6
        predictions = np.clip(predictions, epsilon, 1 - epsilon)

        N = predictions.shape[0]
        grad = -targets / predictions / N
        return grad
