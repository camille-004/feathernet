from typing import Any

import numpy as np

from feathernet.dl.layers.base import BaseLayer
from feathernet.dl.losses import Loss
from feathernet.dl.optimizers import Optimizer


class Network:
    def __init__(self, optimizer: Optimizer) -> None:
        self.layers = []
        self.optimizer = optimizer

    def add(self, layer: BaseLayer) -> None:
        self.layers.append(layer)

    def forward(self, X: np.ndarray) -> np.ndarray:
        output = X
        for layer in self.layers:
            output = layer.forward(output)
            print(layer.__class__.__name__, X.shape)

        return output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            output_grad = layer.backward(output_grad)
            if hasattr(layer, "weights"):
                self.optimizer.update(layer.weights, layer.weights_grad)
                self.optimizer.update(layer.bias, layer.bias_grad)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int,
        loss_function: Loss,
        batch_size: int = 32,
    ) -> None:
        num_samples = X_train.shape[0]
        num_batches = num_samples // batch_size

        for epoch in range(epochs):
            for b in range(num_batches):
                batch_start, batch_end = b * batch_size, (b + 1) * batch_size
                X_batch, y_batch = (
                    X_train[batch_start:batch_end],
                    y_train[batch_start:batch_end],
                )

                output = self.forward(X_batch)
                loss = loss_function.forward(output, y_batch)
                output_grad = loss_function.backward(output, y_batch)
                self.backward(output_grad)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def serialize(self) -> dict[str, Any]:
        return {
            "layers": [layer.serialize() for layer in self.layers],
            "optimizer": self.optimizer.serialize(),
        }
