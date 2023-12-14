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
    ) -> None:
        for epoch in range(epochs):
            output = self.forward(X_train)
            loss = loss_function.forward(output, y_train)
            output_grad = loss_function.backward(output, y_train)
            self.backward(output_grad)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")

    def serialize(self) -> list[dict[str, Any]]:
        return [layer.serialize() for layer in self.layers]
