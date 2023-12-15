from typing import Any

import numpy as np
from tqdm import tqdm

from feathernet.dl.layers.base import BaseLayer
from feathernet.dl.losses import Loss
from feathernet.dl.optimizers import Optimizer


class Network:
    def __init__(self, optimizer: Optimizer, verbose: bool = False) -> None:
        self.layers = []
        self.optimizer = optimizer
        self.verbose = verbose

    def add(self, layer: BaseLayer) -> None:
        self.layers.append(layer)

    def forward(self, X: np.ndarray) -> np.ndarray:
        output = X
        for layer in self.layers:
            input_shape = output.shape
            output = layer.forward(output)
            output_shape = output.shape
            if self.verbose:
                print(
                    f"Forward - {layer.__class__.__name__} - Input shape: {input_shape}, Output shape: {output_shape}"
                )

        return output

    def backward(
        self, output_grad: np.ndarray, clip_value: float = 1.0
    ) -> np.ndarray:
        for layer in reversed(self.layers):
            output_grad = layer.backward(output_grad)
            output_grad = np.clip(output_grad, -clip_value, clip_value)
            if self.verbose:
                print(
                    f"Backward - {layer.__class__.__name__} - Output gradient shape: {output_grad.shape}"
                )
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
        batch_size = min(batch_size, num_samples)
        num_batches = (num_samples + batch_size - 1) // batch_size

        for epoch in range(epochs):
            epoch_loss = 0
            for b in tqdm(range(num_batches), desc="Batches", leave=False):
                batch_start, batch_end = b * batch_size, min(
                    (b + 1) * batch_size, num_samples
                )
                X_batch, y_batch = (
                    X_train[batch_start:batch_end],
                    y_train[batch_start:batch_end],
                )
                output = self.forward(X_batch)
                loss = loss_function.forward(output, y_batch)
                if loss < 1:
                    print(output[:2], y_batch[:2])
                epoch_loss += loss
                output_grad = loss_function.backward(output, y_batch)
                self.backward(output_grad)

            avg_epoch_loss = epoch_loss / num_batches
            print(
                f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_epoch_loss:.4f}"
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def serialize(self) -> dict[str, Any]:
        return {
            "layers": [layer.serialize() for layer in self.layers],
            "optimizer": self.optimizer.serialize(),
        }
