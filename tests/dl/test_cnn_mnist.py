import unittest

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from feathernet.dl.initializers import he_initializer
from feathernet.dl.layers.activations import ReLU
from feathernet.dl.layers.convolution import Conv2D
from feathernet.dl.layers.core import BatchNorm, Dense, Dropout
from feathernet.dl.losses import CrossEntropy
from feathernet.dl.metrics import accuracy
from feathernet.dl.network import Network
from feathernet.dl.optimizers import SGD


class TestCNNMNIST(unittest.TestCase):
    @staticmethod
    def load_mnist_dataset() -> (
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ):
        mnist = fetch_openml("mnist_784", version=1, parser="auto")
        X, y = mnist["data"], mnist["target"]

        X = X / 255.0
        X = X.values.reshape((-1, 1, 28, 28))

        enc = OneHotEncoder(sparse_output=False)
        y = enc.fit_transform(y.values.reshape(-1, 1))

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def test_cnn_mnist(self) -> None:
        X_train, X_test, y_train, y_test = self.load_mnist_dataset()

        model = Network(SGD())
        model.add(
            Conv2D(
                input_dim=1,
                output_dim=8,
                kernel_size=3,
                stride=1,
                padding=1,
                initializer=he_initializer,
            )
        )
        model.add(BatchNorm())
        model.add(ReLU())
        model.add(Dropout(rate=0.5))
        model.add(Dense(8 * 28 * 28, 128))
        model.add(ReLU())
        model.add(Dense(128, 10))

        model.train(X_train, y_train, epochs=3, loss_function=CrossEntropy())

        y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
        test_accuracy = accuracy(y_pred, y_test)
        self.assertGreater(test_accuracy, 0.7)


if __name__ == "__main__":
    unittest.main()
