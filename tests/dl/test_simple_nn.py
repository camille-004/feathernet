import unittest

from feathernet.dl.layers.activations import ReLU
from feathernet.dl.layers.core import Dense
from feathernet.dl.losses import CrossEntropy
from feathernet.dl.optimizers import Optimizer, SGD
from feathernet.dl.network.base import Network
from tests.utils.dataset import make_dataset


class TestableNetwork(Network):
    def __init__(self, optimizer: Optimizer) -> None:
        super(TestableNetwork, self).__init__(optimizer)
        self.add(Dense(10, 5))
        self.add(ReLU())
        self.add(Dense(5, 2))


class TestSimpleNN(unittest.TestCase):
    def setUp(self) -> None:
        self.X, self.y = make_dataset()
        self.model = TestableNetwork(SGD(learning_rate=0.01))

    def test_forward_pass(self):
        predictions = self.model.forward(self.X)
        self.assertEqual(predictions.shape, (self.X.shape[0], 2))

    def test_training(self):
        loss_function = CrossEntropy()
        initial_loss = loss_function.forward(
            self.model.forward(self.X), self.y
        )
        self.model.train(
            self.X,
            self.y,
            epochs=1,
            loss_function=loss_function,
        )
        post_training_loss = loss_function.forward(
            self.model.forward(self.X), self.y
        )
        self.assertLess(
            post_training_loss,
            initial_loss,
            "Training should reduce the loss.",
        )


if __name__ == "__main__":
    unittest.main()
