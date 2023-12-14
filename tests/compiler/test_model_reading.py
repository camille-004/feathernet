import unittest

import numpy as np

from feathernet.dl.layers.activations import ReLU
from feathernet.dl.layers.core import Dense
from feathernet.dl.losses import MeanSquaredError, CrossEntropy
from feathernet.dl.network.base import Network
from feathernet.dl.optimizers import SGD


class TestMeanSquaredErrorSerialization(unittest.TestCase):
    def test_serialization(self) -> None:
        mse = MeanSquaredError()
        serialized_mse = mse.serialize()
        self.assertEqual(serialized_mse["type"], "MeanSquaredError")


class TestCrossEntropySerialization(unittest.TestCase):
    def test_serialization(self) -> None:
        ce = CrossEntropy()
        serialized_ce = ce.serialize()
        self.assertEqual(serialized_ce["type"], "CrossEntropy")


class TestSGDSerialization(unittest.TestCase):
    def test_serialization(self) -> None:
        learning_rate = 0.01
        sgd = SGD(learning_rate)
        serialized_sgd = sgd.serialize()
        self.assertEqual(serialized_sgd["type"], "SGD")
        self.assertEqual(serialized_sgd["learning_rate"], learning_rate)


class TestDenseSerialization(unittest.TestCase):
    def test_serialization(self) -> None:
        input_dim, output_dim = 3, 2
        dense = Dense(input_dim, output_dim)
        serialized_data = dense.serialize()

        self.assertEqual(serialized_data["type"], "Dense")
        self.assertEqual(serialized_data["input_dim"], input_dim)
        self.assertEqual(serialized_data["output_dim"], output_dim)


class TestNetworkSerialization(unittest.TestCase):
    def test_serialization(self) -> None:
        network = Network(SGD(learning_rate=0.01))
        network.add(Dense(3, 2))
        network.add(ReLU())

        serialized_network = network.serialize()

        self.assertIsInstance(serialized_network, list)
        self.assertEqual(len(serialized_network), 2)
        self.assertEqual(serialized_network[0]["type"], "Dense")
        self.assertEqual(serialized_network[1]["type"], "ReLU")