import unittest

from feathernet.dl.initializers import he_initializer
from feathernet.dl.layers.activations import ReLU
from feathernet.dl.layers.convolution import Conv2D
from feathernet.dl.layers.core import Dense
from feathernet.dl.layers.pooling import Pooling
from feathernet.dl.losses import CrossEntropy, MeanSquaredError
from feathernet.dl.network import Network
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


class TestConv2DSerialization(unittest.TestCase):
    def test_serialization(self) -> None:
        conv2d = Conv2D(
            input_dim=1, output_dim=2, kernel_size=3, stride=1, padding=0
        )
        serialized_data = conv2d.serialize()

        self.assertEqual(serialized_data["type"], "Conv2D")
        self.assertEqual(serialized_data["input_dim"], 1)
        self.assertEqual(serialized_data["output_dim"], 2)
        self.assertEqual(serialized_data["kernel_size"], 3)
        self.assertEqual(serialized_data["stride"], 1)
        self.assertEqual(serialized_data["padding"], 0)
        self.assertEqual(serialized_data["initializer"], "random_initializer")


class TestPoolingSerialization(unittest.TestCase):
    def test_serialize_max_pool(self) -> None:
        pool = Pooling(pool_size=2, stride=2, strategy="max")
        serialized_data = pool.serialize()

        self.assertEqual(serialized_data["type"], "Pooling")
        self.assertEqual(serialized_data["pool_size"], 2)
        self.assertEqual(serialized_data["stride"], 2)
        self.assertEqual(serialized_data["strategy"], "max")

    def test_serialize_avg_pool(self) -> None:
        pool = Pooling(pool_size=2, stride=2, strategy="average")
        serialized_data = pool.serialize()

        self.assertEqual(serialized_data["type"], "Pooling")
        self.assertEqual(serialized_data["pool_size"], 2)
        self.assertEqual(serialized_data["stride"], 2)
        self.assertEqual(serialized_data["strategy"], "average")


class TestNetworkSerialization(unittest.TestCase):
    def test_serialization(self) -> None:
        network = Network(SGD(learning_rate=0.01))
        network.add(Dense(3, 2, he_initializer))
        network.add(ReLU())

        serialized_network = network.serialize()

        self.assertIsInstance(serialized_network, dict)
        self.assertEqual(len(serialized_network), 2)
        self.assertEqual(serialized_network["layers"][0]["type"], "Dense")
        self.assertEqual(
            serialized_network["layers"][0]["initializer"], "he_initializer"
        )
        self.assertEqual(serialized_network["layers"][1]["type"], "ReLU")


if __name__ == "__main__":
    unittest.main()
