import unittest

from feathernet.compiler.ir import ModelIR, create_ir_from_model
from feathernet.dl.layers.activations import ReLU
from feathernet.dl.layers.core import Dense
from feathernet.dl.network import Network
from feathernet.dl.optimizers import SGD


class TestModelIR(unittest.TestCase):
    def test_add_layer(self):
        ir = ModelIR()
        ir.add_layer("Dense", input_dim=10, output_dim=5)

        self.assertEqual(len(ir.layers), 1)
        self.assertEqual(ir.layers[0]["type"], "Dense")
        self.assertEqual(ir.layers[0]["params"]["input_dim"], 10)
        self.assertEqual(ir.layers[0]["params"]["output_dim"], 5)

    def test_create_ir_from_model(self):
        model = Network(SGD())
        model.add(Dense(10, 5))
        model.add(ReLU())

        ir = create_ir_from_model(model)

        self.assertEqual(len(ir.layers), 2)
        self.assertEqual(ir.layers[0]["type"], "Dense")
        self.assertEqual(ir.layers[1]["type"], "ReLU")
        self.assertIsNotNone(ir.optimizer)
        self.assertEqual(ir.optimizer["type"], "SGD")


if __name__ == "__main__":
    unittest.main()
