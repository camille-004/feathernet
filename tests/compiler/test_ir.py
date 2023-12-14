import unittest

from feathernet.compiler.ir import ModelIR, create_ir_from_model
from feathernet.dl.layers.activations import ReLU
from feathernet.dl.layers.core import Dense
from feathernet.dl.network import Network
from feathernet.dl.optimizers import SGD


class TestModelIR(unittest.TestCase):
    def test_add_node(self) -> None:
        ir = ModelIR()
        ir.add_node("Dense", input_dim=10, output_dim=5)

        self.assertEqual(len(ir.nodes), 1)
        self.assertEqual(ir.nodes[0]["type"], "Dense")
        self.assertEqual(ir.nodes[0]["params"]["input_dim"], 10)
        self.assertEqual(ir.nodes[0]["params"]["output_dim"], 5)

    def test_add_edge(self) -> None:
        ir = ModelIR()
        ir.add_node("Dense", input_dim=10, output_dim=5)
        ir.add_node("ReLU")
        ir.add_edge(0, 1)

        self.assertEqual(len(ir.edges), 1)
        self.assertEqual(ir.edges[0]["from"], 0)
        self.assertEqual(ir.edges[0]["to"], 1)

    def test_create_ir_from_model(self) -> None:
        model = Network(SGD())
        model.add(Dense(10, 5))
        model.add(ReLU())

        ir = create_ir_from_model(model)

        self.assertEqual(len(ir.nodes), 2)
        self.assertEqual(ir.nodes[0]["type"], "Dense")
        self.assertEqual(ir.nodes[1]["type"], "ReLU")
        self.assertEqual(ir.edges[0]["from"], 0)
        self.assertEqual(ir.edges[0]["to"], 1)
        self.assertIsNotNone(ir.optimizer)
        self.assertEqual(ir.optimizer["type"], "SGD")


if __name__ == "__main__":
    unittest.main()
