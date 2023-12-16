import unittest

from feathernet.compiler.ir import IRNode, ModelIR, create_ir_from_model
from feathernet.dl.layers import Conv2D
from feathernet.dl.network import Network
from feathernet.dl.optimizers import SGD


class TestIRNode(unittest.TestCase):
    def test_irnode_initialization(self) -> None:
        node = IRNode("Conv2D", kernel_size=3, stride=1)
        self.assertEqual(node.layer_type, "Conv2D")
        self.assertEqual(node.params, {"kernel_size": 3, "stride": 1})

    def test_irnode_setters(self) -> None:
        node = IRNode("Conv2D", kernel_size=3, stride=1)
        node.layer_type = "Dense"
        node.params = {"input_dim": 3}
        self.assertEqual(node.layer_type, "Dense")
        self.assertEqual(node.params, {"input_dim": 3})


class TestModelIR(unittest.TestCase):
    def test_modelir_initialization(self) -> None:
        ir = ModelIR()
        self.assertEqual(ir.nodes, [])
        self.assertEqual(ir.edges, [])
        self.assertIsNone(ir.optimizer)

    def test_modelir_add_node(self) -> None:
        ir = ModelIR()
        ir.add_node("Conv2D", kernel_size=3, stride=1)
        self.assertEqual(len(ir.nodes), 1)
        self.assertIsInstance(ir.nodes[0], IRNode)
        self.assertEqual(ir.nodes[0].layer_type, "Conv2D")

    def test_modelir_add_edge(self) -> None:
        ir = ModelIR()
        ir.add_edge(0, 1)
        self.assertEqual(len(ir.edges), 1)
        self.assertEqual(ir.edges[0], {"from": 0, "to": 1})


class TestCreateIRFromModel(unittest.TestCase):
    def setUp(self) -> None:
        self.model = Network(SGD())
        self.model.add(
            Conv2D(
                input_dim=3, output_dim=16, kernel_size=3, stride=1, padding=1
            )
        )
        self.model.add(
            Conv2D(
                input_dim=16, output_dim=32, kernel_size=3, stride=1, padding=1
            )
        )

    def test_create_ir_from_model(self) -> None:
        ir = create_ir_from_model(self.model)

        self.assertIsInstance(ir, ModelIR)
        self.assertEqual(len(ir.nodes), 2)
        self.assertEqual(ir.nodes[0].layer_type, "Conv2D")
        self.assertEqual(ir.nodes[1].layer_type, "Conv2D")
        self.assertEqual(len(ir.edges), 1)
        self.assertEqual(ir.edges[0], {"from": 0, "to": 1})


if __name__ == "__main__":
    unittest.main()
