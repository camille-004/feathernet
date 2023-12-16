import unittest

import numpy as np

from feathernet.compiler.ir import (
    IRNode,
    ModelIR,
    convert_model_to_ir,
    create_ir_from_model,
)
from feathernet.dl.layers import Conv2D, Dense
from feathernet.dl.network import Network


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

    def test_modelir_repr_formatting(self):
        ir = ModelIR()
        ir.add_node("Dense", input_dim=10, output_dim=5)
        ir.add_node("ReLU")
        ir.add_edge(0, 1)

        ir_repr = repr(ir)
        print(ir_repr)

        self.assertTrue(len(ir_repr.split("\n")) > 5)


class TestCreateIRFromModel(unittest.TestCase):
    def setUp(self) -> None:
        self.model = Network()
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


class TestModelToIRConversion(unittest.TestCase):
    def setUp(self) -> None:
        self.model = Network()
        self.model.add(
            Conv2D(
                input_dim=3, output_dim=16, kernel_size=3, stride=1, padding=1
            )
        )
        self.model.add(Dense(input_dim=16, output_dim=10))

    def test_conversion_with_optimizations(self) -> None:
        optimized_ir = convert_model_to_ir(
            self.model,
            optimizations={
                "fusion": {},
                "pruning": {"threshold": 0.01},
                "quantization": {"precision": np.int8},
            },
        )

        self.assertTrue(len(optimized_ir.nodes) > 0)

        for node in optimized_ir.nodes:
            if node.layer_type == "Dense":
                self.assertEqual(node.params["weights"].dtype, np.int8)

        for node in optimized_ir.nodes:
            if node.layer_type == "Dense":
                pruned_weights = node.params["weights"]
                num_zero_weights = np.count_nonzero(pruned_weights == 0)
                self.assertGreater(num_zero_weights, 0)


if __name__ == "__main__":
    unittest.main()
