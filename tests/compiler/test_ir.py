import unittest

from feathernet.compiler.fusion import fuse_layers, fuse_layers_in_model
from feathernet.compiler.ir import IRNode, ModelIR, create_ir_from_model
from feathernet.dl.layers.convolution import Conv2D
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


class TestLayerFusion(unittest.TestCase):
    def test_fuse_layers_conv2d_batch_norm(self) -> None:
        conv2D_node = IRNode(
            "Conv2D", input_dim=28, output_dim=28, kernel_size=3, stride=1
        )
        batch_norm_node = IRNode("BatchNorm")
        fused_layer = fuse_layers(conv2D_node, batch_norm_node)
        self.assertIsInstance(fused_layer, IRNode)


class TestModelLayerFusion(unittest.TestCase):
    def test_fuse_layers_in_model(self) -> None:
        model_ir = ModelIR()

        model_ir.add_node(
            "Conv2D", input_dim=28, output_dim=28, kernel_size=3, stride=1
        )
        model_ir.add_node("BatchNorm")

        fuse_layers_in_model(model_ir)

        self.assertEqual(len(model_ir.nodes), 1)
        self.assertIsInstance(model_ir.nodes[0], IRNode)
        self.assertEqual(model_ir.nodes[0].layer_type, "Conv2D")


if __name__ == "__main__":
    unittest.main()
