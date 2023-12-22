import unittest

import numpy as np

from feathernet.compiler.graph_opt import (
    Fusion,
    Pruning,
    Quantization,
    can_fuse,
    update_edge,
)
from feathernet.compiler.ir import IRNode, ModelIR
from feathernet.dl.layers import Dense


class TestLayerFusion(unittest.TestCase):
    def test_fuse_layers_conv2d_batch_norm(self) -> None:
        conv2D_node = IRNode(
            "Conv2D", input_dim=28, output_dim=28, kernel_size=3, stride=1
        )
        batch_norm_node = IRNode("BatchNorm")

        model_ir = ModelIR()
        model_ir.add_node(conv2D_node.layer_type, **conv2D_node.params)
        model_ir.add_node(batch_norm_node.layer_type, **batch_norm_node.params)

        fusion_optimizer = Fusion()
        fusion_optimizer.optimize(model_ir)

        self.assertEqual(len(model_ir.nodes), 1)
        self.assertIsInstance(model_ir.nodes[0], IRNode)
        self.assertEqual(model_ir.nodes[0].layer_type, "Conv2D")

    def test_fuse_layers_successive_dense(self) -> None:
        input_dim = 10
        hidden_dim = 5
        output_dim = 3

        dense_layer_1 = Dense(input_dim=input_dim, output_dim=hidden_dim)
        dense_layer_2 = Dense(input_dim=hidden_dim, output_dim=output_dim)

        model_ir = ModelIR()
        model_ir.add_node("Dense", **dense_layer_1.serialize())
        model_ir.add_node("Dense", **dense_layer_2.serialize())

        fusion_optimizer = Fusion()
        fusion_optimizer.optimize(model_ir)

        self.assertEqual(len(model_ir.nodes), 1)
        self.assertIsInstance(model_ir.nodes[0], IRNode)
        self.assertEqual(model_ir.nodes[0].layer_type, "Dense")
        self.assertEqual(model_ir.nodes[0].params["input_dim"], input_dim)
        self.assertEqual(model_ir.nodes[0].params["output_dim"], output_dim)


class TestFusionFunctions(unittest.TestCase):
    def test_can_fuse_conv2d_batchnorm(self) -> None:
        conv2d_node = IRNode("Conv2D")
        bn_node = IRNode("BatchNorm")
        self.assertTrue(can_fuse(conv2d_node, bn_node))

    def test_can_fuse_layer_activation(self) -> None:
        conv2d_node = IRNode("Conv2D")
        activation_node = IRNode("ReLU")
        self.assertTrue(can_fuse(conv2d_node, activation_node))

    def test_can_fuse_dense_dense(self) -> None:
        dense_node1 = IRNode("Dense", input_dim=4, output_dim=2)
        dense_node2 = IRNode("Dense", input_dim=2, output_dim=4)
        self.assertTrue(can_fuse(dense_node1, dense_node2))

    def test_cannot_fuse_different_layer_types(self) -> None:
        conv2d_node = IRNode("Conv2D")
        dense_node = IRNode("Dense")
        self.assertFalse(can_fuse(conv2d_node, dense_node))

    def test_update_edge(self) -> None:
        ir = ModelIR()
        ir.add_edge(0, 1)
        ir.add_edge(1, 2)
        ir.add_edge(2, 3)

        update_edge(ir, 1)

        expected_edges = [{"from": 0, "to": 1}, {"from": 1, "to": 3}]
        updated_edges = ir.edges

        self.assertEqual(expected_edges, updated_edges)


class TestPruning(unittest.TestCase):
    def test_prune_model(self):
        dense_weights = np.array(
            [[0.01, -0.02, 0.03, 0, 0.05] for _ in range(10)]
        )

        model_ir = ModelIR()
        model_ir.add_node("Dense", weights=dense_weights)

        pruning_optimizer = Pruning(threshold=0.03)
        pruning_optimizer.optimize(model_ir)

        pruned_weights = model_ir.nodes[0].params["weights"]
        for weight in pruned_weights.flatten():
            self.assertTrue(weight == 0 or abs(weight) >= 0.03)

        num_non_zero = np.count_nonzero(pruned_weights)
        self.assertLess(num_non_zero, dense_weights.size)


class TestQuantization(unittest.TestCase):
    def test_quantize_model(self) -> None:
        dense_layer = Dense(10, 5)
        dense_layer.weights = np.random.randn(10, 5).astype(np.float32)

        model_ir = ModelIR()
        model_ir.add_node("Dense", weights=dense_layer.weights)

        quantization_optimizer = Quantization(precision=np.int8)
        quantization_optimizer.optimize(model_ir)

        quantized_weights = model_ir.nodes[0].params["weights"]
        self.assertTrue(quantized_weights.dtype == np.int8)


if __name__ == "__main__":
    unittest.main()
