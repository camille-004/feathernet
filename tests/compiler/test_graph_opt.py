import unittest

import numpy as np

from feathernet.compiler.ir import IRNode, ModelIR
from feathernet.compiler.graph_opt import (
    fuse_layers,
    fuse_layers_in_model,
    prune_model,
    quantize_model,
)
from feathernet.dl.layers import Dense


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


class TestPruning(unittest.TestCase):
    def test_prune_model(self):
        dense_weights = np.array(
            [[0.01, -0.02, 0.03, 0, 0.05] for _ in range(10)]
        )

        model_ir = ModelIR()
        model_ir.add_node("Dense", weights=dense_weights)

        prune_model(model_ir, threshold=0.03)

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

        quantize_model(model_ir, precision=np.int8)

        quantized_weights = model_ir.nodes[0].params["weights"]
        self.assertTrue(quantized_weights.dtype == np.int8)


if __name__ == "__main__":
    unittest.main()
