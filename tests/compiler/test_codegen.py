import unittest

from feathernet.compiler.codegen import LayerCodeGen
from feathernet.compiler.ir import IRNode
from feathernet.dl.layers import Dense


class TestLayerCodeGen(unittest.TestCase):
    def setUp(self) -> None:
        input_dim, output_dim = 4, 2
        dense_layer = Dense(input_dim, output_dim)
        dense_params = dense_layer.serialize()
        self.dense_node = IRNode("Dense", **dense_params)

    def test_generate_dense_code(self):
        codegen = LayerCodeGen(self.dense_node)
        generated_code = codegen.generate_layer_code("layer0")

        self.assertIn("void layerFunction_layer0", generated_code)

        self.assertNotIn("@INPUT_DIM @", generated_code)
        self.assertNotIn("@OUTPUT_DIM @", generated_code)
        self.assertNotIn("@WEIGHTS @", generated_code)
        self.assertNotIn("@BIASES @", generated_code)


if __name__ == "__main__":
    unittest.main()
