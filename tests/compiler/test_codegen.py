import unittest

from feathernet.compiler.codegen import CodeGen, Executor
from feathernet.compiler.ir import IRNode
from feathernet.dl.layers import Dense


class TestCodeGen(unittest.TestCase):
    def setUp(self) -> None:
        input_dim, output_dim = 4, 2
        dense_layer = Dense(input_dim, output_dim)
        dense_params = dense_layer.serialize()
        self.dense_node = IRNode("Dense", **dense_params)

    def test_generate_dense_code(self):
        codegen = CodeGen(self.dense_node)
        generated_code = codegen.generate()

        self.assertIn(
            "denseLayer(input, output, weights, bias);", generated_code
        )

        self.assertNotIn("@INPUT_DIM @", generated_code)
        self.assertNotIn("@OUTPUT_DIM @", generated_code)
        self.assertNotIn("@WEIGHTS @", generated_code)
        self.assertNotIn("@BIASES @", generated_code)


class TestExecutor(unittest.TestCase):
    def setUp(self) -> None:
        self.input_dim, self.output_dim = 4, 2
        dense_layer = Dense(self.input_dim, self.output_dim)
        dense_params = dense_layer.serialize()
        dense_node = IRNode("Dense", **dense_params)

        self.codegen = CodeGen(dense_node)

    def test_compile_and_execute(self):
        generated_code = self.codegen.generate()
        executor = Executor(generated_code)

        if not executor.compile():
            self.fail("Compilation failed.")

        output = executor.exec()
        executor.cleanup()

        try:
            output_values = [float(value) for value in output.strip().split()]
        except ValueError:
            self.fail("Output values are not valid floats.")

        self.assertEqual(len(output_values), self.output_dim)
        self.assertIsInstance(output_values, list)


if __name__ == "__main__":
    unittest.main()
