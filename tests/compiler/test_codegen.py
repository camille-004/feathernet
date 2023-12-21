import unittest
from pathlib import Path

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

        self.assertIn("@WEIGHTS @", generated_code)
        self.assertIn("@BIASES @", generated_code)


class TestExecutor(unittest.TestCase):
    def setUp(self) -> None:
        input_dim, output_dim = 4, 2
        dense_layer = Dense(input_dim, output_dim)
        dense_params = dense_layer.serialize()
        dense_node = IRNode("Dense", **dense_params)

        self.codegen = CodeGen(dense_node)
        self.cpp_source = Path("generated_dense.cpp")
        self.binary_path = Path("dense_executable")

    def test_compile_and_execute(self):
        generated_code = self.codegen.generate()

        with open(self.cpp_source, "w") as f:
            f.write(generated_code)

        executor = Executor(self.cpp_source)

        executor.compile()
        output = executor.exec()
        print(output)


if __name__ == "__main__":
    unittest.main()
