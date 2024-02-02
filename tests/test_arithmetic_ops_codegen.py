import unittest

from feathernet.ops.basic import add, div, mult, sub
from feathernet.translator.ir_nodes import IRLiteral, IROperation
from feathernet.translator.translators import IRTranslator


def ir_to_cuda(ir_node: IROperation) -> str:
    return IRTranslator.translate_node(ir_node)


class TestArithmeticOpsCodegen(unittest.TestCase):
    def assertCUDAEqual(self, cuda: str, expected_expression: str) -> None:
        start = cuda.find("{") + 1
        end = cuda.find("}")
        extracted = cuda[start:end].strip()
        self.assertEqual(extracted, expected_expression)

    def test_add_operation(self) -> None:
        add_op = add(IRLiteral(5), IRLiteral(10))
        cuda = ir_to_cuda(add_op)

        self.assertCUDAEqual(cuda, "(5 + 10);")

    def test_sub_operation(self) -> None:
        sub_op = sub(IRLiteral(5), IRLiteral(10))
        cuda = ir_to_cuda(sub_op)
        self.assertCUDAEqual(cuda, "(5 - 10);")

    def test_mult_operation(self) -> None:
        mult_op = mult(IRLiteral(5), IRLiteral(10))
        cuda = ir_to_cuda(mult_op)
        self.assertCUDAEqual(cuda, "(5 * 10);")

    def test_div_operation(self) -> None:
        div_op = div(IRLiteral(5), IRLiteral(10))
        cuda = ir_to_cuda(div_op)
        self.assertCUDAEqual(cuda, "(5 / 10);")


if __name__ == "__main__":
    unittest.main()
