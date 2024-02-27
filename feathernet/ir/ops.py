from feathernet.common.enums import OpType
from feathernet.ir.core import Node


class AddNode(Node):
    def __init__(self, operands: list["Node"] = []) -> None:
        super().__init__(operands, op_type=OpType.ADD)


class MatMulNode(Node):
    def __init__(self, operands: list["Node"] = []) -> None:
        super().__init__(operands, op_type=OpType.MATMUL)
