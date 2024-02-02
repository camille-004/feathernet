from typing import Callable

import numpy as np
from pycuda.compiler import SourceModule

from feathernet.ops.basic import add, div, mult, sub
from feathernet.translator.ir_nodes import IRLiteral, IROperation, IRVariable
from feathernet.translator.translators import IRTranslator


class Tensor:
    def __init__(self, data: float | list | np.ndarray) -> None:
        self.data = data
        self.grad: Tensor = None

    def __add__(self, other: "Tensor") -> "Tensor":
        return self._op(add, self, other)

    def __sub__(self, other: "Tensor") -> "Tensor":
        return self._op(sub, self, other)

    def __mul__(self, other: "Tensor") -> "Tensor":
        return self._op(mult, self, other)

    def __div__(self, other: "Tensor") -> "Tensor":
        return self._op(div, self, other)

    def _op(
        self,
        operation: Callable[[IROperation, IROperation], IROperation],
        left: "Tensor",
        right: "Tensor",
    ):
        ir_left = self._to_ir(left)
        ir_right = self._to_ir(right)
        ir_res = operation(ir_left, ir_right)
        result = exec_op(ir_res)
        return Tensor(result)

    def _to_ir(self, tensor: "Tensor") -> IROperation:
        if isinstance(tensor.data, (int, float)):
            return IRLiteral(tensor.data)
        elif isinstance(tensor.data, IRVariable):
            return tensor.data
        else:
            raise NotImplementedError(
                "Conversion for the given tensor type is not implemented."
            )


def exec_op(ir_op: IROperation) -> np.ndarray:
    cuda = IRTranslator.translate_node(ir_op)
    print("Generated CUDA code:\n", cuda)
    mod = SourceModule(cuda)
    compute = mod.get_function("compute")
    output = np.empty_like(ir_op)
    compute(cuda.Out(output), block=(1, 1, 1), grid=(1, 1))
    return output
