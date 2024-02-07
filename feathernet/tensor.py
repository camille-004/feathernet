from typing import Callable

import numpy as np
import pycuda.autoinit  # noqa
from pycuda import driver as cuda
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

    def __truediv__(self, other: "Tensor") -> "Tensor":
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
    cuda_code = IRTranslator.translate_node(ir_op)
    # print("Generated CUDA code:\n", cuda_code)
    mod = SourceModule(cuda_code)
    kernel = mod.get_function("compute")
    a_np = np.asarray(ir_op.left.value, dtype=np.float32)
    b_np = np.asarray(ir_op.right.value, dtype=np.float32)
    a_gpu = cuda.mem_alloc(a_np.nbytes)
    b_gpu = cuda.mem_alloc(b_np.nbytes)
    result_gpu = cuda.mem_alloc(a_np.nbytes)
    cuda.memcpy_htod(a_gpu, a_np)
    cuda.memcpy_htod(b_gpu, b_np)
    kernel(a_gpu, b_gpu, result_gpu, block=(a_np.size, 1, 1), grid=(1, 1))
    result_np = np.empty_like(a_np)
    cuda.memcpy_dtoh(result_np, result_gpu)
    return result_np
