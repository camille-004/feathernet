from typing import Literal

import numpy as np
import pycuda.autoinit  # noqa

from feathernet.ir.core import Node
from feathernet.ir.dispatch_tables import CPU_OPERATIONS, GPU_OPERATIONS
from feathernet.tensor import Tensor

DeviceType = Literal["cpu", "gpu"]
ShapeType = tuple[int, ...]


class GraphExecutor:
    def __init__(self) -> None:
        self.cache: dict[Node, np.ndarray] = {}

    def evaluate(self, entity: Node | Tensor) -> np.ndarray:
        if isinstance(entity, Tensor):
            operation = self._get_tensor_op(entity)
            if operation:
                return self.evaluate(operation)
            elif entity.data is not None:
                return entity.data
            else:
                raise ValueError(
                    "Tensor is not part of a computational graph and has no data."
                )

        if isinstance(entity, tuple):
            return tuple(self.evaluate(operand) for operand in entity)

        if entity in self.cache:
            return self.cache[entity]

        operands = [self.evaluate(operand) for operand in entity.operands]

        if hasattr(entity, "device") and entity.device == "gpu":
            result = self._exec_gpu(entity, operands)
        else:
            result = self._exec_cpu(entity, operands)

        self.cache[entity] = result
        return result

    def _exec_cpu(self, node: Node | Tensor, operands: list) -> np.ndarray:
        op_func = CPU_OPERATIONS.get(type(node))
        if op_func:
            return op_func(operands)
        else:
            raise NotImplementedError(
                f"CPU operation '{type(node).__name__}' not implemented."
            )

    def _exec_gpu(self, node: Node | Tensor, operands: list) -> np.ndarray:
        op_func = GPU_OPERATIONS.get(type(node))
        if op_func:
            return op_func(operands)
        else:
            raise NotImplementedError(
                f"GPU operation '{type(node).__name__}' not implemented."
            )

    def _get_tensor_op(self, tensor: Tensor) -> Node | None:
        return tensor._ir_node
