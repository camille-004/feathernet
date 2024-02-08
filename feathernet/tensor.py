from typing import Literal

import numpy as np
import pycuda.autoinit  # noqa
from numpy.typing import ArrayLike

from feathernet.ir.core import Node
from feathernet.ir.dispatch_tables import CPU_OPERATIONS, GPU_OPERATIONS
from feathernet.ir.ops import AddNode

DeviceType = Literal["cpu", "gpu"]


class Tensor:
    def __init__(
        self, data: ArrayLike | Node, device: DeviceType = "cpu"
    ) -> None:
        self.device = device
        if isinstance(data, Node):
            self._ir_node: Node | None = data
            self._data = None
            self._shape = None
            self._dtype = None
        else:
            self._ir_node = None
            self._data = np.array(data)
            self._shape = self.data.shape
            self._dtype = self.data.dtype

    @property
    def data(self) -> np.ndarray:
        if self._data is None and self._ir_node is not None:
            return self._ir_node.data
        else:
            return self._data

    @property
    def shape(self) -> tuple[int, ...] | None:
        if self._data is not None:
            return self._data.shape
        else:
            return None

    @property
    def dtype(self) -> np.dtype | None:
        if self._data is not None:
            return self._data.dtype
        else:
            return None

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, device={self.device})\n{self.data}"  # noqa

    def _prepare(self, other: "Tensor", operation: str) -> "Tensor":
        if not isinstance(other, Tensor):
            raise ValueError(f"Operand must be a Tensor for {operation}.")
        if self.shape != other.shape:
            raise ValueError(
                f"Tensors must have the same shape for {operation}."
            )

        if self.device != other.device:
            other = other.to(self.device)

        return other

    def to(self, device: DeviceType) -> "Tensor":
        if device == self.device:
            return self

        new = (
            Tensor(self.data, device=device)
            if self.data is not None
            else Tensor(self._ir_node, device=device)
        )
        new.ir_node = self._ir_node
        return new

    def evaluate(self) -> "Tensor":
        executor = GraphExecutor()
        result_data = executor.evaluate(self._ir_node)
        return Tensor(result_data, device=self.device)

    def __add__(self, other: "Tensor") -> "Tensor":
        other = self._prepare(other, "addition")
        add_node = AddNode([self, other])
        return Tensor(add_node, device=self.device).evaluate()


class GraphExecutor:
    def __init__(self) -> None:
        self.cache: dict[Node, np.ndarray] = {}

    def evaluate(self, node: Node | Tensor) -> np.ndarray:
        # If node is a tensor with raw data.
        if node is None or (
            isinstance(node, Tensor) and node.data is not None
        ):
            return node.data

        # Avoid redundant computation.
        if node in self.cache:
            return self.cache[node]

        operands = [self.evaluate(operand) for operand in node.operands]

        if hasattr(operands[0], "device") and operands[0].device == "gpu":
            result = self._exec_gpu(node, operands)
        else:
            result = self._exec_cpu(node, operands)

        self.cache[node] = result
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
        op_func = GPU_OPERATIONS.get(type(Node))
        if op_func:
            return op_func(operands)
        else:
            raise NotImplementedError(
                f"GPU operation '{type(node).__name__}' not implemented."
            )
