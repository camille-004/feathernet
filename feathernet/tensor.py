from typing import Literal

import numpy as np
import pycuda.autoinit  # noqa
from numpy.typing import ArrayLike

from feathernet.ir.core import Node
from feathernet.ir.dispatch_tables import CPU_OPERATIONS, GPU_OPERATIONS
from feathernet.ir.ops import AddNode, MatMulNode

DeviceType = Literal["cpu", "gpu"]


class Tensor:
    def __init__(
        self, data: ArrayLike | Node, device: DeviceType = "cpu"
    ) -> None:
        self._device = device
        if isinstance(data, Node):
            self._ir_node: Node | None = data
            self._data = None
            self._shape = None
            self._dtype = None
        else:
            self._ir_node = None
            if np.isscalar(data):
                self._data = data
                self._shape = ()
                self._dtype = type(data)
            else:
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
            if np.isscalar(self._data):
                return ()
            return self._data.shape
        else:
            return None

    @property
    def dtype(self) -> np.dtype | None:
        if self._data is not None:
            return self._data.dtype
        else:
            return None

    @property
    def device(self) -> DeviceType:
        return self._device

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, device={self.device})\n{self.data}"  # noqa

    def _prepare(self, other: "Tensor", operation: str) -> "Tensor":
        if not isinstance(other, Tensor):
            raise ValueError(f"Operand must be a Tensor for {operation}.")

        if operation != "matmul":
            if self.shape != other.shape:
                raise ValueError(
                    f"Tensors must have the same shape for {operation}."
                )
        else:
            # Check for vectors.
            self_shape = (
                self.shape if len(self.shape) > 1 else (1, self.shape[0])
            )
            other_shape = (
                other.shape if len(other.shape) > 1 else (other.shape[0], 1)
            )

            if self_shape[-1] != other_shape[-2]:
                raise ValueError(
                    f"Shape mismatch for matrix multiplication: {self.shape} and {other.shape}"
                )

        if self.device != other.device:
            other = self.tensors_to_device([other], self.device)[0]

        return other

    @staticmethod
    def tensors_to_device(
        tensors: list["Tensor"], device: DeviceType
    ) -> list["Tensor"]:
        return [tensor.to(device) for tensor in tensors]

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
        if np.isscalar(result_data) or np.prod(result_data.shape) == 1:
            result_data = result_data.item()
        return Tensor(result_data, device=self.device)

    def matmul(self, other: "Tensor") -> "Tensor":
        other = self._prepare(other, "matmul")
        matmul_node = MatMulNode([self, other])
        return Tensor(matmul_node, device=self.device).evaluate()

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return self.matmul(other)

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
