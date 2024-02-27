import numpy as np
import pycuda.autoinit  # noqa
from numpy.typing import ArrayLike

from feathernet.common.types import DeviceType, ShapeType, TensorSupported
from feathernet.ir.core import Node
from feathernet.ir.ops import AddNode, SubNode
from feathernet.tensor_utils import (
    broadcast_shapes,
    perform_matmul,
    prepare_tensors,
)


def convert_to_tensor(
    other: np.ndarray | int | float, device: DeviceType
) -> "Tensor":
    if isinstance(other, (int, float)):
        return Tensor(other, device=device)
    elif isinstance(other, np.ndarray):
        return Tensor(other, device=device)
    elif not isinstance(other, Tensor):
        raise ValueError("Operand must be a Tensor.")
    return other


class Tensor:
    def __init__(
        self, data: ArrayLike | Node, device: DeviceType = "cpu"
    ) -> None:
        self._device = device
        self._ir_node = None
        self._data = None
        self._shape = None
        self._dtype = None
        if isinstance(data, Node):
            self._ir_node = data
        else:
            if np.isscalar(data):
                self._data = data
                self._shape = ()
                self._dtype = type(data)
            else:
                self._data = np.array(data)
                self._shape = self._data.shape
                self._dtype = self._data.dtype

    @property
    def data(self) -> np.ndarray:
        if self._data is not None:
            return self._data
        if self._ir_node:
            self.evaluate()
            return self._data
        return None

    @property
    def shape(self) -> ShapeType:
        if self._shape is not None:
            return self._shape
        elif self._ir_node and hasattr(self._ir_node, "shape"):
            return self._ir_node.shape
        else:
            return None

    @property
    def dtype(self) -> np.dtype | None:
        if self._dtype is not None:
            return self._dtype
        elif self._ir_node and hasattr(self._ir_node, "dtype"):
            return self._ir_node.dtype
        else:
            return None

    @property
    def device(self) -> DeviceType:
        return self._device

    def __repr__(self) -> str:
        dtype_name = getattr(self.dtype, "__name__", str(self.dtype))
        return (
            f"Tensor(shape={self.shape}, dtype={dtype_name}, "
            f"device={self.device})\nData: {self.data}\n"
        )  # noqa

    def _convert_and_prepare_other(
        self, other: TensorSupported, operation: str
    ) -> tuple["Tensor", "Tensor", ShapeType, ShapeType]:
        other = convert_to_tensor(other, device=self.device)
        return self._prepare(other, operation)

    def _prepare(
        self, other: "Tensor", operation: str
    ) -> tuple["Tensor", "Tensor", ShapeType, ShapeType]:
        if not isinstance(other, Tensor):
            raise ValueError(f"Operand must be a Tensor for {operation}.")
        if self.device != other.device:
            other = self.tensors_to_device([other], self.device)[0]

        self_shape, other_shape = prepare_tensors(self, other, operation)
        return self, other, self_shape, other_shape

    @staticmethod
    def tensors_to_device(
        tensors: list["Tensor"], device: DeviceType
    ) -> list["Tensor"]:
        return [tensor.to(device) for tensor in tensors]

    def reshape(self, new_shape: tuple) -> "Tensor":
        new_tensor = Tensor(self._data, self._device)
        new_tensor._shape = new_shape
        return new_tensor

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
        from feathernet.executor import GraphExecutor

        if self._ir_node is not None:
            executor = GraphExecutor()
            result_data = executor.evaluate(self._ir_node)
            if np.isscalar(result_data):
                self._data = result_data
                self._shape = ()
                self._dtype = type(result_data)
            else:
                self._data = result_data
                self._shape = result_data.shape
                self._dtype = result_data.dtype
            return self
        else:
            raise ValueError(
                "This tensor does not have an associated computation graph node"
            )

    def matmul(self, *others: "Tensor") -> "Tensor":
        if len(others) < 1:
            raise ValueError(
                "At least one operand is required for matrix multiplication."
            )

        result_tensor = self
        for other in others:
            _, other, self_shape, other_shape = result_tensor._prepare(
                other, "matmul"
            )
            node, device, result_shape, result_dtype = perform_matmul(
                result_tensor, other, self_shape, other_shape
            )
            result_tensor = Tensor(data=node, device=device)
            result_tensor._shape = result_shape
            result_tensor._dtype = result_dtype

        return result_tensor.evaluate()

    def __matmul__(self, other: "Tensor") -> "Tensor":
        other = convert_to_tensor(other, device=self.device)
        if not isinstance(other, Tensor):
            raise ValueError("Operand must be a Tensor.")
        return self.matmul(other)

    def _apply_bin_op(
        self, other: "Tensor", op_node_class: type[Node]
    ) -> "Tensor":
        _, other, _, _ = self._convert_and_prepare_other(
            other, op_node_class.__name__.lower()
        )
        if np.isscalar(other.data):
            other_shape = broadcast_shapes(self.shape, other.shape)
            other_data = np.full(other_shape, other.data, dtype=self.dtype)
            other = Tensor(other_data, device=self.device)

        op_node = op_node_class([self, other])
        return Tensor(op_node, device=self.device).evaluate()

    def __add__(self, other: "Tensor") -> "Tensor":
        return self._apply_bin_op(other, AddNode)

    def __sub__(self, other: "Tensor") -> "Tensor":
        return self._apply_bin_op(other, SubNode)
