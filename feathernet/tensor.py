import numpy as np
import pycuda.autoinit  # noqa
import pycuda.gpuarray as gpuarray
from numpy.typing import ArrayLike

from feathernet.common.types import DeviceType, ShapeType, TensorSupported
from feathernet.ir.core import Node
from feathernet.ir.ops import AddNode, SubNode
from feathernet.tensor_utils import perform_matmul, reshape, transfer_to_device


def convert_to_tensor(other: TensorSupported, device: DeviceType) -> "Tensor":
    if isinstance(other, (int, float)):
        return Tensor(other, device=device)
    elif isinstance(other, np.ndarray):
        return Tensor(other, device=device)
    elif not isinstance(other, Tensor):
        raise ValueError("Operand must be a Tensor.")
    return other


class Tensor:
    def __init__(
        self,
        data: ArrayLike | Node | gpuarray.GPUArray,
        device: DeviceType = "cpu",
    ) -> None:
        self._device = device
        self._ir_node = None

        if isinstance(data, Node):
            self._ir_node = data
            self._data = None
            self._shape = None
            self._dtype = None
        else:
            self._data = (
                np.array(data, dtype=np.float32)
                if not isinstance(data, (np.ndarray, gpuarray.GPUArray))
                else data
            )
            self._shape = self._data.shape
            self._dtype = self._data.dtype
            if device == "gpu" and not isinstance(data, gpuarray.GPUArray):
                self._data = gpuarray.to_gpu(self._data)

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def shape(self) -> ShapeType:
        return self._shape

    @property
    def dtype(self) -> np.dtype | None:
        return self._dtype

    @property
    def device(self) -> DeviceType:
        return self._device

    def __repr__(self) -> str:
        dtype_name = getattr(self.dtype, "__name__", str(self.dtype))
        return (
            f"Tensor(shape={self.shape}, dtype={dtype_name}, "
            f"device={self.device}, data={self.data})\n"
        )

    @staticmethod
    def tensors_to_device(
        tensors: list["Tensor"], device: DeviceType
    ) -> list["Tensor"]:
        return [tensor.to(device) for tensor in tensors]

    def reshape(self, new_shape: tuple) -> "Tensor":
        reshaped_data, device = reshape(self, new_shape)
        return Tensor(reshaped_data, device=device)

    def to(self, device: DeviceType) -> "Tensor":
        new_data, device = transfer_to_device(self, device)
        return Tensor(new_data, device=device)

    def evaluate(self) -> "Tensor":
        if self._ir_node is not None:
            from feathernet.executor import GraphExecutor

            executor = GraphExecutor()
            self._data = executor.evaluate(self._ir_node)
            if self._data.ndim == 2 and self._data.shape[1] == 1:
                self._data = self._data.reshape(-1)

            self._shape = self._data.shape
            self._dtype = self._data.dtype
            return self
        else:
            raise ValueError(
                "This tensor does not have an associated computation graph node"
            )

    def matmul(self, *others: "Tensor") -> "Tensor":
        result_tensor = self
        for other in others:
            node, result_shape, result_dtype = perform_matmul(
                result_tensor, other
            )

            result_tensor = Tensor(data=node, device=self.device)
            result_tensor._shape = result_shape
            result_tensor._dtype = result_dtype

            # If the result is a vector, reshape it from a Nx1 matrix.
            if result_shape == (result_shape[0], 1):
                result_tensor = result_tensor.reshape((result_shape[0],))

        return result_tensor.evaluate()

    def __matmul__(self, other: "Tensor") -> "Tensor":
        return self.matmul(other)

    def _apply_bin_op(
        self, other: "Tensor", op_node_class: type[Node]
    ) -> "Tensor":
        other = convert_to_tensor(other, self.device)
        op_node = op_node_class([self, other])
        return Tensor(op_node, device=self.device).evaluate()

    def __add__(self, other: "Tensor") -> "Tensor":
        return self._apply_bin_op(other, AddNode)

    def __sub__(self, other: "Tensor") -> "Tensor":
        return self._apply_bin_op(other, SubNode)
