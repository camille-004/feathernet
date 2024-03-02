import numpy as np
import pycuda.autoinit  # noqa
import pycuda.gpuarray as gpuarray
from numpy.typing import ArrayLike

from feathernet.common.types import DeviceType, ShapeType, TensorSupported
from feathernet.gpu.gpu_ops import gpu_reshape
from feathernet.ir.core import Node
from feathernet.ir.ops import AddNode, SubNode
from feathernet.tensor_utils import (
    broadcast_shapes,
    perform_matmul,
    prepare_tensors,
)


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
        self._data = None
        self._shape = None
        self._dtype = None

        if isinstance(data, Node):
            self._ir_node = data
        else:
            if not isinstance(data, (np.ndarray, gpuarray.GPUArray)):
                data = np.array(data, dtype=np.float32)
            if device == "cpu":
                self._data = data
            else:
                if isinstance(data, gpuarray.GPUArray):
                    self._data = data
                else:
                    self._data = gpuarray.to_gpu(data)

            self._shape = self._data.shape
            self._dtype = self._data.dtype

    @property
    def data(self) -> np.ndarray:
        if self._data is not None:
            return self._data
        if self._ir_node:
            self.evaluate()  # Set _data.
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
            f"device={self.device}, data={self.data})\n"
        )

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
        # Calculate new shape with '-1' replaced by the actual dimension.
        inferred_shape = tuple(
            np.prod(self.shape) // np.prod(new_shape) if dim == -1 else dim
            for dim in new_shape
        )

        if self.shape == inferred_shape:
            return self  # No reshaping needed.

        total_elements = np.prod(self.shape)
        if total_elements != np.prod(new_shape):
            raise ValueError(
                f"Cannot reshape tensor or shape {self.shape} to {new_shape}."
            )

        if self.device == "gpu" and not isinstance(
            self._data, gpuarray.GPUArray
        ):
            t_gpu = self.to("gpu")
        else:
            t_gpu = self

        if self.device == "cpu":
            reshaped = np.reshape(self._data, new_shape)
            return Tensor(reshaped, device="cpu")
        elif self.device == "gpu":
            reshaped = gpu_reshape(t_gpu._data, new_shape)
            return Tensor(reshaped, device="gpu")
        else:
            raise ValueError(f"Unsupported device: {self.device}")

    def to(self, device: DeviceType) -> "Tensor":
        if device == self.device:
            return self

        if device == "gpu":
            if isinstance(self._data, np.ndarray) or np.isscalar(self._data):
                data = (
                    np.array([self._data])
                    if np.isscalar(self._data)
                    else self._data
                )
                gpu_data = gpuarray.to_gpu(data)
                return Tensor(gpu_data, device="gpu")
            elif isinstance(self._data, gpuarray.GPUArray):
                return self
            else:
                raise ValueError(
                    "Data must be a NumPy array to transfer to GPU."
                )
        elif device == "cpu":
            if isinstance(self._data, gpuarray.GPUArray):
                cpu_data = self._data.get()
                return Tensor(cpu_data, device="cpu")
            elif isinstance(self._data, np.ndarray) or np.isscalar(self._data):
                return self
            else:
                raise ValueError(
                    "Data must be a GPUArray array to transfer to CPU."
                )
        else:
            raise ValueError(f"Unsupported device: {device}")

    def evaluate(self) -> "Tensor":
        from feathernet.executor import GraphExecutor

        if self._ir_node is not None:
            executor = GraphExecutor()
            result_data = executor.evaluate(self._ir_node)
            if result_data.shape == ():
                self._data = result_data
                self._shape = ()
                self._dtype = type(result_data)
            else:
                if result_data.shape[-1] == 1 and len(result_data.shape) == 2:
                    result_data = result_data.reshape((result_data.shape[0],))

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
                result_tensor, other
            )

            # Set the correct attributes for the output.
            result_tensor = Tensor(data=node, device=device)
            result_tensor._shape = result_shape
            result_tensor._dtype = result_dtype

            # If the result is a vector, reshape it from a Nx1 matrix.
            if result_shape == (result_shape[0], 1):
                result_tensor = result_tensor.reshape((result_shape[0],))

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
        if np.isscalar(other.data):  # Account for scalar operands.
            other_shape = broadcast_shapes(self.shape, other.shape)
            other_data = np.full(other_shape, other.data, dtype=self.dtype)
            other = Tensor(other_data, device=self.device)

        op_node = op_node_class([self, other])
        return Tensor(op_node, device=self.device).evaluate()

    def __add__(self, other: "Tensor") -> "Tensor":
        return self._apply_bin_op(other, AddNode)

    def __sub__(self, other: "Tensor") -> "Tensor":
        return self._apply_bin_op(other, SubNode)
