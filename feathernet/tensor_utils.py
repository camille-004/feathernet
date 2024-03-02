import numpy as np
from pycuda.gpuarray import GPUArray

from feathernet.common.enums import OpType
from feathernet.common.types import DeviceType, ShapeType
from feathernet.ir.ops import MatMulNode


def gpuarray_to_numpy(array: GPUArray) -> np.ndarray:
    if isinstance(array, GPUArray):
        return array.get()
    elif isinstance(array, np.ndarray):
        vfunc = np.vectorize(
            lambda x: gpuarray_to_numpy(x) if isinstance(x, GPUArray) else x
        )
        return vfunc(array)
    else:
        return array


def prepare_tensors(
    a: "Tensor", b: "Tensor", operation: OpType
) -> tuple[ShapeType | None, ShapeType | None]:
    self_shape, other_shape = None, None

    if operation == OpType.MATMUL:
        # Check for vectors.
        self_shape = a.shape if len(a.shape) > 1 else (1, a.shape[0])
        other_shape = b.shape if len(b.shape) > 1 else (b.shape[0], 1)

        if self_shape[-1] != other_shape[-2]:
            raise ValueError(
                f"Shape mismatch for matrix multiplication: {a.shape} and {b.shape}."
            )
    elif operation in [OpType.ADD, OpType.SUB]:
        if np.isscalar(a.data) or np.isscalar(b.data):
            self_shape, other_shape = broadcast_shapes(a.shape, b.shape)
        elif a.shape != b.shape:
            raise ValueError(
                f"Tensors must have the same shape for {operation}."
            )
        else:
            self_shape, other_shape = a.shape, b.shape

    return self_shape, other_shape


def perform_matmul(
    a: "Tensor",
    b: "Tensor",
) -> tuple[MatMulNode, DeviceType, ShapeType, np.dtype]:
    if len(a.shape) == 1:
        a = a.reshape((1,) + a.shape)  # Vector as 1xN matrix.
    if len(b.shape) == 1:
        b = b.reshape(b.shape + (1,))  # Vector as Nx1 matrix.

    if a.shape[-1] != b.shape[-2]:
        raise ValueError("Incompatible dimensions for matrix multiplication.")

    out_shape = a.shape[:-1] + b.shape[-1:]
    if out_shape[-1] == 1 and len(out_shape) > 1:  # Result is a vector.
        out_shape = (out_shape[0],)

    matmul_node = MatMulNode([a, b])
    result_dtype = np.result_type(a.dtype, b.dtype)

    return matmul_node, a.device, out_shape, result_dtype


def broadcast_shapes(shape1: ShapeType, shape2: ShapeType) -> ShapeType:
    if not shape1:
        return shape2
    elif not shape2:
        return shape1
    elif len(shape1) == len(shape2):
        return tuple(max(s1, s2) for s1, s2 in zip(shape1, shape2))
    elif len(shape1) < len(shape2):
        return shape2
    else:
        return shape1
