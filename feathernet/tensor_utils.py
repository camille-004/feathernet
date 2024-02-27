import numpy as np

from feathernet.common.enums import OpType
from feathernet.common.types import DeviceType, ShapeType
from feathernet.ir.ops import MatMulNode


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
    self_shape: ShapeType | None,
    other_shape: ShapeType | None,
) -> tuple[MatMulNode, DeviceType, ShapeType, np.dtype]:
    if len(a.shape) > 2 or len(b.shape) > 2:
        self_flattened_shape = (-1,) + a.shape[-2:]
        other_flattened_shape = (-1,) + b.shape[-2:]
        self_flattened = a.reshape(self_flattened_shape)
        other_flattened = b.reshape(other_flattened_shape)
        matmul_node = MatMulNode([self_flattened, other_flattened])
    else:
        matmul_node = MatMulNode([a, b])

    result_shape = (
        (a.shape[0], b.shape[1])
        if len(a.shape) > 1 and len(b.shape) > 1
        else None
    )
    result_dtype = a.dtype if self_shape and other_shape else None

    return matmul_node, a.device, result_shape, result_dtype


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
