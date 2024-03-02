import numpy as np
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray

from feathernet.common.types import DeviceType, ShapeType
from feathernet.gpu.gpu_ops import gpu_reshape
from feathernet.ir.ops import MatMulNode

cuda.init()
context = cuda.Device(0).make_context()


def gpuarray_to_numpy(array: gpuarray.GPUArray) -> np.ndarray:
    if isinstance(array, gpuarray.GPUArray):
        return array.get()
    elif isinstance(array, np.ndarray):
        vfunc = np.vectorize(
            lambda x: (
                gpuarray_to_numpy(x) if isinstance(x, gpuarray.GPUArray) else x
            )
        )
        return vfunc(array)
    else:
        return array


def reshape(
    tensor: "Tensor", new_shape: tuple
) -> tuple[gpuarray.GPUArray | np.ndarray, DeviceType]:
    inferred_shape = tuple(
        np.prod(tensor.shape) // np.prod(new_shape) if dim == -1 else dim
        for dim in new_shape
    )
    if tensor.shape == inferred_shape:
        return tensor.data, tensor.device
    if np.prod(tensor.shape) != np.prod(new_shape):
        raise ValueError(
            f"Cannot reshape tensor of shape {tensor.shape} to {new_shape}."
        )
    if tensor.device == "cpu":
        reshaped_data = np.reshape(tensor.data, new_shape)
    elif tensor.device == "gpu":
        reshaped_data = gpu_reshape(tensor.data, new_shape)
    else:
        raise ValueError(f"Unsupported device: {tensor.device}.")
    return reshaped_data, tensor.device


def transfer_to_device(
    tensor: "Tensor", device: DeviceType
) -> tuple[gpuarray.GPUArray | np.ndarray, DeviceType]:
    if tensor.device == device:
        return tensor.data, device

    if device == "gpu":
        if isinstance(tensor.data, np.ndarray):
            contiguous_data = np.ascontiguousarray(
                tensor.data, dtype=tensor.data.dtype
            )
            gpu_data = gpuarray.to_gpu(contiguous_data)
            return gpu_data, "gpu"
    elif device == "cpu":
        if isinstance(tensor.data, gpuarray.GPUArray):
            cpu_data = tensor.data.get()
            return cpu_data, "cpu"
    else:
        raise ValueError(f"Unsupported device: {device}.")


def matmul_shapes(a: "Tensor", b: "Tensor") -> tuple["Tensor", "Tensor"]:
    if len(a.shape) == 1:
        a = a.reshape((1,) + a.shape)  # Vector as 1xN matrix.
    if len(b.shape) == 1:
        b = b.reshape(b.shape + (1,))  # Vector as Nx1 matrix.

    return a, b


def perform_matmul(
    a: "Tensor",
    b: "Tensor",
) -> tuple[MatMulNode, ShapeType, np.dtype]:
    a, b = matmul_shapes(a, b)

    if a.shape[-1] != b.shape[-2]:
        raise ValueError("Incompatible dimensions for matrix multiplication.")

    out_shape = a.shape[:-1] + b.shape[-1:]
    if out_shape[-1] == 1 and len(out_shape) > 1:  # Result is a vector.
        out_shape = (out_shape[0],)

    matmul_node = MatMulNode([a, b])
    result_dtype = np.result_type(a.dtype, b.dtype)

    return matmul_node, out_shape, result_dtype


context.pop()
