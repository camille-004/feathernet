import numpy as np
import pycuda.driver as cuda
from pycuda.gpuarray import GPUArray

from feathernet.gpu.kernel_manager import KernelManager

cuda.init()
device = cuda.Device(0)
print("Max Threads Per Block:", device.max_threads_per_block)
print(
    "Max Grid Dimensions:",
    device.max_grid_dim_x,
    device.max_grid_dim_y,
    device.max_grid_dim_z,
)
kernels = KernelManager()


def gpu_reshape(array: GPUArray, new_shape: tuple) -> GPUArray:
    total_elements = np.prod(array.shape)
    if total_elements != np.prod(new_shape):
        raise ValueError(
            "Total number of elements must remain constant during reshape."
        )

    mem_size = int(total_elements * array.dtype.itemsize)
    reshaped = cuda.mem_alloc(mem_size)

    reshape = kernels.compile("reshape")

    block_size = (128, 1, 1)
    grid_size_x = min(
        int((total_elements + block_size[0] - 1) // block_size[0]),
        device.MAX_GRID_DIM_X,
    )
    grid_size = (grid_size_x, 1)

    reshape(
        array,
        reshaped,
        np.int32(total_elements),
        block=block_size,
        grid=grid_size,
    )

    return GPUArray(new_shape, array.dtype, gpudata=reshaped)


def gpu_add(a: GPUArray, b: GPUArray) -> GPUArray:
    if a.shape != b.shape:
        raise ValueError(
            "Arrays must have the same shape for element-wise addition."
        )

    add = kernels.compile("add")
    result = cuda.mem_alloc(a.nbytes)

    block_size = (32, 1, 1)
    grid_size = ((a.size + block_size[0] - 1) // block_size[0], 1)
    add(a, b, result, np.int32(a.size), block=block_size, grid=grid_size)

    return GPUArray(a.shape, a.dtype, gpudata=result)


def gpu_sub(a: GPUArray, b: GPUArray) -> GPUArray:
    if a.shape != b.shape:
        raise ValueError(
            "Arrays must have the same shape for element-wise subtraction."
        )

    sub = kernels.compile("sub")
    result = cuda.mem_alloc(a.nbytes)

    block_size = (32, 1, 1)
    grid_size = ((a.size + block_size[0] - 1) // block_size[0], 1)
    sub(a, b, result, np.int32(a.size), block=block_size, grid=grid_size)

    return GPUArray(a.shape, a.dtype, gpudata=result)


def gpu_matmul(a: GPUArray, b: GPUArray) -> GPUArray:
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            "A's last dimension be the same as B's second-to-last dimension for matrix multiplication."
        )

    output_shape = a.shape[:-1] + b.shape[-2:]

    block_size = (16, 16, 1)
    grid_size = (
        (output_shape[-1] + block_size[0] - 1) // block_size[0],
        (output_shape[-2] + block_size[1] - 1) // block_size[1],
    )

    result = cuda.mem_alloc(np.prod(output_shape) * a.dtype.itemsize)
    matmul = kernels.compile("matmul")

    matmul(
        a,
        b,
        result,
        np.int32(a.shape[0]),
        np.int32(a.shape[1]),
        np.int32(b.shape[1]),
        block=block_size,
        grid=grid_size,
    )

    return GPUArray((a.shape[0], b.shape[1]), a.dtype, gpudata=result)
