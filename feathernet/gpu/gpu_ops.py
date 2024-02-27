import numpy as np
import pycuda.driver as cuda
from pycuda.gpuarray import GPUArray

from feathernet.gpu.kernel_manager import KernelManager

kernels = KernelManager()


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


def gpu_matmul(a: GPUArray, b: GPUArray) -> GPUArray:
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            "A's columns must be the same as B's rows for matrix multiplication."
        )

    matmul = kernels.compile("matmul")
    result = cuda.mem_alloc(a.shape[0] * b.shape[1] * a.dtype.itemsize)

    block_size = (16, 16, 1)
    grid_size = (
        (b.shape[1] + block_size[0] - 1) // block_size[0],
        (a.shape[0] + block_size[1] - 1) // block_size[1],
    )
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
