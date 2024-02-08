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
