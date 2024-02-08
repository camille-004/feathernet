from pathlib import Path

import pycuda.autoinit  # noqa
import pycuda.driver as cuda
from pycuda.compiler import SourceModule


class KernelManager:
    def __init__(self) -> None:
        self.compiled_kernels: dict[str, cuda.Function] = {}

    def compile(self, kernel_name: str) -> cuda.Function:
        template_path = Path("kernels", f"{kernel_name}.cu")

        if kernel_name in self.compiled_kernels:
            return self.compiled_kernels[kernel_name]

        with open(template_path, "r") as f:
            kernel_template = f.read()

        mod = SourceModule(kernel_template)
        compiled_kernel = mod.get_function(kernel_name)
        self.compiled_kernels[kernel_name] = compiled_kernel

        return compiled_kernel
