from typing import Callable

from feathernet.gpu.gpu_ops import gpu_add, gpu_matmul
from feathernet.ir.core import Node
from feathernet.ir.cpu_ops import cpu_add, cpu_matmul
from feathernet.ir.ops import AddNode, MatMulNode

CPU_OPERATIONS: dict[Node, Callable] = {
    AddNode: cpu_add,
    MatMulNode: cpu_matmul,
}

GPU_OPERATIONS: dict[Node, Callable] = {
    AddNode: gpu_add,
    MatMulNode: gpu_matmul,
}
