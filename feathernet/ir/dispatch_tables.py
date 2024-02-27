from typing import Callable

from feathernet.gpu.gpu_ops import gpu_add, gpu_matmul, gpu_sub
from feathernet.ir.core import Node
from feathernet.ir.cpu_ops import cpu_add, cpu_matmul, cpu_sub
from feathernet.ir.ops import AddNode, MatMulNode, SubNode

CPU_OPERATIONS: dict[Node, Callable] = {
    AddNode: cpu_add,
    SubNode: cpu_sub,
    MatMulNode: cpu_matmul,
}

GPU_OPERATIONS: dict[Node, Callable] = {
    AddNode: gpu_add,
    SubNode: gpu_sub,
    MatMulNode: gpu_matmul,
}
