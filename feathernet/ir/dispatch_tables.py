from typing import Callable

from feathernet.gpu.gpu_ops import gpu_add
from feathernet.ir.core import Node
from feathernet.ir.cpu_ops import cpu_add
from feathernet.ir.ops import AddNode

CPU_OPERATIONS: dict[Node, Callable] = {AddNode: cpu_add}

GPU_OPERATIONS: dict[Node, Callable] = {AddNode: gpu_add}
