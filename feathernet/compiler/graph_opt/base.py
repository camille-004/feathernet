from abc import ABC, abstractmethod

from feathernet.compiler.ir_base import ModelIR


class GraphOptimizer(ABC):
    @abstractmethod
    def optimize(self, model_ir: ModelIR) -> None:
        pass
