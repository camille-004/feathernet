import numpy as np

from feathernet.compiler.graph_opt import GraphOptimizer
from feathernet.compiler.ir_base import ModelIR
from feathernet.dl.utils import quantize_weights


class Quantization(GraphOptimizer):
    def __init__(self, precision: type = np.int8) -> None:
        self.precision = precision

    def optimize(self, model_ir: ModelIR) -> None:
        for layer in model_ir.nodes:
            if "weights" in layer.params:
                layer.params["weights"] = quantize_weights(
                    layer.params["weights"], self.precision
                )
