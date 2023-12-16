from feathernet.compiler.graph_opt import GraphOptimizer
from feathernet.compiler.ir_base import ModelIR
from feathernet.dl.utils import prune_weights


class Pruning(GraphOptimizer):
    def __init__(self, threshold: float = 0.1) -> None:
        self.threshold = threshold

    def optimize(self, model_ir: ModelIR) -> None:
        for layer in model_ir.nodes:
            if layer.layer_type == "Dense":
                layer.params["weights"] = prune_weights(
                    layer.params["weights"], self.threshold
                )
