from feathernet.compiler.ir_base import ModelIR
from feathernet.dl.utils import prune_weights


def prune_model(model_ir: ModelIR, threshold: float = 0.1):
    for layer in model_ir.nodes:
        if layer.layer_type == "Dense":
            layer.params["weights"] = prune_weights(
                layer.params["weights"], threshold
            )
