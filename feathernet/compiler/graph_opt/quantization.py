import numpy as np

from feathernet.compiler.ir_base import ModelIR
from feathernet.dl.utils import quantize_weights


def quantize_model(model_ir: ModelIR, precision: type = np.int8):
    for layer in model_ir.nodes:
        if "weights" in layer.params:
            layer.params["weights"] = quantize_weights(
                layer.params["weights"], precision
            )
