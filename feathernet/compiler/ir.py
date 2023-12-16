from feathernet.compiler.graph_opt import (
    fuse_layers_in_model,
    prune_model,
    quantize_model,
)
from feathernet.compiler.ir_base import IRNode, ModelIR
from feathernet.dl.network import Network

__all__ = ["IRNode", "ModelIR", "create_ir_from_model", "convert_model_to_ir"]


def create_ir_from_model(model: Network) -> ModelIR:
    """Convert a model into the IR."""
    ir = ModelIR()
    network_data = model.serialize()

    for i, layer_data in enumerate(network_data["layers"]):
        ir.add_node(layer_data["type"], **layer_data)
        if i > 0:
            ir.add_edge(i - 1, i)

    return ir


def convert_model_to_ir(
    model: Network, prune_threshold: float, quantize_precision: type
) -> ModelIR:
    model_ir = create_ir_from_model(model)
    fuse_layers_in_model(model_ir)
    prune_model(model_ir, prune_threshold)
    quantize_model(model_ir, quantize_precision)
    return model_ir
