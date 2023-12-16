from feathernet.compiler.graph_opt import (
    Fusion,
    GraphOptimizer,
    Pruning,
    Quantization,
)
from feathernet.compiler.ir_base import IRNode, ModelIR
from feathernet.dl.network import Network

__all__ = ["IRNode", "ModelIR", "create_ir_from_model", "convert_model_to_ir"]

OPTIMIZERS: dict[str, type[GraphOptimizer]] = {
    "fusion": Fusion,
    "pruning": Pruning,
    "quantization": Quantization,
}


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
    model: Network, optimizations: [str, [dict[dict, str]]]
) -> ModelIR:
    model_ir = create_ir_from_model(model)

    for optimizer_name, params in optimizations.items():
        optimizer_class = OPTIMIZERS.get(optimizer_name.lower())
        if optimizer_class:
            optimizer = optimizer_class(**params)
            optimizer.optimize(model_ir)

    return model_ir
