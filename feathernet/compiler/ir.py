from typing import Any

from feathernet.dl.network.base import Network


class ModelIR:
    """Class to represent the IR.

    This will hold information about layers, their types, parameters, and
    connections.
    """

    def __init__(self) -> None:
        self.layers = []

    def add_layer(self, layer_type: str, **params: Any) -> None:
        layer = {"type": layer_type, "params": params}
        self.layers.append(layer)

    def serialize(self) -> list:
        return self.layers


def create_ir_from_model(model: Network) -> ModelIR:
    """Convert a model into the IR."""
    ir = ModelIR()
    for layer in model.layers:
        layer_data = layer.serialize()
        ir.add_layer(layer["type"], **layer_data)
    return ir
