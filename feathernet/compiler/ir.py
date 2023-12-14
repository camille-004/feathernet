from typing import Any

from feathernet.dl.network import Network


class ModelIR:
    """Class to represent the IR.

    This will hold information about layers, their types, parameters, and
    connections.
    """

    def __init__(self) -> None:
        self.layers = []
        self.optimizer = None

    def add_layer(self, layer_type: str, **params: Any) -> None:
        layer = {"type": layer_type, "params": params}
        self.layers.append(layer)

    def serialize(self) -> list:
        return {"layers": self.layers, "optimizer": self.optimizer}


def create_ir_from_model(model: Network) -> ModelIR:
    """Convert a model into the IR."""
    ir = ModelIR()
    network_data = model.serialize()

    for layer_data in network_data["layers"]:
        ir.add_layer(layer_data["type"], **layer_data)

    ir.optimizer = network_data["optimizer"]
    return ir
