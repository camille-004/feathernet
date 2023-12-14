from typing import Any

from feathernet.dl.network import Network


class ModelIR:
    """Class to represent the IR.

    This will hold information about layers, their types, parameters, and
    connections.
    """

    def __init__(self) -> None:
        self.nodes = []
        self.edges = []
        self.optimizer = None

    def add_node(self, layer_type: str, **params: Any) -> None:
        node = {"type": layer_type, "params": params}
        self.nodes.append(node)

    def add_edge(self, from_node: int, to_node: int) -> None:
        edge = {"from": from_node, "to": to_node}
        self.edges.append(edge)

    def serialize(self) -> list:
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "optimizer": self.optimizer,
        }


def create_ir_from_model(model: Network) -> ModelIR:
    """Convert a model into the IR."""
    ir = ModelIR()
    network_data = model.serialize()

    for i, layer_data in enumerate(network_data["layers"]):
        ir.add_node(layer_data["type"], **layer_data)
        if i > 0:
            ir.add_edge(i - 1, i)

    ir.optimizer = network_data["optimizer"]
    return ir
