from typing import Any

from feathernet.dl.network import Network
from feathernet.dl.optimizers import Optimizer


class IRNode:
    def __init__(self, layer_type: str, **params: Any) -> None:
        self._layer_type = layer_type
        self._params = params

    @property
    def layer_type(self) -> str:
        return self._layer_type

    @layer_type.setter
    def layer_type(self, layer_type: str) -> None:
        self._layer_type = layer_type

    @property
    def params(self) -> dict[str, Any]:
        return self._params

    @params.setter
    def params(self, params: dict[str, Any]) -> None:
        self._params = params

    def serialize(self) -> dict[str, Any]:
        return {"type": self.layer_type, "params": self.params}


class ModelIR:
    """Class to represent the IR.

    This will hold information about layers, their types, parameters, and
    connections.
    """

    def __init__(self) -> None:
        self._nodes = []
        self._edges = []
        self._optimizer = None

    @property
    def nodes(self) -> list[IRNode]:
        return self._nodes

    @nodes.setter
    def nodes(self, nodes: list[IRNode]) -> None:
        self._nodes = nodes

    @property
    def edges(self) -> list[dict[str, IRNode]]:
        return self._edges

    @edges.setter
    def edges(self, edges: list[dict[str, IRNode]]):
        self._edges = edges

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: Optimizer) -> None:
        self._optimizer = optimizer

    def add_node(self, layer_type: str, **params: Any) -> None:
        node = IRNode(layer_type, **params)
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
