from typing import Any


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

    def __repr__(self) -> str:
        if not self._params:
            return f"IRNode(layer_type={self.layer_type})"

        params_str = ", ".join(
            f"{key}={value}" for key, value in self._params.items()
        )
        return f"IRNode(layer_type={self.layer_type}, {params_str})"


class ModelIR:
    """Class to represent the IR.

    This will hold information about layers, their types, parameters, and
    connections.
    """

    def __init__(self) -> None:
        self._nodes = []
        self._edges = []

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
        }

    def __repr__(self) -> str:
        nodes_repr = "\n\t".join([repr(node) for node in self._nodes])
        nodes_str = f"[\n\t{nodes_repr}\n  ]" if nodes_repr else "[]"

        edges_repr = "\n\t".join(
            [f"{edge['from']} -> {edge['to']}" for edge in self._edges]
        )
        edges_str = f"[\n\t{edges_repr}\n  ]" if edges_repr else "[]"

        return f"ModelIR(\n  Nodes: {nodes_str},\n  Edges: {edges_str},\n)"
