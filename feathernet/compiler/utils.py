from typing import Any

from feathernet.compiler.ir import ModelIR

ACTIVATION_TYPES: set[str] = {"ReLU", "Sigmoid", "Softmax"}


def can_fuse(node1: dict[str, Any], node2: dict[str, Any]) -> bool:
    """Check if two given nodes can be fused based on predefined rules."""
    if node1["type"] == "Conv2D" and node2["type"] == "BatchNorm":
        return True
    if (node1["type"] in ["Conv2D", "Dense"]) and node2[
        "type"
    ] in ACTIVATION_TYPES:
        return True
    return False


def update_edge(ir: ModelIR, fused_node_idx: int) -> None:
    for edge in ir.edges:
        if edge["from"] == fused_node_idx + 1:
            edge["from"] = fused_node_idx
        if edge["to"] == fused_node_idx + 1:
            edge["to"] = fused_node_idx

    ir.edges = [edge for edge in ir.edges if edge["from"] != edge["to"]]
