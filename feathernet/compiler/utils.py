from feathernet.compiler.ir import IRNode, ModelIR

ACTIVATION_TYPES: set[str] = {"ReLU", "Sigmoid", "Softmax"}
LAYER_TYPES: set[str] = {"Conv2D", "Dense"}


def can_fuse(node1: IRNode, node2: IRNode) -> bool:
    """Check if two given nodes can be fused based on predefined rules."""
    if node1.layer_type == "Conv2D" and node2.layer_type == "BatchNorm":
        return True
    if (
        node1["type"].layer_type in LAYER_TYPES
    ) and node2.layer_type in ACTIVATION_TYPES:
        return True
    return False


def update_edge(ir: ModelIR, fused_node_idx: int) -> None:
    for edge in ir.edges:
        if edge["from"] == fused_node_idx + 1:
            edge["from"] = fused_node_idx
        if edge["to"] == fused_node_idx + 1:
            edge["to"] = fused_node_idx

    ir.edges = [edge for edge in ir.edges if edge["from"] != edge["to"]]
