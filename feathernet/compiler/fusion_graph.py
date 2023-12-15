from feathernet.compiler.fusion import fuse_layers
from feathernet.compiler.ir import ModelIR
from feathernet.compiler.utils import can_fuse, update_edge


def fuse_layers_in_model(model_ir: ModelIR) -> None:
    i = 0
    while i < len(model_ir.nodes) - 1:
        curr_node = model_ir.nodes[i]
        next_node = model_ir.nodes[i + 1]

        if can_fuse(curr_node, next_node):
            fused_node = fuse_layers(curr_node, next_node)
            model_ir.nodes[i] = fused_node
            del model_ir.nodes[i + 1]
            update_edge(model_ir, i)
        else:
            i += 1
