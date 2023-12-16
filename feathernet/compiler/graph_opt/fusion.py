import numpy as np

from feathernet.compiler.ir_base import IRNode, ModelIR
from feathernet.compiler.utils import can_fuse, update_edge
from feathernet.dl.layers.convolution import Conv2D
from feathernet.dl.layers.core import BatchNorm


def fuse_layers(node1: IRNode, node2: IRNode) -> IRNode:
    if node1.layer_type == "Conv2D" and node2.layer_type == "BatchNorm":
        return fuse_conv_batch_norm(node1, node2)
    if node1.layer_type == "Dense" and node2.layer_type == "Dense":
        return fuse_dense_layers(node1, node2)

    return None


def fuse_conv_batch_norm(conv_node: IRNode, bn_node: IRNode) -> IRNode:
    conv_params = conv_node.params
    bn_params = bn_node.params

    conv_layer = Conv2D(**conv_params)
    bn_layer = BatchNorm(**bn_params)

    # Adjust weights and biases of the convolutional layer based on the BN
    # parameters.
    scale = 1 / np.sqrt(bn_layer.variance + bn_layer.epsilon)
    fused_weights = conv_layer.weights * scale.reshape((1, -1, 1, 1))
    fused_biases = (conv_layer.bias - bn_layer.mean) * scale

    # Create new Conv2D layer with fused weights and biases.
    fused_conv = Conv2D(
        input_dim=conv_layer.input_dim,
        output_dim=conv_layer.output_dim,
        kernel_size=conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
    )
    fused_conv.weights = fused_weights
    fused_conv.bias = fused_biases

    fused_layer_params = fused_conv.serialize()
    layer_type = fused_layer_params["type"]
    fused_layer_params.pop("type", None)
    return IRNode(layer_type, **fused_layer_params)


def fuse_dense_layers(dense_node1: IRNode, dense_node2: IRNode) -> IRNode:
    fused_weights = np.dot(
        dense_node1.params["weights"], dense_node2.params["weights"]
    )
    fused_biases = dense_node2.params["bias"] + np.dot(
        dense_node1.params["bias"], dense_node2.params["weights"]
    )
    return IRNode(
        "Dense",
        weights=fused_weights,
        biases=fused_biases,
        input_dim=dense_node1.params["input_dim"],
        output_dim=dense_node2.params["output_dim"],
    )


def fuse_layers_in_model(model_ir: ModelIR) -> None:
    i = 0
    while i < len(model_ir.nodes) - 1:
        curr_node = model_ir.nodes[i]
        next_node = model_ir.nodes[i + 1]

        if can_fuse(curr_node, next_node):
            fused_node = fuse_layers(curr_node, next_node)
            if fused_node is not None:
                fused_ir_node = IRNode(
                    fused_node.layer_type, **fused_node.params
                )
                model_ir.nodes[i] = fused_ir_node
                del model_ir.nodes[i + 1]
                update_edge(model_ir, i)
        else:
            i += 1
