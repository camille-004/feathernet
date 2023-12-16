import numpy as np

from feathernet.compiler.ir import IRNode, ModelIR
from feathernet.compiler.utils import can_fuse, update_edge
from feathernet.dl.layers.convolution import Conv2D
from feathernet.dl.layers.core import BatchNorm


def fuse_layers(node1: IRNode, node2: IRNode) -> IRNode:
    if node1.layer_type == "Conv2D" and node2.layer_type == "BatchNorm":
        conv_layer = Conv2D(**node1.params)
        batch_norm_layer = BatchNorm(**node2.params)
        fused_layer = fuse_conv_batch_norm(conv_layer, batch_norm_layer)

    fused_layer_params = fused_layer.serialize()
    layer_type = fused_layer_params["type"]
    fused_layer_params.pop("type", None)
    return IRNode(layer_type, **fused_layer_params)


def fuse_conv_batch_norm(
    conv_layer: Conv2D, batch_norm_layer: BatchNorm
) -> Conv2D:
    input_dim = conv_layer.input_dim
    output_dim = conv_layer.output_dim
    kernel = conv_layer.kernel_size
    stride = conv_layer.stride
    padding = conv_layer.padding

    # Adjust weights and biases of the convolutional layer based on the BN
    # parameters.
    scale = 1 / np.sqrt(batch_norm_layer.variance + batch_norm_layer.epsilon)
    fused_weights = conv_layer.weights * scale.reshape((1, -1, 1, 1))
    fused_biases = (conv_layer.bias - batch_norm_layer.mean) * scale

    # Create new Conv2D layer with fused weights and biases.
    fused_conv = Conv2D(input_dim, output_dim, kernel, stride, padding)
    fused_conv.weights = fused_weights
    fused_conv.bias = fused_biases

    return fused_conv


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
