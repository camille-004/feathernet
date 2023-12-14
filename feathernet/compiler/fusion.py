import numpy as np

from feathernet.dl.layers.convolution import Conv2D
from feathernet.dl.layers.core import BatchNorm


def fuse_layers(node1, node2):
    if node1["type"] == "Conv2D" and node2["type"] == "BatchNorm":
        conv_layer = Conv2D(**node1["params"])
        batch_norm_layer = BatchNorm(**node2["params"])
        return fuse_conv_batchnorm(conv_layer, batch_norm_layer)
    return None


def fuse_conv_batchnorm(
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
