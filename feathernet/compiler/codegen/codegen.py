import re
from importlib import resources
from typing import Any

import numpy as np

from feathernet.compiler.codegen.utils import np_to_cpp_array
from feathernet.compiler.ir_base import IRNode, ModelIR
from feathernet.dl.optimizers import Optimizer


class LayerCodeGen:
    """Generate code for individual layers."""

    def __init__(self, ir_node: IRNode) -> None:
        self.ir_node = ir_node
        self.template = self._load_template()

    def _load_template(self) -> str:
        with resources.path(
            __package__ + ".templates.layers",
            f"{self.ir_node.layer_type.lower()}.cpp",
        ) as template_path:
            template_path = str(template_path)

        with open(template_path, "r") as f:
            template = f.read()

        return template

    def generate_layer_code(self, layer_name) -> str:
        placeholders = re.findall(r"@([A-Z_]+) @", self.template)

        for placeholder in placeholders:
            param_name = placeholder.lower()
            value = self.ir_node.params.get(param_name)

            formatted_val = self._format_param(value)
            self.template = self.template.replace(
                f"@{placeholder} @", formatted_val
            )

        self.template = self.template.replace(
            "layerFunction", f"layerFunction_{layer_name}"
        )
        return self.template

    def _format_param(self, param: Any) -> str:
        if isinstance(param, np.ndarray):
            return np_to_cpp_array(param)
        return str(param)


class OptimizerCodeGen:
    """Generate code for the optimizer."""

    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer = optimizer
        self.template = self._load_template()

    def _load_template(self) -> str:
        with resources.path(
            __package__ + ".templates.optimizers",
            f"{self.optimizer.serialize()['type'].lower()}.cpp",
        ) as template_path:
            template_path = str(template_path)

        with open(template_path, "r") as f:
            template = f.read()

        return template

    def generate_optimizer_code(self) -> str:
        placeholders = re.findall(r"@([A-Z_]+) @", self.template)
        serialized_data = self.optimizer.serialize()

        for placeholder in placeholders:
            param_name = placeholder.lower()
            val = serialized_data.get(param_name)

            if val is None:
                raise ValueError(
                    f"Hyperparameter '{param_name}' not provided for optimizer"
                    f"'{serialized_data['type']}'."
                )

            formatted_val = str(val)
            self.template = self.template.replace(
                f"@{placeholder} @", formatted_val
            )

        return self.template


class NetworkCodeGen:
    """Generate code for forward and backward passes."""

    def __init__(self, model_ir: ModelIR) -> None:
        self.model_ir = model_ir
        self.main_template_path = self._load_main_template()

    def _load_main_template(self) -> str:
        with resources.path(
            __package__ + ".templates", "main.cpp"
        ) as template_path:
            return str(template_path)

    def generate_network_code(self) -> str:
        common_includes = self._generate_common_includes()
        forward_backward_declarations = (
            self._generate_forward_backward_declarations()
        )

        layer_codes = [
            self._generate_layer_code(node, i)
            for i, node in enumerate(self.model_ir.nodes)
        ]

        network_code = (
            common_includes
            + "\n"
            + forward_backward_declarations
            + "\n\n"
            + "\n\n".join(layer_codes)
            + "\n\n"
            + self._generate_main_function()
        )
        return network_code

    def _generate_common_includes(self) -> str:
        return "#include <cassert>\n#include <iostream>\n#include <vector>\n"

    def _generate_forward_backward_declarations(self) -> str:
        declarations = []
        for i in range(len(self.model_ir.nodes)):
            declarations.append(
                f"void layerFunction_layer{i}_Forward(const std::vector<float>"
                f"&, std::vector<float> &, const "
                f"std::vector<float> &, const std::vector<float> &);"
            )
            declarations.append(
                f"void layerFunction_layer{i}_Backward(const std::vector"
                f"<float> &, const std::vector<float> &, std::vector<float> &,"
                f" std::vector<float> &, std::vector<float> &, const "
                f"std::vector<float> &);"
            )
        return "\n".join(declarations)

    def _generate_layer_code(self, node: IRNode, layer_idx: int) -> str:
        codegen = LayerCodeGen(node)
        return codegen.generate_layer_code(f"layer{layer_idx}")

    def _generate_main_function(self) -> str:
        with open(self.main_template_path, "r") as f:
            main_template = f.read()

        network_initialization: list = []
        forward_pass: list = []
        backward_pass: list = []
        final_output_handling: list = []

        for i, node in enumerate(self.model_ir.nodes):
            input_dim = node.params.get("input_dim")
            output_dim = node.params.get("output_dim")
            weights = node.params.get("weights")
            bias = node.params.get("bias")

            input_var = "input0" if i == 0 else f"output{i - 1}"
            output_var = f"output{i}"
            grad_var = f"grad{i}"
            weights_grad_var = f"weights_grad{i}"
            bias_grad_var = f"bias_grad{i}"

            if i == 0:
                network_initialization.append(
                    f"std::vector<float> {input_var}({input_dim}, 0.0f);"
                )

            network_initialization.append(
                f"std::vector<float> {output_var}({output_dim}, 0.0f);"
            )

            weights_init = np_to_cpp_array(weights)
            bias_init = np_to_cpp_array(bias)
            network_initialization.append(
                f"std::vector<float> weights{i} = {weights_init};"
            )
            network_initialization.append(
                f"std::vector<float> bias{i} = {bias_init};"
            )

            network_initialization.append(
                f"std::vector<float> {grad_var}({output_dim}, 0.0f);"
            )
            network_initialization.append(
                f"std::vector<float> {weights_grad_var}("
                f"{input_dim * output_dim}, 0.0f);"
            )
            network_initialization.append(
                f"std::vector<float> {bias_grad_var}({output_dim}, 0.0f);"
            )

            forward_pass.append(
                f"layerFunction_layer{i}_Forward({input_var}, {output_var}, "
                f"weights{i}, bias{i});"
            )

            if i < len(self.model_ir.nodes) - 1:
                next_grad_var = f"grad{i + 1}"
                backward_pass.append(
                    f"layerFunction_layer{i}_Backward({input_var}, "
                    f"{next_grad_var}, {grad_var}, weights_grad{i}, "
                    f"bias_grad{i}, weights{i});"
                )

        backward_pass.reverse()

        final_output_layer = f"output{len(self.model_ir.nodes) - 1}"
        final_output_handling.append(
            f"for (const auto& val : {final_output_layer}) std::cout << val << "
            f"' ';"
        )
        final_output_handling.append("std::cout << std::endl;")

        main_template = main_template.replace(
            "@NETWORK_INITIALIZATION @",
            "\n  ".join(network_initialization),
        )
        main_template = main_template.replace(
            "@FORWARD_PASS @", "\n  ".join(forward_pass)
        )
        main_template = main_template.replace(
            "@BACKWARD_PASS @", "\n  ".join(backward_pass)
        )
        main_template = main_template.replace(
            "@FINAL_OUTPUT_HANDLING @", "\n  ".join(final_output_handling)
        )

        return main_template
