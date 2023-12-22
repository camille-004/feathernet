import re
from importlib import resources
from typing import Any

import numpy as np

from feathernet.compiler.ir_base import IRNode, ModelIR
from feathernet.compiler.codegen.utils import np_to_cpp_array


class LayerCodeGen:
    def __init__(self, ir_node: IRNode) -> None:
        self.ir_node = ir_node
        self._load_template()

    def _load_template(self) -> None:
        with resources.path(
            __package__ + ".templates",
            f"{self.ir_node.layer_type.lower()}.cpp",
        ) as template_path:
            self.template_path = str(template_path)

    def generate_layer_code(self, layer_name) -> str:
        with open(self.template_path, "r") as f:
            template = f.read()

        placeholders = re.findall(r"@([A-Z_]+) @", template)

        for placeholder in placeholders:
            param_name = placeholder.lower()
            value = self.ir_node.params.get(param_name)

            formatted_value = self._format_param(value)
            template = template.replace(f"@{placeholder} @", formatted_value)

        template = template.replace(
            "layerFunction", f"layerFunction_{layer_name}"
        )
        return template

    def _format_param(self, param: Any) -> str:
        if isinstance(param, np.ndarray):
            return np_to_cpp_array(param)
        return str(param)


class NetworkCodeGen:
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
        forward_declarations = self._generate_forward_declarations()

        layer_codes = [
            self._generate_layer_code(node, i)
            for i, node in enumerate(self.model_ir.nodes)
        ]

        network_code = (
            common_includes
            + "\n"
            + forward_declarations
            + "\n\n"
            + "\n\n".join(layer_codes)
            + "\n\n"
            + self._generate_main_function()
        )
        return network_code

    def _generate_common_includes(self) -> str:
        return "#include <cassert>\n#include <iostream>\n#include <vector>\n"

    def _generate_forward_declarations(self) -> str:
        declarations = [
            f"void layerFunction_layer{i}(const std::vector<float> &, std::vector <float> &, const std::vector<float> &, const std::vector<float> &);"
            for i in range(len(self.model_ir.nodes))
        ]
        return "\n".join(declarations)

    def _generate_layer_code(self, node: IRNode, layer_idx: int) -> str:
        codegen = LayerCodeGen(node)
        return codegen.generate_layer_code(f"layer{layer_idx}")

    def _generate_main_function(self) -> str:
        with open(self.main_template_path, "r") as f:
            main_template = f.read()

        network_initialization: list = []
        weights_biases_init: list = []
        network_execution: list = []
        final_output_handling: list = []

        for i, node in enumerate(self.model_ir.nodes):
            input_dim = node.params.get("input_dim")
            output_dim = node.params.get("output_dim")
            weights = node.params.get("weights")
            bias = node.params.get("bias")

            input_var = "input0" if i == 0 else f"output{i - 1}"
            output_var = f"output{i}"

            if i == 0:
                network_initialization.append(
                    f"std::vector<float> {input_var}({input_dim}, 0.0f);"
                )

            network_initialization.append(
                f"std::vector<float> {output_var}({output_dim}, 0.0f);"
            )
            weights_init = np_to_cpp_array(weights)
            bias_init = np_to_cpp_array(bias)

            weights_biases_init.append(
                f"std::vector<float> weights{i} = {weights_init};"
            )
            weights_biases_init.append(
                f"std::vector<float> bias{i} = {bias_init};"
            )

            network_execution.append(
                f"layerFunction_layer{i}({input_var}, {output_var}, weights{i}, bias{i});"
            )

        final_output_layer = f"output{len(self.model_ir.nodes) - 1}"
        final_output_handling.append(
            f"for (const auto& val : {final_output_layer}) std::cout << val << ' ';"
        )
        final_output_handling.append("std::cout << std::endl;")

        main_template = main_template.replace(
            "@NETWORK_INITIALIZATION @",
            "\n  ".join(network_initialization + weights_biases_init),
        )
        main_template = main_template.replace(
            "@NETWORK_EXECUTION @", "\n  ".join(network_execution)
        )
        main_template = main_template.replace(
            "@FINAL_OUTPUT_HANDLING @", "\n  ".join(final_output_handling)
        )

        return main_template