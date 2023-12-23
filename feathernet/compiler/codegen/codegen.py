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

    def _load_training_template(self) -> str:
        with resources.path(
            __package__ + ".templates", "training.cpp"
        ) as template_path:
            training_template_path = str(template_path)

        with open(training_template_path, "r") as f:
            training_template = f.read()

        return training_template

    def _replace_placeholder(
        self, template: str, placeholder: str, content: str
    ) -> str:
        return template.replace(f"@{placeholder} @", content)

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

    def _initialize_vars(self) -> str:
        global_vars = "std::vector<float> input0;\n"
        for i, node in enumerate(self.model_ir.nodes):
            output_var = f"output{i}"
            grad_var = f"grad{i}"
            weights_grad_var = f"weights_grad{i}"
            bias_grad_var = f"bias_grad{i}"

            global_vars += f"std::vector<float> {output_var};\n"
            global_vars += f"std::vector<float> weights{i};\n"
            global_vars += f"std::vector<float> bias{i};\n"
            global_vars += f"std::vector<float> {grad_var};\n"
            global_vars += f"std::vector<float> {weights_grad_var};\n"
            global_vars += f"std::vector<float> {bias_grad_var};\n"

        global_vars += "std::vector<float> final_grad;\n"
        return global_vars

    def _initialize_network(self) -> str:
        init_func = ""
        for i, node in enumerate(self.model_ir.nodes):
            input_dim = node.params.get("input_dim")
            output_dim = node.params.get("output_dim")
            weights = node.params.get("weights")
            bias = node.params.get("bias")

            input_var = "input0" if i == 0 else f"output{i - 1}"
            output_var = f"output{i}"
            weights_grad_var = f"weights_grad{i}"
            bias_grad_var = f"bias_grad{i}"

            init_func += (
                f"  {input_var}.resize({input_dim}, 0.0f);\n"
                if i != 0
                else f"{input_var}.resize({input_dim}, 0.0f);\n"
            )
            init_func += f"  {output_var}.resize({output_dim}, 0.0f);\n"
            init_func += (
                f"  {weights_grad_var}.resize("
                f"{input_dim * output_dim}, 0.0f);\n"
            )
            init_func += f"  {bias_grad_var}.resize({output_dim}, 0.0f);\n"
            init_func += f"  weights{i} = {np_to_cpp_array(weights)};\n"
            init_func += f"  bias{i} = {np_to_cpp_array(bias)};\n"

        return init_func

    def _generate_network_initialization(self, training_template: str) -> str:
        global_vars = self._initialize_vars()
        init_func = self._initialize_network()
        training_template = self._replace_placeholder(
            training_template, "GLOBAL_VARS", global_vars
        )
        training_template = self._replace_placeholder(
            training_template, "NETWORK_INITIALIZATION", init_func.strip()
        )
        return training_template

    def _generate_layer_code(self, node: IRNode, layer_idx: int) -> str:
        codegen = LayerCodeGen(node)
        return codegen.generate_layer_code(f"layer{layer_idx}")

    def _generate_forward_pass(self) -> list:
        forward_pass = []
        for i in range(len(self.model_ir.nodes)):
            input_var = "input0" if i == 0 else f"output{i - 1}"
            output_var = f"output{i}"
            forward_call = (
                f"      layerFunction_layer{i}_Forward({input_var}, "
                f"{output_var}, weights{i}, bias{i});"
            )
            forward_pass.append(forward_call)

        return forward_pass

    def _generate_backward_pass(self) -> list:
        backward_pass = []
        for i in reversed(range(len(self.model_ir.nodes))):
            input_var = "input0" if i == 0 else f"output{i - 1}"
            grad_var = f"grad{i}"
            next_grad_var = (
                f"grad{i + 1}"
                if i < len(self.model_ir.nodes) - 1
                else "final_grad"
            )
            backward_call = (
                f"      layerFunction_layer{i}_Backward({input_var}, "
                f"{next_grad_var}, {grad_var}, weights_grad{i}, "
                f"bias_grad{i}, weights{i});"
            )
            backward_pass.append(backward_call)

        return backward_pass

    def _generate_training_loop(
        self,
        training_template: str,
        optimizer: Optimizer,
        training_params: dict[str, Any],
    ) -> str:
        training_template = self._replace_placeholder(
            training_template,
            "FORWARD_PASS",
            "\n".join(self._generate_forward_pass()),
        )
        training_template = self._replace_placeholder(
            training_template,
            "BACKWARD_PASS",
            "\n".join(self._generate_backward_pass()),
        )
        training_template = self._replace_placeholder(
            training_template,
            "NUM_EPOCHS",
            str(training_params.get("num_epochs", 1)),
        )
        training_template = self._replace_placeholder(
            training_template,
            "NUM_BATCHES",
            str(training_params.get("num_batches", 1)),
        )

        optimizer_updates = []
        for i in range(len(self.model_ir.nodes)):
            update_weights = (
                f"      optimizer.update(weights{i}, weights_grad{i});"
            )
            update_biases = f"      optimizer.update(bias{i}, bias_grad{i});"
            optimizer_updates.extend([update_weights, update_biases])
        training_template = self._replace_placeholder(
            training_template, "OPTIMIZER_UPDATE", "\n".join(optimizer_updates)
        )

        optimizer_data = optimizer.serialize()
        optimizer_init = (
            f"{optimizer_data['type']} optimizer("
            f"{optimizer_data['learning_rate']});"
        )
        training_template = self._replace_placeholder(
            training_template, "OPTIMIZER_INITIALIZATION", optimizer_init
        )

        return training_template

    def _generate_main_function(
        self,
        training_template: str,
        optimizer: Optimizer,
        training_params: dict[str, Any],
    ) -> str:
        training_template = self._generate_network_initialization(
            training_template
        )
        training_template = self._generate_training_loop(
            training_template, optimizer, training_params
        )

        final_output_layer = f"output{len(self.model_ir.nodes) - 1}"
        final_output_handling = [
            f"for (const auto& val : {final_output_layer}) std::cout << val << "
            f"' ';",
            "std::cout << std::endl;",
        ]

        training_template = training_template.replace(
            "@FINAL_OUTPUT_HANDLING @", "\n  ".join(final_output_handling)
        )
        return training_template

    def generate_network_code(
        self, optimizer: Optimizer, training_params: dict[str, Any]
    ) -> str:
        training_template = self._load_training_template()
        common_includes = self._generate_common_includes()
        layer_codes = [
            self._generate_layer_code(node, i)
            for i, node in enumerate(self.model_ir.nodes)
        ]
        optim_codegen = OptimizerCodeGen(optimizer)
        optimizer_code = optim_codegen.generate_optimizer_code()
        network_code = (
            common_includes
            + "\n"
            + optimizer_code
            + "\n"
            + self._generate_forward_backward_declarations()
            + "\n\n"
            + "\n\n".join(layer_codes)
            + "\n\n"
            + self._generate_main_function(
                training_template, optimizer, training_params
            )
        )
        return network_code
