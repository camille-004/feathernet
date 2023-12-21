from importlib import resources
from pathlib import Path
from typing import Callable

import numpy as np

from feathernet.compiler.ir_base import IRNode


class CodeGen:
    def __init__(self, ir_node: IRNode) -> None:
        self.ir_node = ir_node

        with resources.path(
            __package__ + ".templates", f"{ir_node.layer_type.lower()}.cpp"
        ) as template_path:
            self.template_path = str(template_path)

    def generate(self) -> str:
        layer_method: dict[str, Callable] = {
            "Dense": self._dense,
        }

        gen = layer_method.get(self.ir_node.layer_type)
        if not gen:
            raise ValueError(
                f"Unsupported layer type: {self.ir_node.layer_type}."
            )

        replacements = gen()

        with open(self.template_path, "r") as f:
            template = f.read()

        for var, val in replacements.items():
            template = template.replace(var, str(val))

        return template

    def _dense(self) -> dict[str, str]:
        weights = self.ir_node.params.get("weights", np.array([]))
        biases = self.ir_node.params.get("bias", np.array([]))
        input_dim = self.ir_node.params.get("input_dim")
        output_dim = self.ir_node.params.get("output_dim")

        weights_str = self._np_to_cpp_array(weights)
        biases_str = self._np_to_cpp_array(biases)

        replacements: dict[str, str] = {
            "@WEIGHTS @": weights_str,
            "@BIASES @": biases_str,
            "@INPUT_DIM @": input_dim,
            "@OUTPUT_DIM @": output_dim,
        }

        return replacements

    def _np_to_cpp_array(self, arr: np.ndarray) -> str:
        array_str = ", ".join(map(str, arr.flatten()))
        return f"{{ {array_str} }}"


def generate(ir_node: IRNode):
    layer_type = ir_node.layer_type
    template_path = Path(f"{layer_type.lower()}.cpp")
    codegen = CodeGen(ir_node, template_path)
    return codegen.generate()
