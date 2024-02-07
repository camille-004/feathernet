from pathlib import Path

from feathernet.translator.ir_nodes import (
    BinaryOperation,
    IRAssignment,
    IRLiteral,
    IROperation,
    IRVariable,
)
from feathernet.translator.registry import operation_registry, register_op


class IRTranslator:
    @staticmethod
    def load_template(op_name: str) -> str:
        templates_dir = Path(__file__).parent.parent / "templates"
        template_path = templates_dir / "ops" / f"{op_name}.cu"
        return template_path.read_text()

    @staticmethod
    @register_op("IRVariable")
    def translate_variable(ir_node: IRVariable) -> str:
        return ir_node.name

    @staticmethod
    @register_op("IRLiteral")
    def translate_literal(ir_node: IRLiteral) -> str:
        return str(ir_node.value)

    @staticmethod
    @register_op("IRAssignment")
    def translate_assignment(ir_node: IRAssignment) -> str:
        target_code = IRTranslator.translate_node(ir_node.target)
        value_code = IRTranslator.translate_node(ir_node.value)
        return f"{target_code} = {value_code};"

    @staticmethod
    def translate_binary_operation(ir_node: BinaryOperation) -> str:
        op_map = {
            "AddOperation": "add",
            "SubtractOperation": "sub",
            "MultiplyOperation": "mult",
            "DivideOperation": "div",
        }
        op_name = op_map[type(ir_node).__name__]
        if not op_name:
            raise NotImplementedError(
                f"No template registered for {type(ir_node).__name__}."
            )
        template = IRTranslator.load_template(op_name)
        return template

    @staticmethod
    def translate_node(ir_node: IROperation) -> str:
        if isinstance(ir_node, BinaryOperation):
            source = IRTranslator.translate_binary_operation(ir_node)
        else:
            operation_name = type(ir_node).__name__
            translator = operation_registry.get_handler(operation_name)
            if translator:
                source = translator(ir_node)
            else:
                raise NotImplementedError(
                    f"No translator registered for {operation_name}."
                )

        return source


register_op("IRAddOperation")(IRTranslator.translate_binary_operation)
register_op("IRSubtractOperation")(IRTranslator.translate_binary_operation)
register_op("IRMultiplyOperation")(IRTranslator.translate_binary_operation)
register_op("IRDivideOperation")(IRTranslator.translate_binary_operation)
