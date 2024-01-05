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
            "AddOperation": "+",
            "SubtractOperation": "-",
            "MultiplyOperation": "*",
            "DivideOperation": "/",
        }
        operation_symbol = op_map[type(ir_node).__name__]
        left_code = IRTranslator.translate_node(ir_node.left)
        right_code = IRTranslator.translate_node(ir_node.right)
        return f"({left_code} {operation_symbol} {right_code})"

    @staticmethod
    def translate_node(ir_node: IROperation) -> str:
        if isinstance(ir_node, BinaryOperation):
            return IRTranslator.translate_binary_operation(ir_node)

        operation_name = type(ir_node).__name__
        translator = operation_registry.get_handler(operation_name)
        if translator:
            return translator(ir_node)
        else:
            raise NotImplementedError(
                f"No translator registered for {operation_name}."
            )


register_op("IRAddOperation")(IRTranslator.translate_binary_operation)
register_op("IRSubtractOperation")(IRTranslator.translate_binary_operation)
register_op("IRMultiplyOperation")(IRTranslator.translate_binary_operation)
register_op("IRDivideOperation")(IRTranslator.translate_binary_operation)
