import ast

from feathernet.translator.ir_nodes import (
    AddOperation,
    BinaryOperation,
    DivideOperation,
    IRAssignment,
    IRLiteral,
    IROperation,
    IRVariable,
    MultiplyOperation,
    SubtractOperation,
    MatrixMultiplyOperation,
)
from feathernet.translator.registry import operation_registry, register_op


class ASTHandler:
    @staticmethod
    @register_op("Name")
    def handle_name(node: ast.Name) -> IRVariable:
        return IRVariable(node.id)

    @staticmethod
    @register_op("Num")
    def handle_num(node: ast.Num) -> IRLiteral:
        return IRLiteral(node.n)

    @staticmethod
    @register_op("Constant")
    def handle_constant(node: ast.Constant) -> IRLiteral:
        if isinstance(node.value, (int, float, str)):
            return IRLiteral(node.value)
        return NotImplementedError(
            f"Constants of type {type(node.value)} not implemented."
        )

    @staticmethod
    @register_op("Assign")
    def handle_assign(node: ast.Assign) -> IRAssignment:
        assert len(node.targets) == 1
        target = node.targets[0]
        if isinstance(target, ast.Name):
            ir_target = ASTHandler.handle_name(target)
            ir_value = ASTHandler.handle_node(node.value)
            return IRAssignment(ir_target, ir_value)
        raise NotImplementedError(
            "Assignment to non-variable not implemented."
        )

    @staticmethod
    @register_op("Add")
    def handle_add(node: ast.BinOp) -> AddOperation:
        left = ASTHandler.handle_node(node.left)
        right = ASTHandler.handle_node(node.right)
        return AddOperation(left, right)

    @staticmethod
    @register_op("Subtract")
    def handle_subtract(node: ast.BinOp) -> SubtractOperation:
        left = ASTHandler.handle_node(node.left)
        right = ASTHandler.handle_node(node.right)
        return SubtractOperation(left, right)

    @staticmethod
    @register_op("Multiply")
    def handle_multiply(node: ast.BinOp) -> MultiplyOperation:
        left = ASTHandler.handle_node(node.left)
        right = ASTHandler.handle_node(node.right)
        return MultiplyOperation(left, right)

    @staticmethod
    @register_op("Divide")
    def handle_divide(node: ast.BinOp) -> DivideOperation:
        left = ASTHandler.handle_node(node.left)
        right = ASTHandler.handle_node(node.right)
        return DivideOperation(left, right)

    @staticmethod
    @register_op("MatMult")
    def handle_matmult(node: ast.BinOp) -> MatrixMultiplyOperation:
        left = ASTHandler.handle_node(node.left)
        right = ASTHandler.handle_node(node.right)
        return MatrixMultiplyOperation(left, right)

    @staticmethod
    @register_op("BinOp")
    def handle_binop(node: ast.BinOp) -> BinaryOperation:
        if isinstance(node.op, ast.Add):
            return ASTHandler.handle_add(node)
        elif isinstance(node.op, ast.Sub):
            return ASTHandler.handle_subtract(node)
        elif isinstance(node.op, ast.Mult):
            return ASTHandler.handle_multiply(node)
        elif isinstance(node.op, ast.Div):
            return ASTHandler.handle_divide(node)
        elif isinstance(node.op, ast.MatMult):
            return ASTHandler.handle_matmult(node)
        else:
            raise NotImplementedError("Unsupported binary operation.")

    @staticmethod
    def handle_node(node: ast.AST) -> IROperation:
        operation_name = type(node).__name__
        handler = operation_registry.get_handler(operation_name)
        if handler:
            return handler(node)
        else:
            raise NotImplementedError(
                f"No handler registered for {operation_name}."
            )
