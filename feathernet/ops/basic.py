from feathernet.translator.ir_nodes import (
    AddOperation,
    DivideOperation,
    IROperation,
    MultiplyOperation,
    SubtractOperation,
)


def add(left: IROperation, right: IROperation) -> AddOperation:
    return AddOperation(left, right)


def sub(left: IROperation, right: IROperation) -> SubtractOperation:
    return SubtractOperation(left, right)


def mult(left: IROperation, right: IROperation) -> MultiplyOperation:
    return MultiplyOperation(left, right)


def div(left: IROperation, right: IROperation) -> DivideOperation:
    return DivideOperation(left, right)
