import numpy as np


class IROperation:
    pass


class IRVariable(IROperation):
    def __init__(self, name: str, shape: tuple = None) -> None:
        self.name = name
        self.shape = shape


class IRLiteral(IROperation):
    def __init__(self, value: float, shape: tuple = None) -> None:
        self.value = value
        self.shape = value.shape if isinstance(value, np.ndarray) else shape


class IRAssignment(IROperation):
    def __init__(self, target: IRVariable, value: IRVariable) -> None:
        self.target = target
        self.value = value


class BinaryOperation(IROperation):
    def __init__(self, left: IROperation, right: IROperation) -> None:
        self.left = left
        self.right = right


class AddOperation(BinaryOperation):
    pass


class SubtractOperation(BinaryOperation):
    pass


class MultiplyOperation(BinaryOperation):
    pass


class DivideOperation(BinaryOperation):
    pass
