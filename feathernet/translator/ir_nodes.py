class IROperation:
    pass


class IRVariable(IROperation):
    def __init__(
        self, name: str, shape: tuple = None, dtype: str = None
    ) -> None:
        self.name = name
        self.shape = shape
        self.dtype = dtype


class IRLiteral(IROperation):
    def __init__(
        self, value: float, shape: tuple = None, dtype: str = None
    ) -> None:
        self.value = value
        self.shape = shape
        self.dtype = dtype


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


class MatrixMultiplyOperation(BinaryOperation):
    pass


class MatrixTransposeOperation(IROperation):
    def __init__(self, operand: IROperation) -> None:
        self.operand = operand
