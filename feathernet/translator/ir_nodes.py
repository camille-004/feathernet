class IROperation:
    pass


class IRVariable(IROperation):
    def __init__(self, name: str) -> None:
        self.name = name


class IRLiteral(IROperation):
    def __init__(self, value: float) -> None:
        self.value = value


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
