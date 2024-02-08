from feathernet.common.enums import OpType


class Node:
    def __init__(
        self, operands: list["Node"] = [], op_type: OpType = OpType.ADD
    ) -> None:
        self.operands = operands
        self.op_type = op_type

    def __repr__(self) -> str:
        return (
            f"{self.op_type}({', '.join([str(op) for op in self.operands])})"
        )


class Graph:
    def __init__(self, final_node: Node | None = None) -> None:
        self.final_node = final_node  # Final node in the graph.

    def __repr__(self) -> str:
        return f"Graph(Output: {self.final_node})"
