import ast

from feathernet.translator.handlers import ASTHandler
from feathernet.translator.translators import IRTranslator


def translate(python_source: str) -> str:
    ast_tree = ast.parse(python_source)
    ir_tree = [ASTHandler.handle_node(node) for node in ast_tree.body]
    cuda = "\n".join(
        [
            IRTranslator.translate_node(ir_node)
            for ir_node in ir_tree
            if ir_node is not None
        ]
    )
    return cuda
