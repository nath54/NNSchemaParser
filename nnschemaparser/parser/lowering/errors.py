"""
Errors raised by the lowering of a forward method into instructions.
"""

# Import Modules
import ast


class UnsupportedSyntaxError(Exception):
    """
    Error raised when the forward method uses syntax not supported by the lowering.
    """

    def __init__(self, message: str, node: ast.AST) -> None:
        """
        Init function.

        Args:
            message (str): The description of the unsupported syntax.
            node (ast.AST): The syntax tree node responsible for the error.
        """

        # Init parent with the message located at the node line
        super().__init__(f"{message} (line {getattr(node, 'lineno', '?')})")
