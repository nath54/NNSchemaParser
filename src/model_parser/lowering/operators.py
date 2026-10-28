"""
Lowering of the python operators of a forward method into instructions.

The binary, unary, comparison and boolean operations are lowered to calls of
their operator module function.
"""

# Import Modules
from typing import Any, Callable

import ast
import operator

from src.core.instructions import Instruction, VarRef
from src.model_parser.lowering.base import LoweringContext
from src.model_parser.lowering.errors import UnsupportedSyntaxError
from src.model_parser.lowering.runtime_ops import (
    BINARY_OPERATORS,
    COMPARISON_OPERATORS,
    bool_and,
    bool_or
)


class OperatorLowering(LoweringContext):
    """
    Lowers the python operators of a forward method into instructions.
    """

    def _lower_bin_op(self, expr: ast.BinOp, body: list[Instruction]) -> VarRef | Any:
        """
        Lower a binary operation to its operator function.

        Args:
            expr (ast.BinOp): The binary operation expression.
            body (list[Instruction]): The body to append the instructions to.

        Returns:
            VarRef | Any: A variable reference to the result, or a literal.
        """

        # Only the mapped binary operators are supported
        op_type: type = type(expr.op)
        if op_type not in BINARY_OPERATORS:
            raise UnsupportedSyntaxError("Unsupported binary operator", expr)

        # Lower the two operands of the operation
        left: VarRef | Any = self._lower_expr(expr=expr.left, body=body)
        right: VarRef | Any = self._lower_expr(expr=expr.right, body=body)

        # Emit the operation with its operator function
        fn: Callable[[Any, Any], Any] = BINARY_OPERATORS[op_type]
        out_var: str = self._new_temp()
        self._emit_op(
            target=fn,
            qualified_name=fn.__name__,
            args=[left, right],
            kwargs={},
            out_var=out_var,
            body=body
        )

        # Return the reference to the result of the operation
        return VarRef(name=out_var)

    def _lower_unary_op(self, expr: ast.UnaryOp, body: list[Instruction]) -> VarRef | Any:
        """
        Lower a unary operation to its operator function.

        Args:
            expr (ast.UnaryOp): The unary operation expression.
            body (list[Instruction]): The body to append the instructions to.

        Returns:
            VarRef | Any: A variable reference to the result, or a literal.
        """

        # Map the unary operator to its function and name
        fn: Callable[[Any], Any]
        name: str
        if isinstance(expr.op, ast.USub):
            fn, name = operator.neg, "neg"
        elif isinstance(expr.op, ast.Not):
            fn, name = operator.not_, "not_"
        elif isinstance(expr.op, ast.Invert):
            fn, name = operator.invert, "invert"
        else:
            raise UnsupportedSyntaxError("Unsupported unary operator", expr)

        # Lower the operand of the operation
        operand: VarRef | Any = self._lower_expr(expr=expr.operand, body=body)

        # Emit the operation with its operator function
        out_var: str = self._new_temp()
        self._emit_op(
            target=fn,
            qualified_name=name,
            args=[operand],
            kwargs={},
            out_var=out_var,
            body=body
        )

        # Return the reference to the result of the operation
        return VarRef(name=out_var)

    def _lower_compare(self, expr: ast.Compare, body: list[Instruction]) -> VarRef | Any:
        """
        Lower a comparison to its operator function.

        Args:
            expr (ast.Compare): The comparison expression.
            body (list[Instruction]): The body to append the instructions to.

        Returns:
            VarRef | Any: A variable reference to the result, or a literal.
        """

        # Chained comparisons are not supported
        if len(expr.ops) != 1:
            raise UnsupportedSyntaxError("Chained comparisons are not supported", expr)

        # Lower the two sides of the comparison
        left: VarRef | Any = self._lower_expr(expr=expr.left, body=body)
        right: VarRef | Any = self._lower_expr(expr=expr.comparators[0], body=body)

        # The membership test has reversed arguments in the operator module
        op_type: type = type(expr.ops[0])
        args: list[VarRef | Any]
        if op_type is ast.In:
            fn: Callable[[Any, Any], Any] = operator.contains
            name: str = "contains"
            args = [right, left]

        # Else the comparison operator is mapped directly
        elif op_type in COMPARISON_OPERATORS:
            fn = COMPARISON_OPERATORS[op_type]
            name = fn.__name__
            args = [left, right]

        # Any other comparison operator is not supported
        else:
            raise UnsupportedSyntaxError("Unsupported comparison operator", expr)

        # Emit the comparison operation
        out_var: str = self._new_temp()
        self._emit_op(
            target=fn,
            qualified_name=name,
            args=args,
            kwargs={},
            out_var=out_var,
            body=body
        )

        # Return the reference to the result of the comparison
        return VarRef(name=out_var)

    def _lower_bool_op(self, expr: ast.BoolOp, body: list[Instruction]) -> VarRef | Any:
        """
        Lower a boolean operation to chained logical operations.

        Args:
            expr (ast.BoolOp): The boolean operation expression.
            body (list[Instruction]): The body to append the instructions to.

        Returns:
            VarRef | Any: A variable reference to the result of the operation.
        """

        # Map the boolean operator to its function and name
        fn: Callable[[Any, Any], Any]
        name: str
        if isinstance(expr.op, ast.And):
            fn, name = bool_and, "and"
        else:
            fn, name = bool_or, "or"

        # Lower all the values of the boolean operation
        values: list[VarRef | Any] = []
        value_expr: ast.expr
        for value_expr in expr.values:
            values.append(self._lower_expr(expr=value_expr, body=body))

        # Chain the values two by two with the boolean function
        current: VarRef | Any = values[0]
        value: VarRef | Any
        for value in values[1:]:
            out_var: str = self._new_temp()
            self._emit_op(
                target=fn,
                qualified_name=name,
                args=[current, value],
                kwargs={},
                out_var=out_var,
                body=body
            )
            current = VarRef(name=out_var)

        # Return the reference to the result of the operation
        return current
