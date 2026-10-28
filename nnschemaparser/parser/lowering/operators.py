"""
Lowering of the python operators of a forward method into instructions.

The binary, unary, comparison and boolean operations are lowered to calls of
their operator module function.
"""

# Import Modules
from typing import Any, Callable

import ast
import operator

from nnschemaparser.core.instructions import ControlInstruction, Instruction, OpInstruction, VarRef
from nnschemaparser.parser.lowering.base import LoweringContext
from nnschemaparser.parser.lowering.errors import UnsupportedSyntaxError
from nnschemaparser.parser.lowering.runtime_ops import (
    BINARY_OPERATORS,
    COMPARISON_OPERATORS,
    bool_and
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

        # A chained comparison is lowered to the conjunction of the
        # pairwise comparisons, with each operand evaluated only once
        if len(expr.ops) > 1:
            return self._lower_chained_compare(expr=expr, body=body)

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

    def _lower_chained_compare(self, expr: ast.Compare, body: list[Instruction]) -> VarRef | Any:
        """
        Lower a chained comparison to the conjunction of its pairs.

        Args:
            expr (ast.Compare): The comparison expression to lower.
            body (list[Instruction]): The body to append the instructions to.

        Returns:
            VarRef | Any: A variable reference to the result of the chain.
        """

        # Lower every operand of the chain exactly once
        operands: list[VarRef | Any] = [self._lower_expr(expr=expr.left, body=body)]
        comparator: ast.expr
        for comparator in expr.comparators:
            operands.append(self._lower_expr(expr=comparator, body=body))

        # Emit the pairwise comparisons
        pair_vars: list[str] = []
        pair_index: int
        op: ast.cmpop
        for pair_index, op in enumerate(expr.ops):
            left: VarRef | Any = operands[pair_index]
            right: VarRef | Any = operands[pair_index + 1]

            # The membership test has reversed arguments
            if type(op) is ast.In:
                fn: Callable[[Any, Any], Any] = operator.contains
                name: str = "contains"
                args: list[VarRef | Any] = [right, left]
            else:
                fn = COMPARISON_OPERATORS[type(op)]
                name = fn.__name__
                args = [left, right]

            # The comparison operation of the pair
            out_var: str = self._new_temp()
            self._emit_op(
                target=fn,
                qualified_name=name,
                args=args,
                kwargs={},
                out_var=out_var,
                body=body
            )
            pair_vars.append(out_var)

        # Chain the pairwise results with the boolean and
        current: VarRef = VarRef(name=pair_vars[0])
        pair_var: str
        for pair_var in pair_vars[1:]:
            out_var = self._new_temp()
            self._emit_op(
                target=bool_and,
                qualified_name="and",
                args=[current, VarRef(name=pair_var)],
                kwargs={},
                out_var=out_var,
                body=body
            )
            current = VarRef(name=out_var)

        # Return the reference to the result of the chain
        return current

    def _lower_bool_op(self, expr: ast.BoolOp, body: list[Instruction]) -> VarRef | Any:
        """
        Lower a boolean operation to a short-circuit nested condition chain.

        The python and/or operators evaluate the operands lazily, so the
        chain is lowered as nested if instructions binding the result,
        instead of an eager evaluation of every operand.

        Args:
            expr (ast.BoolOp): The boolean operation expression.
            body (list[Instruction]): The body to append the instructions to.

        Returns:
            VarRef | Any: A variable reference to the result of the operation.
        """

        # The output variable of the whole chain
        out_var: str = self._new_temp()
        body.extend(
            self._bool_chain(
                values=expr.values,
                index=0,
                out_var=out_var,
                is_and=isinstance(expr.op, ast.And)
            )
        )

        # Return the reference to the result of the operation
        return VarRef(name=out_var)

    def _bool_chain(
        self,
        values: list[ast.expr],
        index: int,
        out_var: str,
        is_and: bool
    ) -> list[Instruction]:
        """
        Build the nested instructions of a short-circuit boolean chain.

        Args:
            values (list[ast.expr]): The operand expressions of the chain.
            index (int): The index of the current operand in the chain.
            out_var (str): The name of the variable bound to the result.
            is_and (bool): Whether the chain is a conjunction.

        Returns:
            list[Instruction]: The instructions of the current chain level.
        """

        # The body of the current chain level
        body: list[Instruction] = []

        # The current operand is evaluated then used as the condition
        condition: OpInstruction = self._lower_condition(
            test=values[index], body=body
        )
        current_var: str = condition.out_var

        # The last operand is the result of the chain
        if index == len(values) - 1:
            self._bind_result(
                result=VarRef(name=current_var),
                out_var=out_var,
                body=body
            )
            return body

        # The continuation evaluates the next operands of the chain
        inner_body: list[Instruction] = self._bool_chain(
            values=values,
            index=index + 1,
            out_var=out_var,
            is_and=is_and
        )

        # The short-circuit branch binds the current operand as the result
        short_body: list[Instruction] = []
        self._bind_result(
            result=VarRef(name=current_var),
            out_var=out_var,
            body=short_body
        )

        # A falsy operand short-circuits an and, a truthy an or
        if is_and:
            body.append(ControlInstruction(
                kind="if",
                condition=condition,
                body=inner_body,
                else_body=short_body,
                bind_var=None
            ))
        else:
            body.append(ControlInstruction(
                kind="if",
                condition=condition,
                body=short_body,
                else_body=inner_body,
                bind_var=None
            ))
        return body
