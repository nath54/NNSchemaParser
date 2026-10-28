"""
Lowering of the side effect statements of a forward method.

The assert, expression, delete and raise statements are lowered to the
operation instructions producing their runtime effects.
"""

# Import Modules
from typing import Any

import ast

from src.core.instructions import Instruction, OpInstruction, VarRef
from src.model_parser.lowering.base import LoweringContext
from src.model_parser.lowering.errors import UnsupportedSyntaxError
from src.model_parser.lowering.naming import dotted_name, resolve_dotted_name
from src.model_parser.lowering.runtime_ops import check, raise_exception, raise_exception_instance


class SideEffectLowering(LoweringContext):
    """
    Lowers the side effect statements of a forward method into instructions.
    """

    def _lower_assert(self, stmt: ast.Assert, body: list[Instruction]) -> None:
        """
        Lower an assert statement to a checked condition.

        Args:
            stmt (ast.Assert): The assert statement to lower.
            body (list[Instruction]): The body to append the instructions to.
        """

        # Lower the tested condition of the assert into a temporary
        condition: OpInstruction = self._lower_condition(
            test=stmt.test,
            body=body
        )

        # Emit the checked condition of the assert
        self._emit_op(
            target=check,
            qualified_name="assert",
            args=[VarRef(name=condition.out_var)],
            kwargs={},
            out_var=self._new_temp(),
            body=body
        )

    def _lower_expr_stmt(self, stmt: ast.Expr, body: list[Instruction]) -> None:
        """
        Lower an expression statement, keeping only its side effects.

        Args:
            stmt (ast.Expr): The expression statement to lower.
            body (list[Instruction]): The body to append the instructions to.
        """

        # Only call expressions have side effects
        if not isinstance(stmt.value, ast.Call):
            raise UnsupportedSyntaxError("Unsupported expression statement", stmt)

        # An in-place method call on a variable keeps the variable binding
        out_var: str = self._new_temp()
        call_expr: ast.Call = stmt.value
        if (
            isinstance(call_expr.func, ast.Attribute)
            and isinstance(call_expr.func.value, ast.Name)
            and call_expr.func.attr.endswith("_")
        ):
            out_var = call_expr.func.value.id

        # Lower the call expression of the statement
        self._lower_call(call=call_expr, body=body, out_var=out_var)

    def _lower_delete(self, stmt: ast.Delete, body: list[Instruction]) -> None:
        """
        Lower a delete statement, which has no tensor effect.

        Args:
            stmt (ast.Delete): The delete statement to lower.
            body (list[Instruction]): The body to append the instructions to.
        """

        # Only the variable deletions are supported
        target: ast.expr
        for target in stmt.targets:
            if not isinstance(target, ast.Name):
                raise UnsupportedSyntaxError(
                    "Only variable deletions are supported", stmt
                )

    def _lower_raise(self, stmt: ast.Raise, body: list[Instruction]) -> None:
        """
        Lower a raise statement to a raising operation.

        Args:
            stmt (ast.Raise): The raise statement to lower.
            body (list[Instruction]): The body to append the instructions to.
        """

        # Only raises with an exception expression are supported
        if stmt.exc is None:
            raise UnsupportedSyntaxError("Bare raises are not supported", stmt)

        # A raise of an exception constructor call
        if isinstance(stmt.exc, ast.Call) and isinstance(
            stmt.exc.func, (ast.Name, ast.Attribute)
        ):
            self._lower_raise_call(stmt=stmt, body=body)
            return

        # A raise of an exception value
        exception_value: VarRef | Any = self._lower_expr(
            expr=stmt.exc,
            body=body
        )
        self._emit_op(
            target=raise_exception_instance,
            qualified_name="raise",
            args=[exception_value],
            kwargs={},
            out_var=self._new_temp(),
            body=body
        )

    def _lower_raise_call(self, stmt: ast.Raise, body: list[Instruction]) -> None:
        """
        Lower a raise of an exception constructor call.

        Args:
            stmt (ast.Raise): The raise statement to lower.
            body (list[Instruction]): The body to append the instructions to.
        """

        # Resolve the exception type from its dotted name
        dotted: str | None = dotted_name(expr=stmt.exc.func)
        if dotted is None:
            raise UnsupportedSyntaxError("Unsupported raised exception", stmt)
        exception_type: Any = resolve_dotted_name(
            dotted=dotted,
            node=stmt.exc.func,
            forward_globals=self.forward_globals
        )

        # Lower the arguments of the exception constructor
        raise_args: list[VarRef | Any] = [exception_type]
        arg: ast.expr
        for arg in stmt.exc.args:
            raise_args.append(self._lower_expr(expr=arg, body=body))

        # Emit the raising operation with the constructor arguments
        self._emit_op(
            target=raise_exception,
            qualified_name="raise",
            args=raise_args,
            kwargs={},
            out_var=self._new_temp(),
            body=body
        )
