"""
Lowering of the f-string expressions of a forward method into instructions.

The f-strings are lowered to formatted part operations joined into a string.
"""

# Import Modules
from typing import Any

import ast

from src.core.instructions import Instruction, VarRef
from src.model_parser.lowering.base import LoweringContext
from src.model_parser.lowering.runtime_ops import format_part, join_str


class FStringLowering(LoweringContext):
    """
    Lowers the f-string expressions of a forward method into instructions.
    """

    def _lower_joined_str(self, expr: ast.JoinedStr, body: list[Instruction]) -> VarRef | Any:
        """
        Lower an f-string to the join of its formatted parts.

        Args:
            expr (ast.JoinedStr): The f-string expression to lower.
            body (list[Instruction]): The body to append the instructions to.

        Returns:
            VarRef | Any: A variable reference to the built string.
        """

        # Lower all the parts of the f-string
        parts: list[VarRef | Any] = []
        part: ast.expr
        for part in expr.values:
            parts.append(self._lower_expr(expr=part, body=body))

        # Emit the join of the parts of the f-string
        out_var: str = self._new_temp()
        self._emit_op(
            target=join_str,
            qualified_name="fstring",
            args=parts,
            kwargs={},
            out_var=out_var,
            body=body
        )

        # Return the reference to the built string
        return VarRef(name=out_var)

    def _lower_formatted_value(
        self,
        expr: ast.FormattedValue,
        body: list[Instruction]
    ) -> VarRef | Any:
        """
        Lower a formatted value of an f-string to its formatted value.

        Args:
            expr (ast.FormattedValue): The formatted value to lower.
            body (list[Instruction]): The body to append the instructions to.

        Returns:
            VarRef | Any: A variable reference to the formatted value.
        """

        # Lower the value of the formatted value
        value: VarRef | Any = self._lower_expr(expr=expr.value, body=body)

        # Lower the format spec of the formatted value when defined
        format_spec: str | None = None
        if expr.format_spec is not None:
            spec_result: VarRef | Any = self._lower_expr(
                expr=expr.format_spec,
                body=body
            )
            if isinstance(spec_result, VarRef):
                format_spec = self._scope_placeholder(spec_result)
            else:
                format_spec = spec_result

        # Emit the formatting of the value of the f-string part
        out_var: str = self._new_temp()
        self._emit_op(
            target=format_part,
            qualified_name="format",
            args=[value],
            kwargs={"conversion": expr.conversion, "format_spec": format_spec},
            out_var=out_var,
            body=body
        )

        # Return the reference to the formatted value
        return VarRef(name=out_var)
