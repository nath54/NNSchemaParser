"""
Lowering of the expressions of a forward method into instructions.

Each expression is lowered into a variable reference of the block scope or a
literal value, emitting the intermediate operations into the body.
"""

# Import Modules
from typing import Any

import ast
import builtins
import operator

from src.core.instructions import ControlInstruction, Instruction, OpInstruction, VarRef
from src.model_parser.lowering.base import LoweringContext
from src.model_parser.lowering.errors import UnsupportedSyntaxError
from src.model_parser.lowering.naming import RESOLVE_NAMESPACE, resolve_dotted_name
from src.model_parser.lowering.runtime_ops import make_list, make_tuple


class ExpressionLowering(LoweringContext):
    """
    Lowers the expressions of a forward method into instructions.
    """

    def _lower_expr(self, expr: ast.expr, body: list[Instruction]) -> VarRef | Any:
        """
        Lower an expression, returning a variable reference or a literal value.

        Args:
            expr (ast.expr): The expression to lower.
            body (list[Instruction]): The body to append the instructions to.

        Returns:
            VarRef | Any: A variable reference to the result, or a literal.
        """

        # A variable name is a direct reference to the scope
        if isinstance(expr, ast.Name):

            # The self name is a reference to the source module at run time
            if expr.id == "self":
                return VarRef(name="$self")

            # The forward parameters are always references to the scope
            if expr.id in self._param_names:
                return VarRef(name=expr.id)

            # A global class or function is embedded as a constant value
            if (
                expr.id in self.forward_globals
                or expr.id in RESOLVE_NAMESPACE
                or hasattr(builtins, expr.id)
            ):
                return resolve_dotted_name(
                    dotted=expr.id,
                    node=expr,
                    forward_globals=self.forward_globals
                )

            # Else the name is a reference to the scope
            return VarRef(name=expr.id)

        # A constant is a literal value
        if isinstance(expr, ast.Constant):
            return expr.value

        # A call is lowered to an instruction bound to a temporary variable
        if isinstance(expr, ast.Call):
            out_var: str = self._new_temp()
            self._lower_call(call=expr, body=body, out_var=out_var)
            return VarRef(name=out_var)

        # An attribute access is handled depending on its object
        if isinstance(expr, ast.Attribute):
            return self._lower_attribute(expr=expr, body=body)

        # A subscript is lowered to a getitem operation
        if isinstance(expr, ast.Subscript):
            return self._lower_subscript(expr=expr, body=body)

        # A binary, unary, comparison or boolean operation
        if isinstance(expr, ast.BinOp):
            return self._lower_bin_op(expr=expr, body=body)
        if isinstance(expr, ast.UnaryOp):
            return self._lower_unary_op(expr=expr, body=body)
        if isinstance(expr, ast.Compare):
            return self._lower_compare(expr=expr, body=body)
        if isinstance(expr, ast.BoolOp):
            return self._lower_bool_op(expr=expr, body=body)

        # A tuple or list display is lowered to its constructor
        if isinstance(expr, (ast.Tuple, ast.List)):
            return self._lower_sequence(expr=expr, body=body)

        # A conditional expression is lowered to an if instruction binding
        if isinstance(expr, ast.IfExp):
            return self._lower_ifexp(expr=expr, body=body)

        # An f-string is lowered to its formatted parts joined together
        if isinstance(expr, ast.JoinedStr):
            return self._lower_joined_str(expr=expr, body=body)

        # A formatted value is lowered to its formatted value
        if isinstance(expr, ast.FormattedValue):
            return self._lower_formatted_value(expr=expr, body=body)

        # Any other expression is not supported
        raise UnsupportedSyntaxError(
            f"Unsupported expression: {type(expr).__name__}", expr
        )

    def _lower_attribute(self, expr: ast.Attribute, body: list[Instruction]) -> VarRef | Any:
        """
        Lower an attribute access expression.

        Args:
            expr (ast.Attribute): The attribute access expression.
            body (list[Instruction]): The body to append the instructions to.

        Returns:
            VarRef | Any: A variable reference to the result, or a literal.
        """

        # An attribute of self is resolved in the symbol table of the module
        if isinstance(expr.value, ast.Name) and expr.value.id == "self":
            return self._resolve_self_attr(
                attr=expr.attr,
                node=expr,
                body=body
            )

        # Else the object is lowered then accessed with a getattr operation
        obj: VarRef | Any = self._lower_expr(expr=expr.value, body=body)
        out_var: str = self._new_temp()
        self._emit_op(
            target=getattr,
            qualified_name="getattr",
            args=[obj, expr.attr],
            kwargs={},
            out_var=out_var,
            body=body
        )

        # Return the reference to the attribute value
        return VarRef(name=out_var)

    def _resolve_self_attr(
        self,
        attr: str,
        node: ast.AST,
        body: list[Instruction]
    ) -> VarRef | Any:
        """
        Resolve an attribute of self in the symbol table of the module.

        Args:
            attr (str): The name of the attribute of self.
            node (ast.AST): The node of the access, for error reporting.
            body (list[Instruction]): The body to append the instructions to.

        Returns:
            VarRef | Any: A variable reference to the attribute value, or a
                literal.
        """

        # A weight or buffer is referenced by its name in the block scope
        if attr in self.weight_names:
            return VarRef(name=attr)

        # A literal attribute is embedded as a constant value
        if attr in self.literals:
            return self.literals[attr]

        # Else the attribute is resolved on the source module at run time
        out_var: str = self._new_temp()
        self._emit_op(
            target=getattr,
            qualified_name="getattr",
            args=[VarRef(name="$self"), attr],
            kwargs={},
            out_var=out_var,
            body=body
        )

        # Return the reference to the attribute value
        return VarRef(name=out_var)

    def _lower_subscript(self, expr: ast.Subscript, body: list[Instruction]) -> VarRef | Any:
        """
        Lower a subscript expression to a getitem operation.

        Args:
            expr (ast.Subscript): The subscript expression.
            body (list[Instruction]): The body to append the instructions to.

        Returns:
            VarRef | Any: A variable reference to the result, or a literal.
        """

        # A subscript of a container of self is only valid as a called function
        if (
            isinstance(expr.value, ast.Attribute)
            and isinstance(expr.value.value, ast.Name)
            and expr.value.value.id == "self"
            and expr.value.attr in self.container_names
        ):
            raise UnsupportedSyntaxError(
                "A subscript of a container is only valid as a called function", expr
            )

        # Lower the object and the index of the subscript
        obj: VarRef | Any = self._lower_expr(expr=expr.value, body=body)
        index: VarRef | Any = self._lower_index(index=expr.slice, body=body)

        # Emit the getitem operation of the subscript
        out_var: str = self._new_temp()
        self._emit_op(
            target=operator.getitem,
            qualified_name="getitem",
            args=[obj, index],
            kwargs={},
            out_var=out_var,
            body=body
        )

        # Return the reference to the subscript value
        return VarRef(name=out_var)

    def _lower_index(self, index: ast.expr, body: list[Instruction]) -> VarRef | Any:
        """
        Lower a subscript index expression, including slices.

        Args:
            index (ast.expr): The index expression of the subscript.
            body (list[Instruction]): The body to append the instructions to.

        Returns:
            VarRef | Any: A variable reference to the index, or a literal.
        """

        # A slice is lowered to the slice constructor
        if isinstance(index, ast.Slice):
            return self._lower_slice(index=index, body=body)

        # A tuple index may contain slices (e.g. x[:, 0:10])
        if isinstance(index, ast.Tuple):

            # Lower all the elements of the tuple index
            elements: list[VarRef | Any] = []
            element: ast.expr
            for element in index.elts:
                elements.append(self._lower_index(index=element, body=body))

            # Emit the tuple constructor with the lowered elements
            out_var: str = self._new_temp()
            self._emit_op(
                target=make_tuple,
                qualified_name="tuple",
                args=elements,
                kwargs={},
                out_var=out_var,
                body=body
            )

            # Return the reference to the tuple index
            return VarRef(name=out_var)

        # Else the index is a regular expression
        return self._lower_expr(expr=index, body=body)

    def _lower_slice(self, index: ast.Slice, body: list[Instruction]) -> VarRef | Any:
        """
        Lower a slice expression to the slice constructor.

        Args:
            index (ast.Slice): The slice expression to lower.
            body (list[Instruction]): The body to append the instructions to.

        Returns:
            VarRef | Any: A variable reference to the built slice.
        """

        # Lower the bounds of the slice, the missing bounds are None
        bounds: list[VarRef | Any] = []
        bound: ast.expr | None
        for bound in (index.lower, index.upper, index.step):
            if bound is None:
                bounds.append(None)
            else:
                bounds.append(self._lower_expr(expr=bound, body=body))

        # Emit the slice constructor with the bounds of the slice
        out_var: str = self._new_temp()
        self._emit_op(
            target=slice,
            qualified_name="slice",
            args=bounds,
            kwargs={},
            out_var=out_var,
            body=body
        )

        # Return the reference to the built slice
        return VarRef(name=out_var)

    def _lower_sequence(self, expr: ast.Tuple | ast.List, body: list[Instruction]) -> VarRef | Any:
        """
        Lower a tuple or list display to its constructor.

        Args:
            expr (ast.Tuple | ast.List): The sequence display expression.
            body (list[Instruction]): The body to append the instructions to.

        Returns:
            VarRef | Any: A variable reference to the constructed sequence.
        """

        # Lower all the elements of the sequence
        elements: list[VarRef | Any] = []
        element: ast.expr
        for element in expr.elts:
            elements.append(self._lower_expr(expr=element, body=body))

        # Emit the constructor of the sequence with its elements
        target: Any = make_tuple if isinstance(expr, ast.Tuple) else make_list
        out_var: str = self._new_temp()
        self._emit_op(
            target=target,
            qualified_name="tuple" if isinstance(expr, ast.Tuple) else "list",
            args=elements,
            kwargs={},
            out_var=out_var,
            body=body
        )

        # Return the reference to the constructed sequence
        return VarRef(name=out_var)

    def _lower_ifexp(self, expr: ast.IfExp, body: list[Instruction]) -> VarRef | Any:
        """
        Lower a conditional expression to an if instruction binding its output.

        Args:
            expr (ast.IfExp): The conditional expression to lower.
            body (list[Instruction]): The body to append the instructions to.

        Returns:
            VarRef | Any: A variable reference to the selected value.
        """

        # Lower the condition of the conditional expression
        condition: OpInstruction = self._lower_condition(
            test=expr.test,
            body=body
        )

        # Lower the two branches binding the same output variable
        out_var: str = self._new_temp()
        if_body: list[Instruction] = []
        else_body: list[Instruction] = []
        self._lower_assign_value(value=expr.body, out_var=out_var, body=if_body)
        self._lower_assign_value(value=expr.orelse, out_var=out_var, body=else_body)

        # Append the control instruction of the conditional expression
        body.append(ControlInstruction(
            kind="if",
            condition=condition,
            body=if_body,
            else_body=else_body,
            bind_var=None,
            out_var=out_var
        ))

        # Return the reference to the selected value
        return VarRef(name=out_var)
