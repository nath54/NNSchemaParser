"""
Lowering of the statements of a forward method into instructions.

Each statement of the forward body is lowered into the instruction list of the
block body, keeping the original variable names of the source code.
"""

# Import Modules
from typing import Any

import ast
import operator

from nnschemaparser.core.instructions import Instruction, OpInstruction, ReturnInstruction, VarRef
from nnschemaparser.parser.lowering.base import LoweringContext
from nnschemaparser.parser.lowering.errors import UnsupportedSyntaxError
from nnschemaparser.parser.lowering.naming import dotted_name, resolve_dotted_name
from nnschemaparser.parser.lowering.runtime_ops import (
    BINARY_OPERATORS,
    check,
    identity,
    raise_exception,
    raise_exception_instance
)


class StatementLowering(LoweringContext):
    """
    Lowers the statements of a forward method into instructions.
    """

    def _lower_stmt(self, stmt: ast.stmt, body: list[Instruction]) -> None:
        """
        Lower a statement of the forward body into the instruction list.

        Args:
            stmt (ast.stmt): The statement to lower.
            body (list[Instruction]): The body to append the instructions to.
        """

        # Docstrings and pass statements are ignored
        if isinstance(stmt, ast.Pass):
            return
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            return

        # A simple assignment to a name or a tuple of names
        if isinstance(stmt, ast.Assign):
            self._lower_assign(stmt=stmt, body=body)
            return

        # An annotated assignment is lowered like a simple assignment
        if isinstance(stmt, ast.AnnAssign):
            self._lower_ann_assign(stmt=stmt, body=body)
            return

        # An augmented assignment (x += y) is lowered to its operator function
        if isinstance(stmt, ast.AugAssign):
            self._lower_aug_assign(stmt=stmt, body=body)
            return

        # An assert statement is lowered to a checked condition
        if isinstance(stmt, ast.Assert):
            self._lower_assert(stmt=stmt, body=body)
            return

        # An expression statement is only useful for its side effects
        if isinstance(stmt, ast.Expr):
            self._lower_expr_stmt(stmt=stmt, body=body)
            return

        # A raise statement is lowered to a raising operation
        if isinstance(stmt, ast.Raise):
            self._lower_raise(stmt=stmt, body=body)
            return

        # The control flow statements are lowered to their instructions
        if isinstance(stmt, ast.If):
            self._lower_if(stmt=stmt, body=body)
            return
        if isinstance(stmt, ast.For):
            self._lower_for(stmt=stmt, body=body)
            return
        if isinstance(stmt, ast.While):
            self._lower_while(stmt=stmt, body=body)
            return
        if isinstance(stmt, ast.Try):
            self._lower_try(stmt=stmt, body=body)
            return
        if isinstance(stmt, ast.Return):
            self._lower_return(stmt=stmt, body=body)
            return

        # The statements without tensor effect are lowered to nothing
        if isinstance(stmt, ast.Delete):
            self._lower_delete(stmt=stmt, body=body)
            return

        # Any other statement is not supported
        raise UnsupportedSyntaxError(
            f"Unsupported statement: {type(stmt).__name__}", stmt
        )

    def _lower_ann_assign(self, stmt: ast.AnnAssign, body: list[Instruction]) -> None:
        """
        Lower an annotated assignment statement.

        Args:
            stmt (ast.AnnAssign): The annotated assignment to lower.
            body (list[Instruction]): The body to append the instructions to.
        """

        # Only annotated assignments with a name target are supported
        if not isinstance(stmt.target, ast.Name):
            raise UnsupportedSyntaxError("Unsupported annotated assignment", stmt)

        # An annotated assignment without value is only a type declaration
        if stmt.value is None:
            return

        # Lower the value of the assignment to the target variable
        self._lower_assign_value(
            value=stmt.value,
            out_var=stmt.target.id,
            body=body
        )

    def _lower_aug_assign(self, stmt: ast.AugAssign, body: list[Instruction]) -> None:
        """
        Lower an augmented assignment statement to its operator function.

        Args:
            stmt (ast.AugAssign): The augmented assignment to lower.
            body (list[Instruction]): The body to append the instructions to.
        """

        # Only variable targets are supported
        if not isinstance(stmt.target, ast.Name):
            raise UnsupportedSyntaxError(
                "Only variable augmented assignments are supported", stmt
            )

        # Only the mapped binary operators are supported
        op_type: type = type(stmt.op)
        if op_type not in BINARY_OPERATORS:
            raise UnsupportedSyntaxError("Unsupported augmented operator", stmt)

        # Lower the right operand then emit the operation on the variable
        right: VarRef | Any = self._lower_expr(expr=stmt.value, body=body)
        fn: Any = BINARY_OPERATORS[op_type]
        self._emit_op(
            target=fn,
            qualified_name=fn.__name__,
            args=[VarRef(name=stmt.target.id), right],
            kwargs={},
            out_var=stmt.target.id,
            body=body
        )






    def _lower_assign(self, stmt: ast.Assign, body: list[Instruction]) -> None:
        """
        Lower an assignment statement to instruction(s).

        Args:
            stmt (ast.Assign): The assignment statement to lower.
            body (list[Instruction]): The body to append the instructions to.
        """

        # Multiple assignment targets share one lowered value
        if len(stmt.targets) != 1:
            self._lower_chained_assign(stmt=stmt, body=body)
            return


        # A single name target is lowered directly
        target: ast.expr = stmt.targets[0]
        if isinstance(target, ast.Name):
            self._lower_assign_value(
                value=stmt.value,
                out_var=target.id,
                body=body
            )
            return

        # An assignment to an attribute of self is lowered to a setattr
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
            self._lower_self_attr_assign(stmt=stmt, attr=target.attr, body=body)
            return

        # An assignment to an attribute of another object is lowered to a
        # setattr of the lowered object (e.g. outputs.pooler_output = x)
        if isinstance(target, ast.Attribute):
            self._lower_attr_assign(stmt=stmt, target=target, body=body)
            return

        # A tuple of name targets is lowered by unpacking a temporary value
        if (
            isinstance(target, ast.Tuple)
            and all(isinstance(elt, ast.Name) for elt in target.elts)
        ):
            self._lower_tuple_assign(stmt=stmt, target=target, body=body)
            return

        # A tuple target containing a starred element is not supported
        if isinstance(target, ast.Tuple):
            raise UnsupportedSyntaxError(
                "Only name targets are supported in tuple assignments", stmt
            )

        # Any other assignment target is not supported
        raise UnsupportedSyntaxError(
            "Only variable assignment targets are supported", stmt
        )

    def _lower_attr_assign(
        self,
        stmt: ast.Assign,
        target: ast.Attribute,
        body: list[Instruction]
    ) -> None:
        """
        Lower an assignment to an attribute of an object as a setattr.

        Args:
            stmt (ast.Assign): The assignment statement to lower.
            target (ast.Attribute): The attribute assignment target.
            body (list[Instruction]): The body to append the instructions to.
        """

        # Lower the object owning the assigned attribute
        object_ref: VarRef | Any = self._lower_expr(expr=target.value, body=body)

        # Lower the assigned value
        value_ref: VarRef | Any = self._lower_expr(expr=stmt.value, body=body)

        # Set the attribute on the object with the builtin setattr
        self._emit_op(
            target=setattr,
            qualified_name="setattr",
            args=[object_ref, target.attr, value_ref],
            kwargs={},
            out_var=self._new_temp(),
            body=body
        )

    def _lower_self_attr_assign(
        self,
        stmt: ast.Assign,
        attr: str,
        body: list[Instruction]
    ) -> None:
        """
        Lower an assignment to an attribute of self as a setattr operation.

        Args:
            stmt (ast.Assign): The assignment statement to lower.
            attr (str): The name of the assigned attribute of self.
            body (list[Instruction]): The body to append the instructions to.
        """

        # Lower the assigned value into a temporary variable
        temp_var: str = self._lower_value_to_var(
            value=stmt.value,
            body=body
        )

        # Emit the setattr of the attribute on the source module
        self._emit_op(
            target=setattr,
            qualified_name="setattr",
            args=[VarRef(name="$self"), attr, VarRef(name=temp_var)],
            kwargs={},
            out_var=self._new_temp(),
            body=body
        )

    def _lower_chained_assign(
        self,
        stmt: ast.Assign,
        body: list[Instruction]
    ) -> None:
        """
        Lower a chained assignment (e.g. a = b = value) to instructions.

        The assigned value is lowered once into a temporary variable, then
        bound to each target of the chain.

        Args:
            stmt (ast.Assign): The chained assignment statement to lower.
            body (list[Instruction]): The body to append the instructions to.
        """

        # Lower the assigned value once into a temporary variable
        temp_var: str = self._new_temp()
        self._lower_assign_value(
            value=stmt.value,
            out_var=temp_var,
            body=body
        )

        # Bind the temporary value to each target of the chain
        target: ast.expr
        for target in stmt.targets:
            self._bind_single_target(
                target=target,
                value_var=temp_var,
                stmt=stmt,
                body=body
            )

    def _bind_single_target(
        self,
        target: ast.expr,
        value_var: str,
        stmt: ast.Assign,
        body: list[Instruction]
    ) -> None:
        """
        Bind an already lowered value to one assignment target.

        Args:
            target (ast.expr): The assignment target, a name, a tuple of
                names or an attribute.
            value_var (str): The name of the variable holding the value.
            stmt (ast.Assign): The assignment statement, for error reporting.
            body (list[Instruction]): The body to append instructions to.
        """

        # A name target is bound with an identity operation
        if isinstance(target, ast.Name):
            self._emit_op(
                target=identity,
                qualified_name="identity",
                args=[VarRef(name=value_var)],
                kwargs={},
                out_var=target.id,
                body=body
            )
            return

        # A tuple of name targets is unpacked element by element
        if (
            isinstance(target, ast.Tuple)
            and all(isinstance(elt, ast.Name) for elt in target.elts)
        ):
            element_index: int
            element: ast.expr
            for element_index, element in enumerate(target.elts):
                self._emit_op(
                    target=operator.getitem,
                    qualified_name="getitem",
                    args=[VarRef(name=value_var), element_index],
                    kwargs={},
                    out_var=element.id,
                    body=body
                )
            return

        # An attribute of self is bound with a setattr operation
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"
        ):
            self._emit_op(
                target=setattr,
                qualified_name="setattr",
                args=[VarRef(name="$self"), target.attr, VarRef(name=value_var)],
                kwargs={},
                out_var=self._new_temp(),
                body=body
            )
            return

        # Any other target is not supported
        raise UnsupportedSyntaxError(
            "Only name, tuple and attribute targets are supported", stmt
        )

    def _lower_tuple_assign(
        self,
        stmt: ast.Assign,
        target: ast.Tuple,
        body: list[Instruction]
    ) -> None:
        """
        Lower an assignment to a tuple of names by unpacking a temporary value.

        Args:
            stmt (ast.Assign): The assignment statement to lower.
            target (ast.Tuple): The tuple of name targets of the assignment.
            body (list[Instruction]): The body to append the instructions to.
        """

        # Lower the value of the assignment into a temporary variable
        temp_var: str = self._new_temp()
        if isinstance(stmt.value, ast.Call):
            self._lower_call(call=stmt.value, body=body, out_var=temp_var)
        else:
            result: VarRef | Any = self._lower_expr(expr=stmt.value, body=body)
            self._bind_result(result=result, out_var=temp_var, body=body)

        # Unpack the temporary value element by element
        self._bind_single_target(
            target=target,
            value_var=temp_var,
            stmt=stmt,
            body=body
        )

    def _lower_assign_value(
        self,
        value: ast.expr,
        out_var: str,
        body: list[Instruction]
    ) -> None:
        """
        Lower the value of an assignment to the given target variable.

        Args:
            value (ast.expr): The value expression of the assignment.
            out_var (str): The name of the target variable of the assignment.
            body (list[Instruction]): The body to append the instructions to.
        """

        # A call expression becomes an operation or block instruction
        if isinstance(value, ast.Call):
            self._lower_call(call=value, body=body, out_var=out_var)
            return

        # Else the expression is lowered then bound to the target variable
        result: VarRef | Any = self._lower_expr(expr=value, body=body)
        self._bind_result(result=result, out_var=out_var, body=body)

    def _lower_value_to_var(self, value: ast.expr, body: list[Instruction]) -> str:
        """
        Lower a value expression into a temporary variable.

        Args:
            value (ast.expr): The value expression to lower.
            body (list[Instruction]): The body to append the instructions to.

        Returns:
            str: The name of the variable holding the value.
        """

        # A call expression is lowered to an instruction bound to a temporary
        if isinstance(value, ast.Call):
            out_var: str = self._new_temp()
            self._lower_call(call=value, body=body, out_var=out_var)
            return out_var

        # Else the expression is lowered then bound to a temporary variable
        result: VarRef | Any = self._lower_expr(expr=value, body=body)
        if isinstance(result, VarRef):
            return result.name
        out_var = self._new_temp()
        self._bind_result(result=result, out_var=out_var, body=body)
        return out_var

    def _lower_return(self, stmt: ast.Return, body: list[Instruction]) -> None:
        """
        Lower a return statement into a return instruction.

        Args:
            stmt (ast.Return): The return statement to lower.
            body (list[Instruction]): The body to append the instructions to.
        """

        # Initialize the names of the returned variables
        out_vars: list[str] = []
        value: ast.expr | None = stmt.value

        # A return with a value
        if value is not None:

            # A tuple return maps to the list of its element variables
            if isinstance(value, ast.Tuple):
                elt: ast.expr
                for elt in value.elts:
                    out_vars.append(self._return_var(expr=elt, body=body))

            # A single value return
            else:
                out_vars.append(self._return_var(expr=value, body=body))

        # Append the return instruction to the body
        body.append(ReturnInstruction(out_vars=out_vars))

    def _return_var(self, expr: ast.expr, body: list[Instruction]) -> str:
        """
        Get the name of the variable returned by a return element.

        Args:
            expr (ast.expr): The element expression of the return.
            body (list[Instruction]): The body to append the instructions to.

        Returns:
            str: The name of the variable holding the returned value.
        """

        # A returned variable name is used directly
        if isinstance(expr, ast.Name):
            return expr.id

        # Else the expression is lowered then its temporary name is returned
        result: VarRef | Any = self._lower_expr(expr=expr, body=body)

        # A variable result is returned directly
        if isinstance(result, VarRef):
            return result.name

        # A literal result is bound to a temporary variable
        out_var: str = self._new_temp()
        self._bind_result(result=result, out_var=out_var, body=body)
        return out_var
