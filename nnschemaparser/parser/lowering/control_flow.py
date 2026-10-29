"""
Lowering of the control flow statements of a forward method.

The if statements, the for loops and the try statements are lowered to their
corresponding control flow instructions.
"""

# Import Modules
from typing import Any

import ast

from nnschemaparser.core.instructions import (
    BlockRef,
    ControlInstruction,
    Instruction,
    OpInstruction,
    TryInstruction
)
from nnschemaparser.parser.lowering.base import LoweringContext
from nnschemaparser.parser.lowering.errors import UnsupportedSyntaxError
from nnschemaparser.parser.lowering.naming import dotted_name, resolve_dotted_name


class ControlFlowLowering(LoweringContext):
    """
    Lowers the control flow statements of a forward method into instructions.
    """

    def _lower_if(self, stmt: ast.If, body: list[Instruction]) -> None:
        """
        Lower an if statement to a control instruction.

        Args:
            stmt (ast.If): The if statement to lower.
            body (list[Instruction]): The body to append the instruction to.
        """

        # Lower the condition expression into a temporary operation
        condition: OpInstruction = self._lower_condition(test=stmt.test, body=body)

        # Lower the bodies of the branches of the if statement
        if_body: list[Instruction] = self._lower_body(stmt.body)
        else_body: list[Instruction] = self._lower_body(stmt.orelse)

        # Append the control instruction of the if statement
        body.append(ControlInstruction(
            kind="if",
            condition=condition,
            body=if_body,
            else_body=else_body,
            bind_var=None
        ))

    def _lower_for(self, stmt: ast.For, body: list[Instruction]) -> None:
        """
        Lower a for loop statement to a control instruction.

        Args:
            stmt (ast.For): The for loop statement to lower.
            body (list[Instruction]): The body to append the instruction to.
        """

        # The for-else syntax is not supported
        if stmt.orelse:
            raise UnsupportedSyntaxError("The for-else syntax is not supported", stmt)

        # A loop over the sub-blocks of self itself (e.g. nn.Sequential)
        if isinstance(stmt.iter, ast.Name) and stmt.iter.id == "self":
            self._lower_self_loop(stmt=stmt, container_name="$self", body=body)
            return

        # A loop over a sub-block container of the module
        container_name: str | None = self._self_container_attr(expr=stmt.iter)
        if container_name is not None:
            self._lower_self_loop(stmt=stmt, container_name=container_name, body=body)
            return

        # A loop with enumerate over a sub-block container of the module
        if self._is_enumerate_container_loop(stmt=stmt):
            self._lower_enumerate_loop(stmt=stmt, body=body)
            return

        # Else a regular loop over an iterable expression
        self._lower_iterable_loop(stmt=stmt, body=body)

    def _lower_while(self, stmt: ast.While, body: list[Instruction]) -> None:
        """
        Lower a while loop statement to a control instruction.

        Args:
            stmt (ast.While): The while loop statement to lower.
            body (list[Instruction]): The body to append the instruction to.
        """

        # The while-else syntax is not supported
        if stmt.orelse:
            raise UnsupportedSyntaxError(
                "The while-else syntax is not supported", stmt
            )

        # Lower the condition of the loop, re-evaluated at each iteration
        condition: OpInstruction = self._lower_condition(
            test=stmt.test, body=body
        )

        # Lower the body of the loop
        loop_body: list[Instruction] = self._lower_body(stmt.body)

        # Append the control instruction of the while loop
        body.append(ControlInstruction(
            kind="while",
            condition=condition,
            body=loop_body,
            else_body=[],
            bind_var=None
        ))

    def _lower_self_loop(
        self,
        stmt: ast.For,
        container_name: str,
        body: list[Instruction]
    ) -> None:
        """
        Lower a for loop over a sub-block container of self.

        Args:
            stmt (ast.For): The for loop statement to lower.
            container_name (str): The name of the iterated container.
            body (list[Instruction]): The body to append the instruction to.
        """

        # Only simple loop variables are supported over containers
        if not isinstance(stmt.target, ast.Name):
            raise UnsupportedSyntaxError(
                "Only simple loop variables are supported over containers", stmt
            )

        # Lower the container loop with the loop variable name
        self._lower_container_for(
            stmt=stmt,
            container_name=container_name,
            body=body,
            bind_var=stmt.target.id,
            index_var=None
        )

    def _is_enumerate_container_loop(self, stmt: ast.For) -> bool:
        """
        Check whether a for loop is an enumerate over a sub-block container.

        Args:
            stmt (ast.For): The for loop statement to check.

        Returns:
            bool: Whether the loop is an enumerate over a container of self.
        """

        # The iterable must be an enumerate call with a single argument
        return (
            isinstance(stmt.iter, ast.Call)
            and dotted_name(expr=stmt.iter.func) == "enumerate"
            and len(stmt.iter.args) == 1
            and self._self_container_attr(expr=stmt.iter.args[0]) is not None
        )

    def _lower_enumerate_loop(self, stmt: ast.For, body: list[Instruction]) -> None:
        """
        Lower a for loop with enumerate over a sub-block container of self.

        Args:
            stmt (ast.For): The for loop statement to lower.
            body (list[Instruction]): The body to append the instruction to.
        """

        # Get the iterated container of the enumerate call
        inner_container: str | None = self._self_container_attr(
            expr=stmt.iter.args[0]
        )

        # Only tuple targets of two names are supported
        if not (
            isinstance(stmt.target, ast.Tuple)
            and len(stmt.target.elts) == 2
            and all(isinstance(elt, ast.Name) for elt in stmt.target.elts)
        ):
            raise UnsupportedSyntaxError("Unsupported enumerate loop target", stmt)

        # Lower the container loop with the index and module variables
        self._lower_container_for(
            stmt=stmt,
            container_name=inner_container,
            body=body,
            bind_var=stmt.target.elts[1].id,
            index_var=stmt.target.elts[0].id
        )

    def _lower_iterable_loop(self, stmt: ast.For, body: list[Instruction]) -> None:
        """
        Lower a regular for loop over an iterable expression.

        Args:
            stmt (ast.For): The for loop statement to lower.
            body (list[Instruction]): The body to append the instruction to.
        """

        # Only simple loop variables are supported
        if not isinstance(stmt.target, ast.Name):
            raise UnsupportedSyntaxError(
                "Only simple loop variables are supported", stmt
            )

        # Lower the iterable expression into a temporary operation
        iterable: OpInstruction = self._lower_condition(test=stmt.iter, body=body)

        # Lower the body of the loop
        loop_body: list[Instruction] = self._lower_body(stmt.body)

        # Append the control instruction of the for loop
        body.append(ControlInstruction(
            kind="for",
            condition=iterable,
            body=loop_body,
            else_body=[],
            bind_var=stmt.target.id
        ))

    def _lower_container_for(
        self,
        stmt: ast.For,
        container_name: str,
        body: list[Instruction],
        bind_var: str,
        index_var: str | None
    ) -> None:
        """
        Lower a for loop over a sub-block container to a control instruction.

        Args:
            stmt (ast.For): The for loop statement to lower.
            container_name (str): The name of the iterated container.
            body (list[Instruction]): The body to append the instruction to.
            bind_var (str): The name of the loop variable of the entries.
            index_var (str | None): The name of the enumerate index variable.
        """

        # Register the loop variable as a container entry variable for calls
        previous_container: str | None = self._loop_module_vars.get(bind_var)
        self._loop_module_vars[bind_var] = container_name

        # Lower the body of the loop
        loop_body: list[Instruction] = self._lower_body(stmt.body)

        # Restore the previous binding of the loop variable, a container
        # entry name is only valid inside the body of its loop
        if previous_container is None:
            self._loop_module_vars.pop(bind_var, None)
        else:
            self._loop_module_vars[bind_var] = previous_container

        # Append the control instruction of the container loop
        body.append(ControlInstruction(
            kind="for",
            condition=None,
            body=loop_body,
            else_body=[],
            bind_var=bind_var,
            iter_container=BlockRef(path=container_name),
            index_var=index_var
        ))

    def _self_container_attr(self, expr: ast.expr) -> str | None:
        """
        Get the container name of a self.<container> attribute expression.

        Args:
            expr (ast.expr): The expression to check.

        Returns:
            str | None: The container name, or None if not a self container.
        """

        # A self attribute naming a sub-block container of the module
        if (
            isinstance(expr, ast.Attribute)
            and isinstance(expr.value, ast.Name)
            and expr.value.id == "self"
            and expr.attr in self.container_names
        ):
            return expr.attr

        # Any other expression is not a self container
        return None

    def _lower_try(self, stmt: ast.Try, body: list[Instruction]) -> None:
        """
        Lower a try statement to a try instruction.

        Args:
            stmt (ast.Try): The try statement to lower.
            body (list[Instruction]): The body to append the instruction to.
        """

        # Lower the body of the try statement
        try_body: list[Instruction] = self._lower_body(stmt.body)

        # Lower the handlers of the try statement
        handlers: list[tuple[tuple[type[BaseException], ...], list[Instruction]]] = []
        handler_names: list[str | None] = []
        handler: ast.ExceptHandler
        for handler in stmt.handlers:
            handlers.append(self._lower_handler(handler=handler))
            handler_names.append(handler.name)

        # Lower the finally body of the try statement
        final_body: list[Instruction] = self._lower_body(stmt.finalbody)

        # Append the try instruction to the body
        body.append(TryInstruction(
            body=try_body,
            handlers=handlers,
            handler_names=handler_names,
            final_body=final_body
        ))

    def _lower_handler(
        self,
        handler: ast.ExceptHandler
    ) -> tuple[tuple[type[BaseException], ...], list[Instruction]]:
        """
        Lower an exception handler of a try statement.

        Args:
            handler (ast.ExceptHandler): The handler to lower.

        Returns:
            tuple[tuple[type[BaseException], ...], list[Instruction]]: The
                resolved exception types and the lowered body of the handler.
        """

        # Resolve the exception types of the handler at lowering time
        exception_types: tuple[type[BaseException], ...] = self._handler_types(
            handler=handler
        )

        # Lower the body of the handler
        handler_body: list[Instruction] = self._lower_body(handler.body)

        # Return the resolved types and the lowered body of the handler
        return exception_types, handler_body

    def _handler_types(self, handler: ast.ExceptHandler) -> tuple[type[BaseException], ...]:
        """
        Resolve the exception types of an exception handler.

        Args:
            handler (ast.ExceptHandler): The handler whose types are resolved.

        Returns:
            tuple[type[BaseException], ...]: The resolved exception types.

        Raises:
            UnsupportedSyntaxError: When the handler type is not supported.
        """

        # A handler without a type catches every exception
        if handler.type is None:
            return (BaseException,)

        # A single exception type name is resolved from the forward globals
        if isinstance(handler.type, ast.Name):
            return (
                resolve_dotted_name(
                    dotted=handler.type.id,
                    node=handler.type,
                    forward_globals=self.forward_globals
                ),
            )

        # A tuple of exception type names is resolved element by element
        if isinstance(handler.type, ast.Tuple):
            return tuple(
                resolve_dotted_name(
                    dotted=elt.id,
                    node=elt,
                    forward_globals=self.forward_globals
                )
                for elt in handler.type.elts
                if isinstance(elt, ast.Name)
            )

        # Any other handler type expression is not supported
        raise UnsupportedSyntaxError("Unsupported exception type", handler)

    def _lower_body(self, stmts: list[ast.stmt]) -> list[Instruction]:
        """
        Lower a list of statements of a control flow body.

        Args:
            stmts (list[ast.stmt]): The statements to lower.

        Returns:
            list[Instruction]: The lowered instructions of the body.
        """

        # Lower all the statements into a new instruction list
        body: list[Instruction] = []
        inner: ast.stmt
        for inner in stmts:
            self._lower_stmt(stmt=inner, body=body)

        # Return the lowered body
        return body
