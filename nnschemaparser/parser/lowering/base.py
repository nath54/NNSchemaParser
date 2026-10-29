"""
Base context of the lowering of a forward method into instructions.

Holds the symbol tables of the module and the shared lowering utilities, and
provides the entry point lowering the forward method of a module.
"""

# Import Modules
from typing import Any, Callable

import ast
import inspect
import textwrap

from torch import nn

from nnschemaparser.forward import resolve_forward
from nnschemaparser.core.instructions import Instruction, OpInstruction, ReturnInstruction, VarRef
from nnschemaparser.parser.lowering.errors import UnsupportedSyntaxError
from nnschemaparser.parser.lowering.runtime_ops import identity


class LoweringContext:
    """
    Base context holding the symbol tables and the shared lowering utilities.

    The concrete lowering classes inherit this context and add the lowering of
    the statements, the calls and the expressions on top of it.
    """

    def __init__(
        self,
        sub_block_names: set[str],
        container_names: set[str],
        weight_names: set[str],
        literals: dict[str, Any],
        dict_container_names: set[str] | None = None
    ) -> None:
        """
        Init function.

        Args:
            sub_block_names (set[str]): The names of the direct sub-blocks.
            container_names (set[str]): The names of the sub-block containers
                (nn.ModuleList, nn.ModuleDict) indexed at call time.
            weight_names (set[str]): The names of the parameters and buffers.
            literals (dict[str, Any]): The plain python attribute values.
        """

        # Store here the symbol tables of the module
        self.sub_block_names: set[str] = sub_block_names
        self.container_names: set[str] = container_names
        self.dict_container_names: set[str] = dict_container_names or set()
        self.weight_names: set[str] = weight_names
        self.literals: dict[str, Any] = literals

        # The globals of the forward method, set when the forward is lowered
        self.forward_globals: dict[str, Any] = {}

        # The loop variables bound to the entries of sub-block containers
        self._loop_module_vars: dict[str, str] = {}

        # The parameter names of the lowered forward method
        self._param_names: set[str] = set()

        # The name of the var-keyword parameter of the forward, if any
        self._kwarg_name: str | None = None

        # The local variable names of the lowered forward method
        self._local_names: set[str] = set()

        # The attributes of self stored by the lowered method body
        self._stored_self_attrs: set[str] = set()

        # The module and the function name of the currently lowered method
        self._model: nn.Module | None = None
        self._function_name: str | None = None

        # Counter used to generate the temporary variable names
        self._temp_counter: int = 0

    def lower_forward(self, model: nn.Module) -> tuple[list[Instruction], list[str]]:
        """
        Lower the forward method of a module into a list of instructions.

        Args:
            model (nn.Module): The module whose forward method is lowered.

        Returns:
            tuple[list[Instruction], list[str]]: The instructions of the body
                and the names of the returned variables.
        """

        # Lower the forward function definition of the module
        return self._lower_callable_source(
            model=model, function_name="forward"
        )

    def lower_method(
        self,
        model: nn.Module,
        method_name: str
    ) -> tuple[list[Instruction], list[str]]:
        """
        Lower another method of a module into a list of instructions.

        Args:
            model (nn.Module): The module owning the method.
            method_name (str): The name of the method to lower.

        Returns:
            tuple[list[Instruction], list[str]]: The instructions of the body
                and the names of the returned variables.
        """

        # Lower the function definition of the method
        return self._lower_callable_source(
            model=model, function_name=method_name
        )

    def _lower_callable_source(
        self,
        model: nn.Module,
        function_name: str
    ) -> tuple[list[Instruction], list[str]]:
        """
        Lower one method of a module into a list of instructions.

        Args:
            model (nn.Module): The module owning the lowered method.
            function_name (str): The name of the lowered method.

        Returns:
            tuple[list[Instruction], list[str]]: The instructions of the body
                and the names of the returned variables.
        """

        # Get the source code of the method, dedented then parsed
        method: Any = getattr(model, function_name)
        source: str = textwrap.dedent(inspect.getsource(method))
        tree: ast.Module = ast.parse(source)

        # Keep the globals of the method for the name resolutions
        self.forward_globals = dict(
            resolve_forward(module=model).__globals__
            if function_name == "forward" else method.__globals__
        )

        # Search for the function definition in the parsed tree
        function_def: ast.FunctionDef | None = None
        node: ast.stmt
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                function_def = node
                break

        # If no definition was found, the method cannot be lowered
        if function_def is None:
            raise UnsupportedSyntaxError(
                f"No {function_name} method definition found", tree
            )

        # Keep the parameter names of the method for the name roots
        self._param_names = {arg.arg for arg in function_def.args.args}

        # Keep the module and the function name for the super() resolutions
        self._model = model
        self._function_name = function_name

        # Keep the name of the var-keyword parameter of the method, if any
        self._kwarg_name = (
            function_def.args.kwarg.arg if function_def.args.kwarg else None
        )

        # Keep the local variable names of the method for the name roots
        self._collect_local_names(forward_def=function_def)

        # Lower all the statements of the method body
        body: list[Instruction] = []
        stmt: ast.stmt
        for stmt in function_def.body:
            self._lower_stmt(stmt=stmt, body=body)

        # Extract the output variable names from the last return instruction
        output_vars: list[str] = []
        if body and isinstance(body[-1], ReturnInstruction):
            output_vars = body[-1].out_vars

        # Return the lowered body and the output variable names
        return body, output_vars

    def _collect_local_names(self, forward_def: ast.FunctionDef) -> None:
        """
        Collect the local variable names of the forward method.

        Args:
            forward_def (ast.FunctionDef): The definition of the forward.
        """

        # A local variable name is any name stored in the forward body
        self._local_names = set()
        node_name: ast.Name
        for node_name in ast.walk(forward_def):
            if isinstance(node_name, ast.Name) and isinstance(node_name.ctx, ast.Store):
                self._local_names.add(node_name.id)

        # An attribute of self stored by the body is resolved at run time,
        # even when the attribute is initialized to a literal by the __init__
        self._stored_self_attrs = set()
        node: ast.AST
        for node in ast.walk(forward_def):
            if not isinstance(node, ast.Assign):
                continue
            target: ast.expr
            for target in node.targets:
                if isinstance(target, ast.Attribute) and isinstance(
                    target.value, ast.Name
                ) and target.value.id == "self":
                    self._stored_self_attrs.add(target.attr)

    def _new_temp(self) -> str:
        """
        Generate a new temporary variable name.

        Returns:
            str: The new temporary variable name.
        """

        # Increment the counter then format the temporary name
        self._temp_counter += 1
        return f"$t{self._temp_counter}"

    def _emit_op(
        self,
        target: Callable[..., Any] | VarRef,
        qualified_name: str,
        args: list[VarRef | Any],
        kwargs: dict[str, VarRef | Any],
        out_var: str,
        body: list[Instruction],
        kwargs_var: str | None = None,
        args_var: str | None = None
    ) -> OpInstruction:
        """
        Create an operation instruction and append it to the body.

        Args:
            target (Callable[..., Any] | VarRef): The callable applied by the
                instruction, or a variable reference to a callable.
            qualified_name (str): The display name of the target.
            args (list[VarRef | Any]): The positional arguments of the call.
            kwargs (dict[str, VarRef | Any]): The keyword arguments of the call.
            out_var (str): The name of the variable the result is bound to.
            body (list[Instruction]): The body to append the instruction to.
            kwargs_var (str | None): The name of a scope variable holding a
                dict unpacked as extra keyword arguments, if any.
            args_var (str | None): The name of a scope variable holding a
                tuple unpacked as extra positional arguments, if any.

        Returns:
            OpInstruction: The created and appended operation instruction.
        """

        # Create the operation instruction then append it to the body
        instruction: OpInstruction = OpInstruction(
            target=target,
            qualified_name=qualified_name,
            args=args,
            kwargs=kwargs,
            out_var=out_var,
            kwargs_var=kwargs_var,
            args_var=args_var
        )
        body.append(instruction)

        # Return the created instruction
        return instruction

    def _bind_result(
        self,
        result: VarRef | Any,
        out_var: str,
        body: list[Instruction]
    ) -> None:
        """
        Bind a lowered result to a variable of the block scope.

        Args:
            result (VarRef | Any): The lowered result, a reference or a literal.
            out_var (str): The name of the target variable of the binding.
            body (list[Instruction]): The body to append the instruction to.
        """

        # Emit the identity operation binding the result to the variable
        self._emit_op(
            target=identity,
            qualified_name="identity",
            args=[result],
            kwargs={},
            out_var=out_var,
            body=body
        )

    def _lower_condition(self, test: ast.expr, body: list[Instruction]) -> OpInstruction:
        """
        Lower a condition or iterable expression into an operation instruction.

        Args:
            test (ast.expr): The condition or iterable expression.
            body (list[Instruction]): The body to append the instructions to.

        Returns:
            OpInstruction: The operation instruction producing the value.
        """

        # Lower the expression into the body
        result: VarRef | Any = self._lower_expr(expr=test, body=body)

        # If the result is the output of the last emitted operation, reuse it
        if (
            isinstance(result, VarRef)
            and body
            and isinstance(body[-1], OpInstruction)
            and body[-1].out_var == result.name
        ):
            return body[-1]

        # Else bind the result to a dedicated temporary variable
        out_var: str = self._new_temp()
        condition: OpInstruction = self._emit_op(
            target=identity,
            qualified_name="identity",
            args=[result],
            kwargs={},
            out_var=out_var,
            body=body
        )

        # Return the operation instruction of the condition
        return condition

    def _scope_placeholder(self, value: VarRef) -> str:
        """
        Placeholder for a variable reference used as a constant keyword.

        Args:
            value (VarRef): The variable reference to placeholder.

        Returns:
            str: The name of the referenced variable.
        """

        # Return the name of the referenced variable
        return value.name
