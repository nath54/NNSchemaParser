"""
Lowering of the calls of a forward method into instructions.

The calls to sub-blocks, to sub-block containers and to torch functions are
dispatched to their specific instruction lowering.
"""

# Import Modules
from typing import Any

import ast

import builtins

from nnschemaparser.core.instructions import BlockInstruction, BlockRef, Instruction, VarRef
from nnschemaparser.parser.lowering.base import LoweringContext
from nnschemaparser.core.container_keys import container_entry_keys
from nnschemaparser.parser.lowering.errors import UnsupportedSyntaxError
from nnschemaparser.parser.lowering.naming import (
    RESOLVE_NAMESPACE,
    GlobalCallable,
    dotted_name,
    resolve_dotted_name
)
from nnschemaparser.parser.lowering.runtime_ops import identity


class CallLowering(LoweringContext):
    """
    Lowers the calls of a forward method into instructions.
    """

    def _lower_call(
        self,
        call: ast.Call,
        body: list[Instruction],
        out_var: str
    ) -> str:
        """
        Lower a call expression to an operation or block instruction.

        Args:
            call (ast.Call): The call expression to lower.
            body (list[Instruction]): The body to append the instructions to.
            out_var (str): The name of the variable the result is bound to.

        Returns:
            str: The name of the variable the result is bound to.
        """

        # The kwargs unpacking must be a plain variable name, the starred
        # arguments are handled inside the lowering of the call arguments
        kwargs_var: str | None = self._kwargs_var(call=call)

        # A call to a direct sub-block of the module
        if self._is_self_sub_block_call(call=call):
            return self._lower_block_call(
                block_ref=BlockRef(path=call.func.attr),
                call=call,
                body=body,
                out_var=out_var,
                kwargs_var=kwargs_var
            )

        # A call to a loop variable holding a container entry sub-block
        if isinstance(call.func, ast.Name) and call.func.id in self._loop_module_vars:
            return self._lower_block_call(
                block_ref=BlockRef(
                    path=self._loop_module_vars[call.func.id],
                    index_var=call.func.id
                ),
                call=call,
                body=body,
                out_var=out_var,
                kwargs_var=kwargs_var
            )

        # A call to a sub-block container of the module (e.g. a Sequential)
        if self._is_self_container_call(call=call):
            return self._lower_container_call(
                container_name=call.func.attr,
                call=call,
                body=body,
                out_var=out_var
            )

        # A call to a sub-block of a container (e.g. self.experts[i](x))
        if self._is_indexed_container_call(call=call):
            return self._lower_indexed_container_call(
                call=call,
                body=body,
                out_var=out_var,
                kwargs_var=kwargs_var
            )

        # Else the call is lowered to an operation instruction
        return self._lower_op_call(
            call=call,
            body=body,
            out_var=out_var,
            kwargs_var=kwargs_var
        )

    def _kwargs_var(self, call: ast.Call) -> str | None:
        """
        Get the name of the variable unpacked as the keyword arguments.

        Args:
            call (ast.Call): The call expression to check.

        Returns:
            str | None: The name of the unpacked keyword dict variable, or
                None when the call has no keyword argument unpacking.

        Raises:
            UnsupportedSyntaxError: When the kwargs unpacking is not a plain
                variable name.
        """

        # Each keyword argument unpacking must be a plain variable name
        kwargs_var: str | None = None
        kw: ast.keyword
        for kw in call.keywords:
            if kw.arg is None:
                if not isinstance(kw.value, ast.Name):
                    raise UnsupportedSyntaxError(
                        "Unsupported **kwargs unpacking expression", kw
                    )
                kwargs_var = kw.value.id
        return kwargs_var

    def _is_self_sub_block_call(self, call: ast.Call) -> bool:
        """
        Check whether a call targets a direct sub-block of the module.

        Args:
            call (ast.Call): The call expression to check.

        Returns:
            bool: Whether the call is a self.<sub_block>(...) call.
        """

        # The call must be an attribute of self naming a sub-block
        return (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "self"
            and call.func.attr in self.sub_block_names
        )

    def _is_self_container_call(self, call: ast.Call) -> bool:
        """
        Check whether a call targets a sub-block container of the module.

        Args:
            call (ast.Call): The call expression to check.

        Returns:
            bool: Whether the call is a self.<container>(...) call.
        """

        # The call must be an attribute of self naming a container
        return (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "self"
            and call.func.attr in self.container_names
            and call.func.attr not in self.dict_container_names
        )

    def _is_indexed_container_call(self, call: ast.Call) -> bool:
        """
        Check whether a call targets an indexed entry of a container.

        Args:
            call (ast.Call): The call expression to check.

        Returns:
            bool: Whether the call is a self.<container>[index](...) call.
        """

        # The call must subscript a container attribute of self
        return (
            isinstance(call.func, ast.Subscript)
            and isinstance(call.func.value, ast.Attribute)
            and isinstance(call.func.value.value, ast.Name)
            and call.func.value.value.id == "self"
            and call.func.value.attr in self.container_names
        )

    def _lower_indexed_container_call(
        self,
        call: ast.Call,
        body: list[Instruction],
        out_var: str,
        kwargs_var: str | None = None
    ) -> str:
        """
        Lower a call to an indexed entry of a sub-block container.

        Args:
            call (ast.Call): The call expression to lower.
            body (list[Instruction]): The body to append the instructions to.
            out_var (str): The name of the variable the result is bound to.

        Returns:
            str: The name of the variable the result is bound to.
        """

        # A negative constant index counts from the end of the container, so
        # it is normalized with the entry count of the container. Note that
        # the python syntax -1 is a unary minus on a positive constant, not
        # a negative constant node
        index_expr: ast.expr = call.func.slice
        index_value: int | None = None
        if isinstance(index_expr, ast.Constant) and isinstance(
            index_expr.value, int
        ):
            index_value = index_expr.value
        elif (
            isinstance(index_expr, ast.UnaryOp)
            and isinstance(index_expr.op, ast.USub)
            and isinstance(index_expr.operand, ast.Constant)
            and isinstance(index_expr.operand.value, int)
        ):
            index_value = -index_expr.operand.value
        if index_value is not None and index_value < 0:
            entry_count: int = len(
                container_entry_keys(
                    sub_block_names=self.sub_block_names,
                    container_name=call.func.value.attr
                )
            )
            index_expr = ast.Constant(value=entry_count + index_value)

        # Lower the index expression of the container entry
        index_var: str = self._lower_container_index(
            index_expr=index_expr,
            body=body
        )

        # Lower the block call with the indexed block reference
        return self._lower_block_call(
            block_ref=BlockRef(path=call.func.value.attr, index_var=index_var),
            call=call,
            body=body,
            out_var=out_var,
            kwargs_var=kwargs_var
        )

    def _lower_op_call(
        self,
        call: ast.Call,
        body: list[Instruction],
        out_var: str,
        kwargs_var: str | None = None
    ) -> str:
        """
        Lower a call to a torch function, a builtin or a method of an object.

        Args:
            call (ast.Call): The call expression to lower.
            body (list[Instruction]): The body to append the instructions to.
            out_var (str): The name of the variable the result is bound to.

        Returns:
            str: The name of the variable the result is bound to.
        """

        # Lower the positional and the keyword arguments of the call once
        args: list[VarRef | Any]
        kwargs: dict[str, VarRef | Any]
        args, kwargs, kwargs_var, args_var = self._lower_call_args(
            call=call, body=body
        )

        # A super call is lowered with its arguments resolved statically
        if isinstance(call.func, ast.Name) and call.func.id == "super":
            self._lower_super_call(
                call=call, body=body, out_var=out_var, kwargs_var=kwargs_var
            )
            return out_var

        # A call to a local callable variable (e.g. a loss function instance)
        if isinstance(call.func, ast.Name) and (
            call.func.id in self._param_names
            or call.func.id == self._kwarg_name
            or call.func.id in self._local_names
        ):
            self._emit_op(
                target=VarRef(name=call.func.id),
                qualified_name=call.func.id,
                args=args,
                kwargs=kwargs,
                out_var=out_var,
                body=body,
                kwargs_var=kwargs_var,
                args_var=args_var
            )
            return out_var

        # A call on the result of another call (e.g. nn.Softmax(dim=-1)(x)):
        # the inner call is lowered first, then its result is called
        if isinstance(call.func, ast.Call):
            inner_var: str = self._new_temp()
            self._lower_call(call=call.func, body=body, out_var=inner_var)
            self._emit_op(
                target=VarRef(name=inner_var),
                qualified_name=inner_var,
                args=args,
                kwargs=kwargs,
                out_var=out_var,
                body=body,
                kwargs_var=kwargs_var,
                args_var=args_var
            )
            return out_var

        # A call to a dotted name (e.g. torch.softmax) or a builtin (e.g. range)
        if self._lower_dotted_call(
            call=call,
            args=args,
            kwargs=kwargs,
            out_var=out_var,
            body=body,
            kwargs_var=kwargs_var,
            args_var=args_var
        ):
            return out_var

        # A method call on an object: getattr then call
        return self._lower_method_call(
            call=call,
            args=args,
            kwargs=kwargs,
            out_var=out_var,
            body=body,
            kwargs_var=kwargs_var,
            args_var=args_var
        )

    def _lower_dotted_call(
        self,
        call: ast.Call,
        args: list[VarRef | Any],
        kwargs: dict[str, VarRef | Any],
        out_var: str,
        body: list[Instruction],
        kwargs_var: str | None = None,
        args_var: str | None = None
    ) -> bool:
        """
        Lower a call to a dotted name, when the name is resolvable.

        Args:
            call (ast.Call): The call expression to lower.
            args (list[VarRef | Any]): The lowered positional arguments.
            kwargs (dict[str, VarRef | Any]): The lowered keyword arguments.
            out_var (str): The name of the variable the result is bound to.
            body (list[Instruction]): The body to append the instructions to.
            kwargs_var (str | None): The name of the unpacked keyword dict.
            args_var (str | None): The name of the unpacked positional tuple.

        Returns:
            bool: Whether the call was lowered as a dotted name call.
        """

        # Get the dotted name of the called function
        dotted: str | None = dotted_name(expr=call.func)
        if dotted is None:
            return False

        # Get the root of the dotted name of the call
        root: str = dotted.split(".")[0]

        # The self root is handled by the method call branch
        # The parameter and local roots are not globals of the forward
        if (
            root == "self"
            or root == self._kwarg_name
            or root in self._param_names
            or root in self._local_names
        ):
            return False

        # A known namespace, builtin or forward global is resolved eagerly
        if (
            root in RESOLVE_NAMESPACE
            or hasattr(builtins, root)
            or root in self.forward_globals
        ):
            # Resolve the callable of the call from its dotted name
            call_target: Any = resolve_dotted_name(
                dotted=dotted,
                node=call.func,
                forward_globals=self.forward_globals
            )

            # A failed optional import is resolved lazily at run time
            if call_target is None:
                call_target = GlobalCallable(
                    dotted=dotted,
                    forward_globals=self.forward_globals
                )

        # Else the name is resolved lazily at run time (e.g. a local import)
        else:
            call_target = GlobalCallable(
                dotted=dotted,
                forward_globals=self.forward_globals
            )

        # Emit the operation of the resolved callable
        self._emit_op(
            target=call_target,
            qualified_name=dotted,
            args=args,
            kwargs=kwargs,
            out_var=out_var,
            body=body,
            kwargs_var=kwargs_var,
            args_var=args_var
        )

        # The call was lowered as a dotted name call
        return True

    def _lower_method_call(
        self,
        call: ast.Call,
        args: list[VarRef | Any],
        kwargs: dict[str, VarRef | Any],
        out_var: str,
        body: list[Instruction],
        kwargs_var: str | None = None,
        args_var: str | None = None
    ) -> str:
        """
        Lower a method call on an object to a getattr then call.

        Args:
            call (ast.Call): The call expression to lower.
            args (list[VarRef | Any]): The lowered positional arguments.
            kwargs (dict[str, VarRef | Any]): The lowered keyword arguments.
            out_var (str): The name of the variable the result is bound to.
            body (list[Instruction]): The body to append the instructions to.
            kwargs_var (str | None): The name of the unpacked keyword dict.
            args_var (str | None): The name of the unpacked positional tuple.

        Returns:
            str: The name of the variable the result is bound to.
        """

        # The call must be an attribute access of the method
        if not isinstance(call.func, ast.Attribute):
            raise UnsupportedSyntaxError("Unsupported call expression", call.func)

        # A method of self is resolved on the source module at run time
        if isinstance(call.func.value, ast.Name) and call.func.value.id == "self":
            obj: VarRef | Any = VarRef(name="$self")

        # Else the object of the method call is lowered
        else:
            obj = self._lower_expr(expr=call.func.value, body=body)

        # Resolve the method of the object with a getattr operation
        method_var: str = self._new_temp()
        self._emit_op(
            target=getattr,
            qualified_name="getattr",
            args=[obj, call.func.attr],
            kwargs={},
            out_var=method_var,
            body=body
        )

        # The display name of the receiver for the call instruction
        receiver: str = obj.name if isinstance(obj, VarRef) else repr(obj)[:40]

        # Call the resolved method
        self._emit_op(
            target=VarRef(name=method_var),
            qualified_name=f"{receiver}.{call.func.attr}",
            args=args,
            kwargs=kwargs,
            out_var=out_var,
            body=body,
            kwargs_var=kwargs_var,
            args_var=args_var
        )

        # Return the output variable name of the call
        return out_var

    def _lower_super_call(
        self,
        call: ast.Call,
        body: list[Instruction],
        out_var: str,
        kwargs_var: str | None
    ) -> None:
        """
        Lower a super() call to a super operation.

        The zero-argument form is rewritten with the class where the lowered
        method is defined, so the call is re-executable outside of the
        original function frame.

        Args:
            call (ast.Call): The super call expression to lower.
            body (list[Instruction]): The body to append the instructions to.
            out_var (str): The name of the variable the result is bound to.
            kwargs_var (str | None): The name of the unpacked keyword
                arguments, unsupported for a super call.

        Raises:
            UnsupportedSyntaxError: When the class argument of the super call
                cannot be resolved statically.
        """

        # The keyword arguments are not supported in a super call
        if kwargs_var is not None:
            raise UnsupportedSyntaxError(
                "Keyword arguments are not supported in a super() call", call
            )

        # The explicit two-argument form keeps its own arguments
        if len(call.args) == 2:
            class_value: Any = self._resolve_super_class(
                arg=call.args[0], node=call
            )
            obj: VarRef | Any = self._lower_expr(expr=call.args[1], body=body)

        # The zero-argument form uses the class defining the lowered method
        else:
            class_value = self._defining_class(node=call)
            obj = VarRef(name="$self")

        # Emit the super operation with the resolved arguments
        self._emit_op(
            target=super,
            qualified_name="super",
            args=[class_value, obj],
            kwargs={},
            out_var=out_var,
            body=body
        )

    def _resolve_super_class(
        self,
        arg: ast.expr,
        node: ast.AST
    ) -> type:
        """
        Resolve the class argument of an explicit super call.

        Args:
            arg (ast.expr): The class argument of the super call.
            node (ast.AST): The super call node, for error reporting.

        Returns:
            type: The resolved class object.

        Raises:
            UnsupportedSyntaxError: When the argument is not a name resolvable
                to a class from the globals of the lowered method.
        """

        # A plain name is resolved from the globals of the lowered method
        if isinstance(arg, ast.Name) and isinstance(
            self.forward_globals.get(arg.id), type
        ):
            resolved: Any = self.forward_globals[arg.id]
            return resolved

        # Any other argument cannot be resolved statically
        raise UnsupportedSyntaxError(
            "The class argument of super() must be a resolvable class name",
            node
        )

    def _defining_class(self, node: ast.AST) -> type:
        """
        Get the class where the currently lowered method is defined.

        Args:
            node (ast.AST): The super call node, for error reporting.

        Returns:
            type: The class of the method resolution order defining the
                lowered method.

        Raises:
            UnsupportedSyntaxError: When no lowered method context is
                available or the defining class cannot be found.
        """

        # The lowered method context must be available
        if self._model is None or self._function_name is None:
            raise UnsupportedSyntaxError(
                "No lowered method context for the zero-argument super()",
                node
            )

        # Search the method resolution order for the defining class
        model_type: type = type(self._model)
        function: Any = getattr(model_type, self._function_name)
        cls: type
        for cls in model_type.__mro__:
            if (
                self._function_name in cls.__dict__
                and cls.__dict__[self._function_name] is function
            ):
                return cls

        # The defining class was not found in the resolution order
        raise UnsupportedSyntaxError(
            "The class defining the lowered method was not found", node
        )
