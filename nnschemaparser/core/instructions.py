"""
Instruction types of the NNSchemaParser schema.

Each instruction represents one step of a block body and keeps the original
variable names of the parsed forward source.

Each instruction represents one step of a model block body, and keeps the
original variable names of the parsed Python code so the WebUI can display a
graph that maps back to the source.
"""

# Import Modules
from typing import Any, Callable

from torch import Tensor

from nnschemaparser.core.tensor_schema import TensorRepr, TensorReprWithValue


class VarRef:
    """
    Reference to a variable of the current block scope.
    """

    def __init__(self, name: str) -> None:
        """
        Init function.

        Args:
            name (str): The name of the referenced variable.
        """

        # Store here the name of the referenced variable
        self.name: str = name

    def __repr__(self) -> str:
        """
        Get the developer representation of the variable reference.

        Returns:
            str: E.g. "VarRef('hidden_states')".
        """

        # The name of the referenced variable
        return f"VarRef({self.name!r})"

    def __str__(self) -> str:
        """
        Get the readable representation of the variable reference.

        Returns:
            str: The name of the referenced variable.
        """

        # The name of the referenced variable
        return self.name


class BlockRef:
    """
    Reference to a sub-block of the current block.

    The path is the attribute name of the sub-block inside its parent module
    (e.g. "attention"). For containers (nn.ModuleList, nn.ModuleDict), the
    index_var is the name of the scope variable holding the index or the key.
    """

    def __init__(self, path: str, index_var: str | None = None) -> None:
        """
        Init function.

        Args:
            path (str): The attribute path of the referenced sub-block.
            index_var (str | None): The name of the scope variable holding the
                container index, if the sub-block belongs to a container.
        """

        # Store here the attribute path of the referenced sub-block
        self.path: str = path

        # Store here the index variable name for container sub-blocks
        self.index_var: str | None = index_var

    def __repr__(self) -> str:
        """
        Get the developer representation of the sub-block reference.

        Returns:
            str: E.g. "BlockRef(path='attention', index_var=None)".
        """

        # The attribute path and the optional container index variable
        return f"BlockRef(path={self.path!r}, index_var={self.index_var!r})"

    def __str__(self) -> str:
        """
        Get the readable representation of the sub-block reference.

        Returns:
            str: E.g. "attention", or "experts[expert_index]" for a container
            sub-block iterated with the index variable "expert_index".
        """

        # The container entries are displayed with their index variable
        if self.index_var is None:
            return self.path
        return f"{self.path}[{self.index_var}]"


class Instruction:
    """
    Base class of all the instructions of a block body.
    """

    def __init__(self, out_var: str) -> None:
        """
        Init function.

        Args:
            out_var (str): The name of the variable the result is bound to.
        """

        # Store here the output variable name of the instruction
        self.out_var: str = out_var

        # This field is filled later by the shape resolution pass
        self.resolved: TensorRepr | TensorReprWithValue | None = None

    def __repr__(self) -> str:
        """
        Get the developer representation of the instruction.

        Returns:
            str: E.g. "Instruction(out_var='hidden')", and the concrete class
            name for the subclasses without their own repr.
        """

        # The class name and the bound variable only, the subclasses extend it
        return f"{type(self).__name__}(out_var={self.out_var!r})"

    def __str__(self) -> str:
        """
        Get the readable representation of the instruction.

        Returns:
            str: The bound variable name, or the class name when the
            instruction binds no variable.
        """

        # The bound variable name, or the class name when binding nothing
        return self.out_var or type(self).__name__

    def resolve(self, value: Any, capture_values: bool = False) -> None:
        """
        Fill the resolved representation of the instruction output.

        Args:
            value (Any): The value produced by the instruction at execution.
            capture_values (bool): If True, the tensor values are kept inside
                the resolved representation.
        """

        # If the value is a tensor, we store a tensor representation
        if isinstance(value, Tensor):

            # If values must be captured, keep a representation with the value
            if capture_values:
                self.resolved = TensorReprWithValue(
                    shape=tuple(int(dim) for dim in value.shape),
                    dtype=str(value.dtype),
                    value=value.detach()
                )

            # Else we only keep the shape and the dtype of the tensor
            else:
                self.resolved = TensorRepr(
                    shape=tuple(int(dim) for dim in value.shape),
                    dtype=str(value.dtype)
                )

        # Else the value is a python literal, no shape is available
        else:
            self.resolved = TensorRepr(shape=(), dtype=type(value).__name__)


class OpInstruction(Instruction):
    """
    Call to a torch operation or function (e.g. torch.nn.functional.softmax).
    """

    def __init__(
        self,
        target: Callable[..., Any] | VarRef,
        qualified_name: str,
        args: list[VarRef | Any],
        kwargs: dict[str, VarRef | Any],
        out_var: str,
        kwargs_var: str | None = None,
        args_var: str | None = None
    ) -> None:
        """
        Init function.

        Args:
            target (Callable[..., Any] | VarRef): The callable applied by the
                instruction, or a variable reference to a callable resolved at
                execution time (e.g. a method obtained with getattr).
            qualified_name (str): The display name of the target
                (e.g. "torch.nn.functional.softmax").
            args (list[VarRef | Any]): The positional arguments, variable
                references or literals.
            kwargs (dict[str, VarRef | Any]): The keyword arguments, variable
                references or literals.
            out_var (str): The name of the variable the result is bound to.
        kwargs_var (str | None): The name of a scope variable holding a dict
            unpacked as extra keyword arguments (the **kwargs of the call).
        args_var (str | None): The name of a scope variable holding a tuple
            unpacked as extra positional arguments (the *args of the call).
        """

        # Init parent
        super().__init__(out_var=out_var)

        # Store here the callable applied by the instruction
        self.target: Callable[..., Any] | VarRef = target

        # Store here the display name of the target
        self.qualified_name: str = qualified_name

        # Store here the positional arguments of the call
        self.args: list[VarRef | Any] = args

        # Store here the keyword arguments of the call
        self.kwargs: dict[str, VarRef | Any] = kwargs

        # Store here the name of the unpacked keyword arguments variable
        self.kwargs_var: str | None = kwargs_var

        # Store here the name of the unpacked positional arguments variable
        self.args_var: str | None = args_var

    def __repr__(self) -> str:
        """
        Get the developer representation of the operation call.

        Returns:
            str: E.g. "OpInstruction(out_var='hidden',
            target='torch.nn.functional.linear', args=[input, weight])".
        """

        # The bound variable, the qualified call name and the brief arguments
        return (
            f"OpInstruction(out_var={self.out_var!r}, "
            f"target={self.qualified_name!r}, "
            f"args=[{_format_args(args=self.args, kwargs={})}])"
        )

    def __str__(self) -> str:
        """
        Get the readable representation of the operation call.

        Returns:
            str: E.g. "hidden = CALL torch.nn.functional.linear(input, weight)
            -> torch.float32[2, 256]", with the resolved shape when known.
        """

        # The call site, with the *args / **kwargs splats when present
        call: str = f"{self.qualified_name}({_format_args(args=self.args, kwargs=self.kwargs)}"
        if self.args_var is not None:
            call += f", *{self.args_var}"
        if self.kwargs_var is not None:
            call += f", **{self.kwargs_var}"
        call += ")"

        # The bound variable name followed by the resolved shape when known
        return f"{self.out_var} = CALL {call}{_format_resolved(repr_=self.resolved)}"


class BlockInstruction(Instruction):
    """
    Call to a sub-block of the current block.
    """

    def __init__(
        self,
        block_ref: BlockRef,
        args: list[VarRef | Any],
        out_var: str,
        kwargs: dict[str, VarRef | Any] | None = None,
        splat_args: bool = False,
        kwargs_var: str | None = None,
        args_var: str | None = None
    ) -> None:
        """
        Init function.

        Args:
            block_ref (BlockRef): The reference to the called sub-block.
            args (list[VarRef | Any]): The positional arguments of the call.
                When splat_args is True, this list holds a single variable
                reference whose tuple value is unpacked as the arguments.
            out_var (str): The name of the variable the result is bound to.
            kwargs (dict[str, VarRef | Any] | None): The keyword arguments of
                the call, bound to the sub-block inputs by name.
            splat_args (bool): Whether the argument variable holds a tuple to
                unpack as the positional arguments of the call.
            kwargs_var (str | None): The name of a scope variable holding a dict
                unpacked as extra keyword arguments (the **kwargs of the call).
            args_var (str | None): The name of a scope variable holding a
                tuple unpacked as extra positional arguments (the *args of
                the call).
        """

        # Init parent
        super().__init__(out_var=out_var)

        # Store here the reference to the called sub-block
        self.block_ref: BlockRef = block_ref

        # Store here the arguments of the sub-block call
        self.args: list[VarRef | Any] = args

        # Store here the keyword arguments of the sub-block call
        self.kwargs: dict[str, VarRef | Any] = kwargs or {}

        # Store here whether the argument variable is unpacked as the arguments
        self.splat_args: bool = splat_args

        # Store here the name of the unpacked keyword arguments variable
        self.kwargs_var: str | None = kwargs_var

        # Store here the name of the unpacked positional arguments variable
        self.args_var: str | None = args_var

    def __repr__(self) -> str:
        """
        Get the developer representation of the sub-block call.

        Returns:
            str: E.g. "BlockInstruction(out_var='hidden',
            block_ref=BlockRef(path='fc1', index_var=None), args=[x])".
        """

        # The bound variable, the referenced sub-block and the brief arguments
        return (
            f"BlockInstruction(out_var={self.out_var!r}, "
            f"block_ref={self.block_ref!r}, "
            f"args=[{_format_args(args=self.args, kwargs={})}])"
        )

    def __str__(self) -> str:
        """
        Get the readable representation of the sub-block call.

        Returns:
            str: E.g. "hidden = CALL fc1(x) -> torch.float32[2, 256]", with
            the resolved shape when known, and the splat form for the
            container iteration calls ("hidden = CALL experts(*)").
        """

        # The splat form unpacks a single tuple variable as the arguments
        if self.splat_args:
            call: str = f"{self.block_ref.path}(*)"
        else:
            call = f"{self.block_ref}({_format_args(args=self.args, kwargs=self.kwargs)}"
            if self.args_var is not None:
                call += f", *{self.args_var}"
            if self.kwargs_var is not None:
                call += f", **{self.kwargs_var}"
            call += ")"

        # The bound variable name followed by the resolved shape when known
        return f"{self.out_var} = CALL {call}{_format_resolved(repr_=self.resolved)}"


class ControlInstruction(Instruction):
    """
    Control flow instruction, an if statement or a for loop.

    For an if statement, the condition is an OpInstruction binding a temporary
    variable of the block scope. For a for loop, the iterable is either an
    OpInstruction, or the sub-block container referenced by iter_container,
    whose entries are bound to bind_var (as sub-block schemas).
    """

    def __init__(
        self,
        kind: str,
        condition: OpInstruction | None,
        body: list[Instruction],
        else_body: list[Instruction],
        bind_var: str | None,
        iter_container: BlockRef | None = None,
        index_var: str | None = None,
        out_var: str = ""
    ) -> None:
        """
        Init function.

        Args:
            kind (str): The kind of control flow, either "if" or "for".
            condition (OpInstruction | None): The condition expression for an
                if statement, the iterable expression for a for loop, or None
                when iterating a container of sub-blocks.
            body (list[Instruction]): The instructions of the main body.
            else_body (list[Instruction]): The instructions of the else body.
            bind_var (str | None): The name of the loop variable bound at each
                iteration of a for loop.
            iter_container (BlockRef | None): For a for loop over a container
                of sub-blocks, the reference to the iterated container.
            index_var (str | None): For a for loop with enumerate over a
                container, the name of the variable bound to the index.
            out_var (str): The name of the variable the result is bound to.
        """

        # Init parent
        super().__init__(out_var=out_var)

        # Only the if, for and while kinds are supported
        if kind not in ("if", "for", "while"):
            raise ValueError(f"Unsupported control flow kind: {kind}")

        # Store here the kind of control flow
        self.kind: str = kind

        # Store here the condition or iterable expression of the control flow
        self.condition: OpInstruction | None = condition

        # Store here the instructions of the main body
        self.body: list[Instruction] = body

        # Store here the instructions of the else body
        self.else_body: list[Instruction] = else_body

        # Store here the name of the loop variable of a for loop
        self.bind_var: str | None = bind_var

        # Store here the iterated container reference of a for loop
        self.iter_container: BlockRef | None = iter_container

        # Store here the enumerate index variable of a for loop
        self.index_var: str | None = index_var

    def __repr__(self) -> str:
        """
        Get the developer representation of the control flow instruction.

        Returns:
            str: E.g. "ControlInstruction(kind='if', out_var='', body=3
            instructions, else_body=1 instructions)".
        """

        # The control flow kind with the sizes of its two branches
        return (
            f"ControlInstruction(kind={self.kind!r}, out_var={self.out_var!r}, "
            f"body={len(self.body)} instructions, "
            f"else_body={len(self.else_body)} instructions)"
        )

    def __str__(self) -> str:
        """
        Get the readable representation of the control flow instruction.

        Returns:
            str: E.g. "for item in experts [1 body, 0 else]" for a loop over
            the sub-block container "experts", or "if <condition> [...]" for
            an if or while statement (the condition is re-evaluated at run
            time, its expression lives in the nested condition instruction).
        """

        # The header depends on the control flow kind
        header: str
        if self.kind == "for":
            target: str = "<iterable>"
            if self.iter_container is not None:
                target = str(self.iter_container)
            bind_var: str = self.bind_var or "?"
            header = f"for {bind_var} in {target}"
        elif self.kind == "while":
            header = "while <condition>"
        else:
            header = "if <condition>"

        # The sizes of the two branches, for a quick overview of the node
        return f"{header} [{len(self.body)} body, {len(self.else_body)} else]"


class TryInstruction(Instruction):
    """
    Try instruction of a block body, with exception handlers and a finally.

    The exception types of the handlers are resolved at lowering time and
    stored as python classes. The name of the caught exception object is not
    bound for now.
    """

    def __init__(
        self,
        body: list[Instruction],
        handlers: list[tuple[tuple[type[BaseException], ...], list[Instruction]]],
        handler_names: list[str | None],
        final_body: list[Instruction],
        out_var: str = ""
    ) -> None:
        """
        Init function.

        Args:
            body (list[Instruction]): The instructions of the try body.
            handlers (list[tuple[tuple[type[BaseException], ...], list[Instruction]]]):
                The exception types and bodies of the handlers.
            handler_names (list[str | None]): The caught exception names of
                the handlers, bound in the scope during their execution.
            final_body (list[Instruction]): The instructions of the finally body.
            out_var (str): The name of the variable the result is bound to.
        """

        # Init parent, a try instruction binds no new variable
        super().__init__(out_var=out_var)

        # Store here the instructions of the try body
        self.body: list[Instruction] = body

        # Store here the exception types and bodies of the handlers
        self.handlers: list[
            tuple[tuple[type[BaseException], ...], list[Instruction]]
        ] = handlers

        # Store here the caught exception names of the handlers
        self.handler_names: list[str | None] = handler_names

        # Store here the instructions of the finally body
        self.final_body: list[Instruction] = final_body

    def __repr__(self) -> str:
        """
        Get the developer representation of the try instruction.

        Returns:
            str: E.g. "TryInstruction(out_var='', handlers=1, final_body=0
            instructions)".
        """

        # The number of handlers and the presence of a finally body
        return (
            f"TryInstruction(out_var={self.out_var!r}, "
            f"handlers={len(self.handlers)}, "
            f"final_body={len(self.final_body)} instructions)"
        )

    def __str__(self) -> str:
        """
        Get the readable representation of the try instruction.

        Returns:
            str: E.g. "try [1 handlers, finally: 0 instructions]".
        """

        # The number of handlers and the finally body size
        return (
            f"try [{len(self.handlers)} handlers, "
            f"finally: {len(self.final_body)} instructions]"
        )


class ReturnInstruction(Instruction):
    """
    Return instruction of a block body.

    A return instruction may be nested inside a control flow body, the
    interpreter propagates its outputs up to the enclosing block.
    """

    def __init__(self, out_vars: list[str]) -> None:
        """
        Init function.

        Args:
            out_vars (list[str]): The names of the returned variables.
        """

        # Init parent, a return instruction binds no new variable
        super().__init__(out_var="")

        # Store here the names of the returned variables
        self.out_vars: list[str] = out_vars

    def __repr__(self) -> str:
        """
        Get the developer representation of the return instruction.

        Returns:
            str: E.g. "ReturnInstruction(out_vars=['hidden'])".
        """

        # The names of the returned variables
        return f"ReturnInstruction(out_vars={self.out_vars!r})"

    def __str__(self) -> str:
        """
        Get the readable representation of the return instruction.

        Returns:
            str: E.g. "RETURN hidden" or "RETURN (hidden, mask)" for several
            returned variables.
        """

        # One variable returned directly, several as a tuple
        if len(self.out_vars) == 1:
            return f"RETURN {self.out_vars[0]}"
        return f"RETURN ({', '.join(self.out_vars)})"


class ContinueInstruction(Instruction):
    """
    Continue instruction of a loop body.

    The execution of the current iteration stops and the loop goes to its
    next iteration, like the python continue statement.
    """

    def __init__(self) -> None:
        """
        Init function.
        """

        # Init parent, a continue instruction binds no new variable
        super().__init__(out_var="")

    def __repr__(self) -> str:
        """
        Get the developer representation of the continue instruction.

        Returns:
            str: The constant "ContinueInstruction()".
        """

        # No field beyond the class identity
        return "ContinueInstruction()"

    def __str__(self) -> str:
        """
        Get the readable representation of the continue instruction.

        Returns:
            str: The constant "CONTINUE".
        """

        # The python statement it stands for
        return "CONTINUE"


class BreakInstruction(Instruction):
    """
    Break instruction of a loop body.

    The execution of the loop stops, like the python break statement.
    """

    def __init__(self) -> None:
        """
        Init function.
        """

        # Init parent, a break instruction binds no new variable
        super().__init__(out_var="")

    def __repr__(self) -> str:
        """
        Get the developer representation of the break instruction.

        Returns:
            str: The constant "BreakInstruction()".
        """

        # No field beyond the class identity
        return "BreakInstruction()"

    def __str__(self) -> str:
        """
        Get the readable representation of the break instruction.

        Returns:
            str: The constant "BREAK".
        """

        # The python statement it stands for
        return "BREAK"


def _format_arg(arg: VarRef | Any) -> str:
    """
    Format an instruction argument briefly, for the reprs and the strs.

    Args:
        arg (VarRef | Any): The argument to format, a variable reference or a
            literal value.

    Returns:
        str: The name of the referenced variable, or a short literal repr.
    """

    # The name of the referenced variable
    if isinstance(arg, VarRef):
        return arg.name

    # The sub-block references are described by their readable string
    if isinstance(arg, BlockRef):
        return str(arg)

    # The common scalar literals are shown by their repr, briefly
    if arg is None or isinstance(arg, (bool, int, float, str)):
        return repr(arg)[:40]

    # Any other python value is shown by type only (callables, modules, ...)
    return type(arg).__name__


def _format_args(args: list[VarRef | Any], kwargs: dict[str, VarRef | Any]) -> str:
    """
    Format the positional and the keyword arguments of a call, briefly.

    Args:
        args (list[VarRef | Any]): The positional arguments to format.
        kwargs (dict[str, VarRef | Any]): The keyword arguments to format.

    Returns:
        str: E.g. "x, dim=1" or "" when there is no argument at all.
    """

    # The positional arguments first, then the keyword ones
    parts: list[str] = [_format_arg(arg=arg) for arg in args]
    parts.extend(f"{key}={_format_arg(arg=arg)}" for key, arg in kwargs.items())
    return ", ".join(parts)


def _format_resolved(repr_: TensorRepr | TensorReprWithValue | None) -> str:
    """
    Format the resolved representation of an instruction output, briefly.

    Args:
        repr_ (TensorRepr | TensorReprWithValue | None): The resolved
            representation of the instruction, or None when unresolved.

    Returns:
        str: E.g. " -> torch.float32[2, 16]", or "" when unresolved.
    """

    # No shape available yet, or for a scalar value
    if repr_ is None:
        return ""
    if repr_.shape == ():
        return f" -> {repr_.dtype}"
    return f" -> {repr_.dtype}{list(repr_.shape)}"
