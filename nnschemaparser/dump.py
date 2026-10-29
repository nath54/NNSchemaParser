"""
Readable text dumpers of the schema.

The dumpers turn a ModelBlockSchema tree into a human readable text view:
the recursive block dump shows the inputs, outputs, weights, sub-blocks and
the instruction body of every block, with the resolved shapes of every step.
"""

# Import Modules
from typing import Any

import torch
from torch import Tensor

from nnschemaparser.core.instructions import (
    BlockInstruction,
    ControlInstruction,
    Instruction,
    OpInstruction,
    ReturnInstruction,
    TryInstruction,
    VarRef
)
from nnschemaparser.core.model_schema import ModelBlockSchema
from nnschemaparser.core.tensor_schema import (
    PLACEHOLDER_DTYPE,
    TensorRepr,
    TensorReprWithValue
)

# The maximum number of lines written to a dump file
MAX_LINES: int = 20000


def format_repr(repr_: TensorRepr) -> str:
    """
    Format a tensor representation as its dtype and shape.

    Args:
        repr_ (TensorRepr): The representation to format.

    Returns:
        str: E.g. "torch.float32[2, 16]", or "bool" for scalar values.
    """

    # The scalar python values have no shape brackets
    if repr_.shape == ():
        return repr_.dtype
    return f"{repr_.dtype}{list(repr_.shape)}"


def format_input(repr_: TensorRepr) -> str:
    """
    Format a block input representation, placeholders included.

    Args:
        repr_ (TensorRepr): The representation of the forward parameter.

    Returns:
        str: E.g. "torch.float32[2, 16]", "bool" for scalars, or "(unresolved)"
        for the optional parameters not given by the caller.
    """

    # The optional parameters not reached by the resolution stay placeholders
    if repr_.dtype == PLACEHOLDER_DTYPE:
        return "(unresolved)"
    return format_repr(repr_=repr_)


def describe_value(value: Any) -> str:
    """
    Describe a runtime value with its shape and value statistics.

    Args:
        value (Any): The value to describe.

    Returns:
        str: The human readable description of the value.
    """

    # Tensors get the shape, dtype and value statistics
    if isinstance(value, Tensor):
        detached: Tensor = value.detach()

        # The boolean tensors have no abs implementation, and their
        # statistics are identical in float32
        if detached.dtype == torch.bool:
            detached = detached.float()
        return (
            f"{detached.dtype}{list(detached.shape)} "
            f"abs_sum={detached.abs().sum().item():.4f} "
            f"min={detached.min().item():.4f} max={detached.max().item():.4f}"
        )

    # Sequences are described item by item, briefly
    if isinstance(value, (tuple, list)):
        items: list[str] = [
            describe_value(item) if isinstance(item, (Tensor, tuple, list, dict))
            else type(item).__name__ for item in value
        ]
        return "(" + ", ".join(items) + ")"

    # Dictionaries from the model outputs
    if isinstance(value, dict):
        return "{" + ", ".join(
            f"{key}: {describe_value(item)}"
            for key, item in value.items()
            if isinstance(item, (Tensor, tuple, list, dict))
        ) + "}"

    # The common scalar values are described by their repr, briefly
    if value is None or isinstance(value, (bool, int, float, str)):
        return repr(value)[:60]

    # Any other python value is shown by type and repr, briefly
    return f"{type(value).__name__}({repr(value)[:60]})"


def describe_ref(value: VarRef | Any) -> str:
    """
    Describe an instruction argument, variable references by name.

    Args:
        value (VarRef | Any): The argument to describe.

    Returns:
        str: The name of the referenced variable, or a short repr.
    """

    # The name of the referenced variable
    if isinstance(value, VarRef):
        return value.name

    # The classes are described by their name (e.g. exception types)
    if isinstance(value, type):
        return value.__name__
    return repr(value)[:40]


def describe_repr(repr_: TensorRepr | None) -> str:
    """
    Describe the resolved representation of an instruction output.

    Args:
        repr_ (TensorRepr | None): The resolved representation, or None.

    Returns:
        str: The dtype, shape and value statistics of the representation.
    """

    # No representation resolved yet
    if repr_ is None:
        return "(unresolved)"

    # The captured value statistics of the output
    if isinstance(repr_, TensorReprWithValue):
        return "-> " + describe_value(repr_.value)

    # The plain shape and dtype of the output
    return "-> " + format_repr(repr_)


def _describe_args(
    args: list[VarRef | Any],
    kwargs: dict[str, VarRef | Any]
) -> str:
    """
    Describe the positional and keyword arguments of a call.

    Args:
        args (list[VarRef | Any]): The positional arguments.
        kwargs (dict[str, VarRef | Any]): The keyword arguments.

    Returns:
        str: The comma joined description of all the arguments.
    """

    # The positional then the keyword arguments
    positional: str = ", ".join(describe_ref(arg) for arg in args)
    keyword: str = ", ".join(
        f"{key}={describe_ref(value)}" for key, value in kwargs.items()
    )
    return ", ".join(part for part in (positional, keyword) if part)


def instruction_line(instruction: Instruction) -> str:
    """
    Build the one line description of an instruction.

    Args:
        instruction (Instruction): The instruction to describe.

    Returns:
        str: The human readable description of the instruction.
    """

    # The operation call
    if isinstance(instruction, OpInstruction):
        all_args: str = _describe_args(
            args=instruction.args, kwargs=instruction.kwargs
        )
        return f"{instruction.out_var} = {instruction.qualified_name}({all_args})"

    # The sub-block call
    if isinstance(instruction, BlockInstruction):
        all_args = _describe_args(args=instruction.args, kwargs=instruction.kwargs)
        splat: str = " *splat" if instruction.splat_args else ""
        index: str = ""
        if instruction.block_ref.index_var is not None:
            index = f"[{instruction.block_ref.index_var}]"
        return (
            f"{instruction.out_var} = CALL "
            f"{instruction.block_ref.path}{index}({all_args}){splat}"
        )

    # The return statement
    if isinstance(instruction, ReturnInstruction):
        if not instruction.out_vars:
            return "RETURN"
        return "RETURN " + ", ".join(instruction.out_vars)

    # The control flow is described by the recursive dumper
    if isinstance(instruction, ControlInstruction):

        # The for loops over a sub-block container or over an expression
        if instruction.kind == "for":
            if instruction.iter_container is not None:
                return (
                    f"for {instruction.bind_var} in "
                    f"{instruction.iter_container.path}"
                )
            return f"for {instruction.bind_var} in <expression>"
        return "if"

    # The try statement
    if isinstance(instruction, TryInstruction):
        return "TRY"

    # Any other instruction kind
    return f"{type(instruction).__name__}(out_var={instruction.out_var})"


def dump_instructions(
    body: list[Instruction],
    lines: list[str],
    indent: str
) -> None:
    """
    Recursively write the instructions of a body to the lines.

    Args:
        body (list[Instruction]): The instructions of the body.
        lines (list[str]): The output lines, updated in place.
        indent (str): The current indentation prefix.
    """

    # Write each instruction of the body
    instruction: Instruction
    for instruction in body:

        # Stop when the line budget is reached
        if len(lines) >= MAX_LINES:
            lines.append(indent + "... (truncated)")
            return

        # The operation, block call and return instructions
        line: str = instruction_line(instruction)
        if instruction.resolved is None:
            lines.append(f"{indent}{line}")
        else:
            lines.append(f"{indent}{line}  {describe_repr(instruction.resolved)}")

        # The control flow bodies are dumped recursively
        if isinstance(instruction, ControlInstruction):
            if instruction.condition is not None:
                label: str = "iterable" if instruction.kind == "for" else "condition"
                condition: str = instruction_line(instruction.condition)
                lines.append(f"{indent}  {label}: {condition}")
            dump_instructions(instruction.body, lines, indent + "    ")
            if instruction.else_body:
                lines.append(f"{indent}  else:")
                dump_instructions(instruction.else_body, lines, indent + "    ")

        # The try handlers and the finally body
        if isinstance(instruction, TryInstruction):
            types: tuple[type[BaseException], ...]
            for types, handler_body in instruction.handlers:
                names: str = "|".join(t.__name__ for t in types) or "?"
                lines.append(f"{indent}  except {names}:")
                dump_instructions(handler_body, lines, indent + "    ")
            if instruction.final_body:
                lines.append(f"{indent}  finally:")
                dump_instructions(instruction.final_body, lines, indent + "    ")


def _dump_section(
    lines: list[str],
    indent: str,
    title: str,
    entries: list[str]
) -> None:
    """
    Write one titled section of a block dump, one entry per line.

    Args:
        lines (list[str]): The output lines, updated in place.
        indent (str): The current indentation prefix.
        title (str): The section title (e.g. "weights").
        entries (list[str]): The one line descriptions of the section entries.
    """

    # The section title, then the entries or the empty marker
    lines.append(f"{indent}{title}:")
    if not entries:
        lines.append(f"{indent}  (none)")
        return
    for entry in entries:
        lines.append(f"{indent}  - {entry}")


def dump_block(
    block: ModelBlockSchema,
    lines: list[str],
    indent: str = ""
) -> None:
    """
    Recursively write a block schema to the lines.

    Args:
        block (ModelBlockSchema): The block schema to dump.
        lines (list[str]): The output lines, updated in place.
        indent (str): The current indentation prefix.
    """

    # Stop when the line budget is reached
    if len(lines) >= MAX_LINES:
        return

    # The header with the module class of the block
    module_type: str = type(block.module).__name__ \
        if block.module is not None else "?"
    lines.append(f"{indent}=== BLOCK {module_type} ===")

    # The declared inputs of the block
    _dump_section(
        lines=lines,
        indent=indent,
        title="inputs",
        entries=[
            f"{name} = {format_input(repr_)}"
            for name, repr_ in block.inputs.items()
        ]
    )

    # The declared outputs of the block
    _dump_section(
        lines=lines,
        indent=indent,
        title="outputs",
        entries=[
            format_input(repr_) for repr_ in block.outputs
        ]
    )

    # The direct parameters and buffers of the block
    _dump_section(
        lines=lines,
        indent=indent,
        title="weights",
        entries=[
            f"{name} = {format_repr(repr_)}"
            for name, repr_ in block.weights.items()
        ]
    )

    # The child blocks of the block, by attribute name
    _dump_section(
        lines=lines,
        indent=indent,
        title="sub-blocks",
        entries=[
            f"{name}({type(sub.module).__name__ if sub.module else '?'})"
            for name, sub in block.sub_blocks.items()
        ]
    )

    # The body instructions of the block
    lines.append(f"{indent}body:")
    dump_instructions(block.body, lines, indent + "  ")
    lines.append("")

    # The sub-blocks are dumped recursively
    sub_name: str
    sub_block: ModelBlockSchema
    for sub_name, sub_block in block.sub_blocks.items():
        dump_block(sub_block, lines, indent + "  ")


def dump_schema(schema: ModelBlockSchema, path: str) -> None:
    """
    Write the readable text dump of a block schema.

    Args:
        schema (ModelBlockSchema): The parsed schema to dump.
        path (str): The path of the output text file.
    """

    # The recursive dump of the whole schema tree
    lines: list[str] = []
    dump_block(schema, lines)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
