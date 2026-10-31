"""
Dependencies of the schema instructions and call sites of the methods.

The instruction_dependencies function resolves, for every instruction of a
block body, the variables it reads into a Dependency describing where the
value comes from: a declared input of the block, one of its weights, the
module itself, or the path of the last earlier instruction binding the
variable. The paths encode the nesting of the control flow bodies, so a
viewer can draw the dataflow edges between the exact instruction boxes.

The method_call_sites function mirrors sub_block_call_sites for the methods
of the module parsed as schemas: the top level body indices where each
parsed method is called on self.
"""

# Import Modules
from typing import Any

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


# The prefix of the self references in the qualified names of the calls
_SELF_PREFIX: str = "$self."


class Dependency:
    """
    One variable read by an instruction, resolved to its source.

    A dependency describes where the value of a read variable comes from:
    a declared input of the block ("input"), one of its weights or buffers
    ("weight"), the module itself ("self"), or a previous instruction of
    the body ("instruction", with the path of the last writer). A variable
    that matches none of these (a closure global, a value seeded by the
    interpreter defaults) is reported as "unresolved".
    """

    def __init__(self, name: str, kind: str, origin: str) -> None:
        """
        Init function.

        Args:
            name (str): The name of the referenced variable, as written in
                the forward source.
            kind (str): The kind of the source, one of "input", "weight",
                "self", "instruction" or "unresolved".
            origin (str): The name of the input or weight, "$self" for the
                module itself, or the path of the last instruction binding
                the variable (e.g. "3" at the top level or "2/body/1" in a
                control flow body). Empty when unresolved.
        """

        # Store here the name of the referenced variable
        self.name: str = name

        # Store here the kind of the dependency source
        self.kind: str = kind

        # Store here the origin of the dependency
        self.origin: str = origin

    def __repr__(self) -> str:
        """
        Get the readable representation of the dependency.

        Returns:
            str: E.g. "input:x" or "instruction:3/body/0".
        """

        # The kind and origin of the dependency
        return f"{self.kind}:{self.origin or self.name}"


def _resolve_dependency(
    name: str,
    block: ModelBlockSchema,
    writers: dict[str, str]
) -> Dependency:
    """
    Resolve the source of one read variable of a block body.

    Args:
        name (str): The name of the referenced variable.
        block (ModelBlockSchema): The block schema owning the body.
        writers (dict[str, str]): The paths of the last instructions
            binding each variable, filled during the walk.

    Returns:
        Dependency: The resolved dependency of the variable.
    """

    # A local binding of an earlier instruction shadows the declarations
    if name in writers:
        return Dependency(name=name, kind="instruction", origin=writers[name])

    # The declared inputs of the block
    if name in block.inputs:
        return Dependency(name=name, kind="input", origin=name)

    # The weights and buffers of the block
    if name in block.weights:
        return Dependency(name=name, kind="weight", origin=name)

    # The module itself, readable in the body through the self reference
    if name == "$self":
        return Dependency(name=name, kind="self", origin="$self")

    # The closure globals and the interpreter seeded values
    return Dependency(name=name, kind="unresolved", origin="")


def _instruction_reads(instruction: Instruction) -> list[str]:
    """
    Get the names of the variables read by an instruction.

    Args:
        instruction (Instruction): The instruction to inspect.

    Returns:
        list[str]: The referenced variable names, in reading order.
    """

    # The names read by the instruction, in argument order
    reads: list[str] = []

    def _collect(value: VarRef | Any) -> None:
        """Collect one argument when it references a variable."""
        if isinstance(value, VarRef):
            reads.append(value.name)

    # The operation instructions read their target callable and arguments
    if isinstance(instruction, OpInstruction):
        if isinstance(instruction.target, VarRef):
            reads.append(instruction.target.name)
        for argument in instruction.args:
            _collect(value=argument)
        for keyword_value in instruction.kwargs.values():
            _collect(value=keyword_value)

    # The block calls read their arguments and unpacked variables
    elif isinstance(instruction, BlockInstruction):
        for argument in instruction.args:
            _collect(value=argument)
        for keyword_value in instruction.kwargs.values():
            _collect(value=keyword_value)

    # The return instructions read their returned variables
    elif isinstance(instruction, ReturnInstruction):
        reads.extend(instruction.out_vars)

    # The unpacked star arguments of the calls
    if isinstance(instruction, (OpInstruction, BlockInstruction)):
        if instruction.args_var is not None:
            reads.append(instruction.args_var)
        if instruction.kwargs_var is not None:
            reads.append(instruction.kwargs_var)

    # The read variables of the instruction
    return reads


def _register_writer(
    writers: dict[str, str],
    path: str,
    names: str | list[str] | None
) -> None:
    """
    Register the variables bound at an instruction path.

    Args:
        writers (dict[str, str]): The last writer paths by variable name.
        path (str): The path of the binding instruction.
        names (str | list[str] | None): The bound variable name or names.
    """

    # No variable is bound when the names are missing
    if names is None:
        return

    # Register each bound variable name
    if isinstance(names, str):
        writers[names] = path
    else:
        for name in names:
            writers[name] = path


def _walk_instructions(
    instructions: list[Instruction],
    prefix: str,
    block: ModelBlockSchema,
    writers: dict[str, str],
    dependencies: dict[str, list[Dependency]]
) -> None:
    """
    Walk a list of instructions, collecting the dependencies of each one.

    Args:
        instructions (list[Instruction]): The instructions to walk, in
            execution order.
        prefix (str): The path prefix of the walked body ("" at the top
            level).
        block (ModelBlockSchema): The block schema owning the instructions.
        writers (dict[str, str]): The last writer paths by variable name.
        dependencies (dict[str, list[Dependency]]): The collected
            dependencies by instruction path.
    """

    # Walk the instructions of the body in execution order
    index: int
    instruction: Instruction
    for index, instruction in enumerate(instructions):

        # The path of the instruction in the body
        path: str = f"{prefix}/{index}" if prefix else str(index)

        # The control flow instructions publish their condition first,
        # unless the lowering already placed it in the walked body itself
        if isinstance(instruction, ControlInstruction) \
                and instruction.condition is not None \
                and instruction.condition not in instructions:
            condition_path: str = f"{path}/condition"
            condition_reads: list[str] = _instruction_reads(
                instruction=instruction.condition
            )
            dependencies[condition_path] = [
                _resolve_dependency(
                    name=name, block=block, writers=writers
                )
                for name in condition_reads
            ]
            _register_writer(
                writers=writers,
                path=condition_path,
                names=instruction.condition.out_var
            )

        # The dependencies of the instruction itself
        reads: list[str] = _instruction_reads(instruction=instruction)
        dependencies[path] = [
            _resolve_dependency(name=name, block=block, writers=writers)
            for name in reads
        ]

        # The loop variables of a control flow are bound by the loop
        if isinstance(instruction, ControlInstruction):
            _register_writer(
                writers=writers,
                path=path,
                names=instruction.bind_var
            )
            _register_writer(
                writers=writers,
                path=path,
                names=instruction.index_var
            )

        # The instruction binds its output variable when it has one
        _register_writer(
            writers=writers,
            path=path,
            names=instruction.out_var or None
        )

        # The nested bodies of the control flow instructions
        if isinstance(instruction, ControlInstruction):
            _walk_instructions(
                instructions=instruction.body,
                prefix=f"{path}/body",
                block=block,
                writers=writers,
                dependencies=dependencies
            )
            _walk_instructions(
                instructions=instruction.else_body,
                prefix=f"{path}/else",
                block=block,
                writers=writers,
                dependencies=dependencies
            )

        # The nested bodies of the try instructions
        elif isinstance(instruction, TryInstruction):
            _walk_instructions(
                instructions=instruction.body,
                prefix=f"{path}/body",
                block=block,
                writers=writers,
                dependencies=dependencies
            )
            handler_index: int
            for handler_index, (_, handler_body) in enumerate(
                instruction.handlers
            ):
                _walk_instructions(
                    instructions=handler_body,
                    prefix=f"{path}/handlers/{handler_index}",
                    block=block,
                    writers=writers,
                    dependencies=dependencies
                )
                _register_writer(
                    writers=writers,
                    path=f"{path}/handlers/{handler_index}",
                    names=instruction.handler_names[handler_index]
                )
            _walk_instructions(
                instructions=instruction.final_body,
                prefix=f"{path}/finally",
                block=block,
                writers=writers,
                dependencies=dependencies
            )


def instruction_dependencies(
    block: ModelBlockSchema
) -> dict[str, list[Dependency]]:
    """
    Get the dependencies of every instruction of a block body.

    The returned paths encode the nesting of the instructions: the top
    level instructions are keyed by their body index ("0", "1", ...), the
    instructions of a control flow or try body by "{parent}/body/{index}",
    "{parent}/else/{index}", "{parent}/handlers/{handler}/{index}" and
    "{parent}/finally/{index}", and the condition expression of a control
    flow by "{parent}/condition". The variables are resolved in execution
    order, so a dependency on a rebound variable points to the last
    earlier instruction binding it.

    Args:
        block (ModelBlockSchema): The block schema to inspect.

    Returns:
        dict[str, list[Dependency]]: The read dependencies of each
        instruction holding reads, by instruction path.
    """

    # The last writer of each variable, filled during the walk
    writers: dict[str, str] = {}

    # The collected dependencies of the instructions
    dependencies: dict[str, list[Dependency]] = {}
    _walk_instructions(
        instructions=block.body,
        prefix="",
        block=block,
        writers=writers,
        dependencies=dependencies
    )
    return dependencies


def method_call_sites(block: ModelBlockSchema) -> dict[str, list[int]]:
    """
    Get the body indices where each parsed method of the block is called.

    The keys are the names of the methods of the module parsed as schemas
    (the block.methods keys), and the values are the sorted top level body
    indices of their calls on self. A call nested in a control flow or try
    body is reported at the index of its outermost enclosing instruction,
    like the sub-block call sites. A self method call with no parsed
    schema is not reported.

    Args:
        block (ModelBlockSchema): The block schema to inspect.

    Returns:
        dict[str, list[int]]: The call site indices of each parsed method,
        in body order.
    """

    # One entry per parsed method name, filled by the recursive walk
    sites: dict[str, list[int]] = {}
    _collect_method_sites(
        instructions=block.body,
        block=block,
        top_index=None,
        sites=sites
    )
    return sites


def _collect_method_sites(
    instructions: list[Instruction],
    block: ModelBlockSchema,
    top_index: int | None,
    sites: dict[str, list[int]]
) -> None:
    """
    Collect the parsed method call sites of a list of instructions.

    Args:
        instructions (list[Instruction]): The instructions to walk.
        block (ModelBlockSchema): The block schema owning the instructions.
        top_index (int | None): The top level body index reported for the
            call sites found in the instructions.
        sites (dict[str, list[int]]): The call sites to fill, by method
            name.
    """

    # Walk the instructions, recursing into the nested bodies
    index: int
    instruction: Instruction
    for index, instruction in enumerate(instructions):

        # The top level index of the nested call sites
        enclosing: int = index if top_index is None else top_index

        # A self method call is a call operation qualified by "$self.name"
        if isinstance(instruction, OpInstruction) \
                and instruction.qualified_name.startswith(_SELF_PREFIX):
            method_name: str = instruction.qualified_name[len(_SELF_PREFIX):]

            # Only the methods with a parsed schema are reported
            if method_name in block.methods:
                sites.setdefault(method_name, []).append(enclosing)

        # The control flow bodies nest their own call sites
        if isinstance(instruction, ControlInstruction):
            _collect_method_sites(
                instructions=instruction.body,
                block=block,
                top_index=enclosing,
                sites=sites
            )
            _collect_method_sites(
                instructions=instruction.else_body,
                block=block,
                top_index=enclosing,
                sites=sites
            )

        # The try bodies nest their own call sites
        elif isinstance(instruction, TryInstruction):
            _collect_method_sites(
                instructions=instruction.body,
                block=block,
                top_index=enclosing,
                sites=sites
            )
            handler_body: list[Instruction]
            for _, handler_body in instruction.handlers:
                _collect_method_sites(
                    instructions=handler_body,
                    block=block,
                    top_index=enclosing,
                    sites=sites
                )
            _collect_method_sites(
                instructions=instruction.final_body,
                block=block,
                top_index=enclosing,
                sites=sites
            )
