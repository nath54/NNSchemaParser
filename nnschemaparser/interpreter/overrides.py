"""
Overrides of the values produced during a schema execution.

An Override forces the replacement of one value of a run: the tensor bound
by a precise instruction of a block (an activation, a condition, a sub-block
call result) or a weight of a block seeded in its scope. The replacement is
applied at the binding point of the interpreter, before the listeners and
the resolved representations observe the value, so the traces of the run
carry the overridden tensors and the diffs against a reference run show
exactly how the edit propagates.

The typical entry point is the overrides parameter of capture_run:

    capture_run(
        block=schema,
        scope={"x": x},
        overrides=[
            Override(block_path="encoder.layers.0", instruction="2",
                     values=torch.zeros(2, 10)),
            Override(block_path="encoder.layers.1", out_var="hidden",
                     op_name="torch.nn.functional.linear", occurrence=2,
                     patch=lambda t: torch.nan_to_num(t)),
            Override(block_path="fc1", weight="weight",
                     values=torch.randn_like(schema.weights["weight"].value)),
        ],
    )
"""

# Import Modules
from typing import Any, Callable

from torch import Tensor

import torch

from nnschemaparser.core.instructions import (
    BlockInstruction,
    ControlInstruction,
    Instruction,
    OpInstruction,
    TryInstruction
)
from nnschemaparser.core.model_schema import ModelBlockSchema
from nnschemaparser.core.traces import TensorTrace
from nnschemaparser.interpreter.stepping import step_op_name


class Override:
    """
    One forced replacement of a value during a schema execution.

    The target of the override is either:

    - a precise instruction of a block, by its nested instruction path (the
      "instruction" field, e.g. "3" at the top level of the body or
      "5/body/0" in a nested body, aligned with the dependency paths);
    - the instructions of a block binding a given variable, optionally
      filtered by the display name of the operation (the descriptor mode:
      "out_var" and "op_name");
    - a weight of a block, by name (the "weight" field), replaced when the
      block scope is seeded at its entry.

    The replacement is either raw values converted to a tensor ("values",
    with the optional "shape" and "dtype"), or a callable receiving the
    produced value and returning the replacement ("patch"), e.g.
    "lambda t: torch.zeros_like(t)" for an ablation.

    The "occurrence" field restricts the application to the n-th execution
    of the target (1-based, an int or a list of ints, None for every
    execution), which is how one iteration of a loop body is targeted.
    """

    def __init__(
        self,
        block_path: str = "",
        instruction: str | None = None,
        weight: str | None = None,
        out_var: str | None = None,
        op_name: str | None = None,
        values: Any | None = None,
        patch: Callable[[Any], Any] | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: torch.dtype | None = None,
        occurrence: int | list[int] | None = None
    ) -> None:
        """
        Init function.

        Args:
            block_path (str): The full attribute path of the block owning
                the target, from the root block ("" for the root itself,
                "encoder.layers.0" for a container entry).
            instruction (str | None): The nested path of the targeted
                instruction in the block body (e.g. "3", "5/body/0",
                "5/condition", "5/handlers/0/1"), or None for the
                descriptor or weight modes.
            weight (str | None): The name of the targeted weight of the
                block, or None when targeting an instruction binding.
            out_var (str | None): The descriptor mode: the name of the
                variable the targeted instructions bind.
            op_name (str | None): The descriptor mode: the display name of
                the targeted instructions (the qualified name of the torch
                callable, or the block reference path of a sub-block call).
            values (Any | None): The raw replacement values, any tensor-like
                accepted by torch.as_tensor.
            patch (Callable[[Any], Any] | None): The callable receiving the
                produced value and returning the replacement.
            shape (tuple[int, ...] | None): The shape of the replacement
                tensor, defaulting to the shape of the replaced value.
            dtype (torch.dtype | None): The dtype of the replacement tensor,
                defaulting to the dtype of the replaced value.
            occurrence (int | list[int] | None): The 1-based execution
                numbers of the target the override applies to, None for
                every execution.

        Raises:
            ValueError: When the target or the replacement is inconsistent.
        """

        # Normalize the empty string filters to None descriptors
        out_var: str | None = out_var or None
        op_name: str | None = op_name or None

        # Exactly one of the raw values and the patch callable is required
        if (values is None) == (patch is None):
            raise ValueError(
                "An override requires exactly one of the values and patch "
                "arguments."
            )

        # Exactly one target mode is required
        targeted: list[str] = [
            name
            for name, value in (
                ("instruction", instruction),
                ("weight", weight),
                ("out_var / op_name", out_var if out_var or op_name else None)
            )
            if value is not None
        ]
        if len(targeted) != 1:
            raise ValueError(
                "An override requires exactly one target: an instruction "
                "path, a weight name, or an out_var / op_name descriptor."
            )

        # Store here the target of the override
        self.block_path: str = block_path
        self.instruction: str | None = instruction
        self.weight: str | None = weight
        self.out_var: str | None = out_var
        self.op_name: str | None = op_name

        # Store here the replacement of the override
        self.values: Any | None = values
        self.patch: Callable[[Any], Any] | None = patch
        self.shape: tuple[int, ...] | None = shape
        self.dtype: torch.dtype | None = dtype
        self.occurrence: int | list[int] | None = occurrence

    @classmethod
    def from_container_id(
        cls,
        trace: TensorTrace,
        container_id: str,
        values: Any | None = None,
        patch: Callable[[Any], Any] | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: torch.dtype | None = None
    ) -> "Override":
        """
        Build an override targeting the instruction that produced a captured
        tensor of a previous run.

        The block path, the bound variable and the occurrence of the target
        are resolved from the trace of the reference run, so overriding the
        container of a stored run and re-running the schema reproduces the
        same control flow. The container must be an activation bound to a
        named variable: the inputs are overridden through the run scope and
        the outputs through the return values.

        Args:
            trace (TensorTrace): The trace of the reference run.
            container_id (str): The container id of the targeted tensor.
            values (Any | None): The raw replacement values.
            patch (Callable[[Any], Any] | None): The replacement callable.
            shape (tuple[int, ...] | None): The shape of the replacement.
            dtype (torch.dtype | None): The dtype of the replacement.

        Returns:
            Override: The override targeting the producing instruction.

        Raises:
            ValueError: When the container id is unknown, or the container
                is not a named activation of the run.
        """

        # The step of the requested container id
        index: int
        step: Any
        for index, step in enumerate(trace.steps):
            if step.container_id == container_id:
                break
        else:
            raise ValueError(
                f"Unknown container id '{container_id}' in the given trace."
            )

        # Only the activations of a run are instruction bindings
        if step.slot_type != "activation":
            raise ValueError(
                f"The container '{container_id}' is a '{step.slot_type}' "
                "step: only activation containers can be overridden."
            )
        if not step.out_var:
            raise ValueError(
                f"The container '{container_id}' binds no variable name."
            )

        # The occurrence of the binding among the activations of the run,
        # matching the block path, the variable and the operation
        occurrence: int = 1 + sum(
            1
            for earlier in trace.steps[:index]
            if earlier.slot_type == "activation"
            and earlier.block_path == step.block_path
            and earlier.out_var == step.out_var
            and earlier.op_name == step.op_name
        )

        # The descriptor mode override of the producing instruction
        return cls(
            block_path=step.block_path,
            out_var=step.out_var,
            op_name=step.op_name,
            occurrence=occurrence,
            values=values,
            patch=patch,
            shape=shape,
            dtype=dtype
        )

    def __repr__(self) -> str:
        """
        Get the developer representation of the override.

        Returns:
            str: E.g. "Override(block_path='encoder.layers.0',
            instruction='2', values=zeros)".
        """

        # The target part of the representation
        if self.instruction is not None:
            target: str = f"instruction={self.instruction!r}"
        elif self.weight is not None:
            target = f"weight={self.weight!r}"
        else:
            target = f"out_var={self.out_var!r}, op_name={self.op_name!r}"

        # The replacement part of the representation
        replacement: str = "patch(...)" if self.patch is not None \
            else f"values={self.values!r}"

        # The assembled representation
        return (
            f"Override(block_path={self.block_path!r}, {target}, "
            f"{replacement})"
        )

    def __str__(self) -> str:
        """
        Get the readable representation of the override.

        Returns:
            str: E.g. "override encoder.layers.0 instruction 2 (every
            execution)".
        """

        # The target part of the description
        if self.instruction is not None:
            target: str = f"instruction {self.instruction}"
        elif self.weight is not None:
            target = f"weight {self.weight}"
        else:
            descriptor: str = self.op_name or "*"
            target = f"{self.out_var or '*'} = {descriptor}"

        # The occurrence part of the description
        if self.occurrence is None:
            occurrence_desc: str = "every execution"
        elif isinstance(self.occurrence, int):
            occurrence_desc = f"execution {self.occurrence}"
        else:
            occurrence_desc = f"executions {self.occurrence}"

        # The assembled description
        return (
            f"override {self.block_path or '<root>'} {target} "
            f"({occurrence_desc})"
        )


class OverrideApplier:
    """
    Applier of the overrides of an interpreter run.

    The applier is created by the interpreter with the overrides of the run,
    reset at the top level of every run (resolving the targets against the
    root block and zeroing the occurrence counters), and consulted at every
    instruction binding and block scope seeding. The applier without
    overrides is a no-op with a negligible overhead.
    """

    def __init__(self, overrides: list[Override] | None = None) -> None:
        """
        Init function.

        Args:
            overrides (list[Override] | None): The overrides of the runs.
        """

        # Store here the overrides to apply
        self.overrides: list[Override] = list(overrides or [])

        # The resolved instruction targets, by override identity
        self._instructions: dict[int, Instruction] = {}

        # The resolved owning blocks of the descriptor and weight targets,
        # by override identity
        self._blocks: dict[int, ModelBlockSchema] = {}

        # The execution counters of the targets, by override identity
        self._hits: dict[int, int] = {}

    def reset(self, root_block: ModelBlockSchema) -> None:
        """
        Reset the applier for a new top level run: zero the occurrence
        counters and resolve the targets against the root block.

        Args:
            root_block (ModelBlockSchema): The root block of the run.

        Raises:
            ValueError: When a target does not resolve in the block tree.
        """

        # Without overrides there is nothing to prepare
        if not self.overrides:
            return

        # The zeroed counters and resolved targets of the new run
        self._instructions = {}
        self._blocks = {}
        self._hits = {id(override): 0 for override in self.overrides}

        # Resolve each override against the block tree
        override: Override
        for override in self.overrides:

            # The owning block of the target
            block: ModelBlockSchema | None = _resolve_block_schema(
                root=root_block,
                block_path=override.block_path
            )
            if block is None:
                raise ValueError(
                    f"Unknown block path '{override.block_path}' in the "
                    f"override: {override}"
                )

            # The exact instruction targets are resolved once per run, and
            # must bind a variable to replace
            if override.instruction is not None:
                instruction: Instruction | None = _resolve_instruction_path(
                    block=block,
                    path=override.instruction
                )
                if instruction is None:
                    raise ValueError(
                        f"Unknown instruction path '{override.instruction}' "
                        f"in block '{override.block_path}': {override}"
                    )
                if not instruction.out_var or not isinstance(
                    instruction,
                    (OpInstruction, BlockInstruction)
                ):
                    raise ValueError(
                        f"The instruction '{override.instruction}' of block "
                        f"'{override.block_path}' binds no variable: {override}"
                    )
                self._instructions[id(override)] = instruction
                continue

            # The weight targets are checked against the declared weights
            if override.weight is not None:
                if override.weight not in block.weights:
                    raise ValueError(
                        f"Unknown weight '{override.weight}' in block "
                        f"'{override.block_path}': {override}"
                    )
                self._blocks[id(override)] = block
                continue

            # The descriptor targets require at least one filter
            if not override.out_var and not override.op_name:
                raise ValueError(
                    f"A descriptor override requires an out_var or an "
                    f"op_name filter: {override}"
                )
            self._blocks[id(override)] = block

    def apply_binding(
        self,
        instruction: Instruction,
        block: ModelBlockSchema,
        value: Any
    ) -> Any:
        """
        Apply the overrides targeting the binding of an instruction.

        Args:
            instruction (Instruction): The instruction that produced the
                value.
            block (ModelBlockSchema): The block executing the instruction.
            value (Any): The produced value.

        Returns:
            Any: The replacement of the first matching override allowed by
                its occurrence, or the produced value unchanged.
        """

        # Without overrides the produced value is returned unchanged
        if not self.overrides:
            return value

        # Check each override of the run against the binding
        override: Override
        for override in self.overrides:

            # The weight targets are applied at the block seeding instead
            if override.weight is not None:
                continue

            # The target of the override must match this binding
            if not self._matches_binding(
                override=override,
                instruction=instruction,
                block=block
            ):
                continue

            # The occurrence counter of the target
            count: int = self._hits[id(override)] + 1
            self._hits[id(override)] = count

            # The occurrence of the binding must be targeted
            if not _occurrence_allowed(
                occurrence=override.occurrence,
                count=count
            ):
                continue

            # The first matching and allowed override wins
            return _replacement_of(override=override, reference=value)

        # No override applied, the produced value is returned unchanged
        return value

    def apply_weights(
        self,
        block: ModelBlockSchema,
        scope: dict[str, Any]
    ) -> None:
        """
        Apply the weight overrides targeting the seeding of a block scope.

        Args:
            block (ModelBlockSchema): The block whose scope was seeded.
            scope (dict[str, Any]): The seeded scope, modified in place.
        """

        # Without overrides the scope is left unchanged
        if not self.overrides:
            return

        # Check each weight override of the run against the block
        override: Override
        for override in self.overrides:

            # Only the weight targets are applied here
            if override.weight is None:
                continue

            # The block of the override must be the seeded one
            if self._blocks.get(id(override)) is not block:
                continue

            # The occurrence counter of the target
            count: int = self._hits[id(override)] + 1
            self._hits[id(override)] = count

            # The occurrence of the seeding must be targeted
            if not _occurrence_allowed(
                occurrence=override.occurrence,
                count=count
            ):
                continue

            # The replaced weight tensor, bound in the block scope
            scope[override.weight] = _replacement_of(
                override=override,
                reference=scope.get(override.weight)
            )

    def _matches_binding(
        self,
        override: Override,
        instruction: Instruction,
        block: ModelBlockSchema
    ) -> bool:
        """
        Check whether an override targets an instruction binding.

        Args:
            override (Override): The override to check.
            instruction (Instruction): The executed instruction.
            block (ModelBlockSchema): The block executing the instruction.

        Returns:
            bool: True when the binding matches the target of the override.
        """

        # The exact instruction targets match by object identity
        if override.instruction is not None:
            return self._instructions.get(id(override)) is instruction

        # The descriptor targets match the block and the binding filters
        if self._blocks.get(id(override)) is not block:
            return False
        if override.out_var is not None \
                and instruction.out_var != override.out_var:
            return False
        if override.op_name is not None \
                and step_op_name(instruction=instruction) != override.op_name:
            return False
        return True


def _resolve_block_schema(
    root: ModelBlockSchema,
    block_path: str
) -> ModelBlockSchema | None:
    """
    Resolve a block schema from the full attribute path from the root.

    The path segments are matched against the sub-block and method keys,
    which may contain dots for the container entries ("inners.0"), so the
    longest key match is preferred at each level.

    Args:
        root (ModelBlockSchema): The root block of the block tree.
        block_path (str): The full path of the targeted block, "" for the
            root itself.

    Returns:
        ModelBlockSchema | None: The resolved block, or None when the path
            matches no block of the tree.
    """

    # The root block is addressed by its empty path
    if not block_path:
        return root

    # The remaining segments of the path to match
    remaining: list[str] = block_path.split(".")

    # The current block of the walk
    block: ModelBlockSchema = root
    while remaining:

        # The longest matching child key at this level, the keys may hold
        # the dot separated index of a container entry
        matched: str | None = None
        for key in sorted(
            list(block.sub_blocks) + list(block.methods),
            key=lambda key: -key.count(".")
        ):
            parts: list[str] = key.split(".")
            if remaining[:len(parts)] == parts:
                matched = key
                break

        # The unmatched path is an unknown block
        if matched is None:
            return None

        # Descend into the matched child with the consumed segments
        block = block.sub_blocks.get(matched) or block.methods[matched]
        remaining = remaining[len(matched.split(".")):]

    # The block of the fully consumed path
    return block


def _resolve_instruction_path(
    block: ModelBlockSchema,
    path: str
) -> Instruction | None:
    """
    Resolve an instruction of a block from its nested path.

    The paths are aligned with the dependency paths: the top level
    instructions are keyed by their body index ("0", "1", ...), the
    instructions of the nested bodies by "{parent}/body/{index}",
    "{parent}/else/{index}", "{parent}/handlers/{handler}/{index}" and
    "{parent}/finally/{index}", and the condition expression of a control
    flow by "{parent}/condition".

    Args:
        block (ModelBlockSchema): The block owning the instruction.
        path (str): The nested path of the instruction.

    Returns:
        Instruction | None: The resolved instruction, or None when the path
            matches no instruction of the block.
    """

    # The first segment is the index in the body of the block
    parts: list[str] = path.split("/")
    try:
        instruction: Instruction = block.body[int(parts[0])]
    except (ValueError, IndexError):
        return None

    # Walk the nested segments of the path
    index: int = 1
    while index < len(parts):

        # The condition expression of a control flow instruction
        if parts[index] == "condition":
            if not isinstance(instruction, ControlInstruction) \
                    or instruction.condition is None:
                return None
            instruction = instruction.condition
            index += 1
            continue

        # The handler segments carry the handler index before the body one
        if parts[index] == "handlers":
            try:
                handler: tuple[tuple[type[BaseException], ...],
                               list[Instruction]]
                handler = instruction.handlers[int(parts[index + 1])]  # type: ignore[attr-defined]
                instruction = handler[1][int(parts[index + 2])]
            except (ValueError, IndexError, AttributeError):
                return None
            index += 3
            continue

        # The body segments of the control flow and try instructions
        if parts[index] in ("body", "else", "finally"):
            try:
                body_index: int = int(parts[index + 1])
                if parts[index] == "body":
                    body: list[Instruction] = instruction.body  # type: ignore[attr-defined]
                elif parts[index] == "else":
                    body = instruction.else_body  # type: ignore[attr-defined]
                else:
                    body = instruction.final_body  # type: ignore[attr-defined]
                instruction = body[body_index]
            except (ValueError, IndexError, AttributeError):
                return None
            index += 2
            continue

        # The unknown segments never match
        return None

    # The instruction of the fully consumed path
    return instruction


def _occurrence_allowed(
    occurrence: int | list[int] | None,
    count: int
) -> bool:
    """
    Check whether an execution count is targeted by an override.

    Args:
        occurrence (int | list[int] | None): The targeted 1-based execution
            numbers, None for every execution.
        count (int): The 1-based execution count of the target.

    Returns:
        bool: True when the execution count is targeted.
    """

    # Every execution is targeted without an occurrence filter
    if occurrence is None:
        return True

    # The single or the listed execution numbers
    if isinstance(occurrence, int):
        return count == occurrence
    return count in occurrence


def _replacement_of(override: Override, reference: Any) -> Any:
    """
    Compute the replacement value of an override.

    Args:
        override (Override): The matched override.
        reference (Any): The replaced value, used for the default shape and
            dtype and received by the patch callables.

    Returns:
        Any: The replacement value.
    """

    # The patch callables produce the replacement themselves
    if override.patch is not None:
        return override.patch(reference)

    # The raw values are converted to a tensor
    return _build_replacement(
        values=override.values,
        shape=override.shape,
        dtype=override.dtype,
        reference=reference
    )


def _build_replacement(
    values: Any,
    shape: tuple[int, ...] | None,
    dtype: torch.dtype | None,
    reference: Any
) -> Any:
    """
    Build the replacement tensor of raw values.

    The dtype defaults to the one of the replaced tensor, and the shape to
    the one of the replaced tensor: raw values of the same element count are
    reshaped to it, and a single value fills the whole replaced tensor.

    Args:
        values (Any): The raw values, any tensor-like accepted by
            torch.as_tensor.
        shape (tuple[int, ...] | None): The explicit shape of the
            replacement, overriding the defaults.
        dtype (torch.dtype | None): The explicit dtype of the replacement,
            overriding the default.
        reference (Any): The replaced value.

    Returns:
        Any: The replacement tensor.

    Raises:
        ValueError: When the raw values cannot be fitted into the shape of
            the replaced tensor and no explicit shape is given.
    """

    # The non tensor references have no shape and dtype to inherit
    if not isinstance(reference, Tensor):
        tensor: Tensor = torch.as_tensor(values, dtype=dtype)
        if shape is not None:
            tensor = tensor.reshape(shape)
        return tensor

    # The dtype of the replaced tensor is the default one
    target_dtype: torch.dtype = dtype if dtype is not None \
        else reference.dtype
    tensor = torch.as_tensor(values, dtype=target_dtype)

    # The explicit shape wins over the shape of the replaced tensor
    if shape is not None:
        return tensor.reshape(shape)

    # The raw values of the same element count are reshaped to the tensor
    if tensor.numel() == reference.numel():
        return tensor.reshape(tuple(reference.shape))

    # A single value fills the whole replaced tensor
    if tensor.numel() == 1:
        return tensor.expand(tuple(reference.shape)).contiguous()

    # The element count mismatch is an error of the override
    raise ValueError(
        f"Cannot fit {tensor.numel()} override values into the shape "
        f"{tuple(reference.shape)} of the replaced tensor: pass an "
        "explicit shape or a matching element count."
    )
