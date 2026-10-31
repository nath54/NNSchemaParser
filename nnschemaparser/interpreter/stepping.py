"""
Stepped execution of a schema: the StepInfo yielded by step_run.

A StepInfo describes one executed instruction of an interpreter run: its
chronological step index, its kind, the scope variable it binds, the live
scope frames of the execution and the produced value. The edit method
patches a scope frame before the next instruction, which is the whole
mid-run tensor editing mechanism.
"""

# Import Modules
from typing import Any, Callable

import threading

import torch
from torch import Tensor

from nnschemaparser.core.instructions import (
    BlockInstruction,
    BreakInstruction,
    ContinueInstruction,
    ControlInstruction,
    Instruction,
    OpInstruction,
    ReturnInstruction,
    TryInstruction
)
from nnschemaparser.core.model_schema import ModelBlockSchema
from nnschemaparser.core.tensor_schema import TensorRepr


class _StepAborted(BaseException):
    """
    Signal unwinding the interpreter when a stepped run is abandoned.

    Derives from BaseException like the loop control signals, so the
    interpreted forwards never catch it while their finally bodies still
    run like in python.
    """


class _Stepper:
    """
    Internal gate synchronizing the step_run worker thread and consumer.

    The worker publishes a step then waits; the consumer releases the gate
    with the edits queued on the yielded StepInfo. Closing the stepper makes
    the next wait raise _StepAborted, unwinding the interpreter.
    """

    def __init__(self) -> None:
        """
        Init function.
        """

        # The condition guarding the advance flag, the close flag and the
        # pending edits of the gate
        self._condition: threading.Condition = threading.Condition()
        self._advance: bool = False
        self._closed: bool = False
        self._pending_edits: list[tuple[int, str, Tensor]] = []

    def wait(self) -> list[tuple[int, str, Tensor]]:
        """
        Wait for the consumer to release the gate, from the worker thread.

        Returns:
            list[tuple[int, str, Tensor]]: The edits to apply to the scope
            frames before the next instruction, as (frame depth, variable
            name, tensor) triples.

        Raises:
            _StepAborted: When the stepped run was closed by the consumer.
        """

        # Block until an advance or a close is requested
        with self._condition:
            while not self._advance and not self._closed:
                self._condition.wait()
            self._advance = False
            edits: list[tuple[int, str, Tensor]] = self._pending_edits
            self._pending_edits = []

            # The closed flag unwinds the interpreter
            if self._closed:
                raise _StepAborted()

            # The edits queued before the release
            return edits

    def queue_edit(self, frame_depth: int, out_var: str, tensor: Tensor) -> None:
        """
        Queue a scope edit to apply before the next instruction.

        Args:
            frame_depth (int): The depth of the scope frame to patch.
            out_var (str): The name of the variable to patch.
            tensor (Tensor): The new value of the variable.
        """

        # The queue is guarded by the gate condition
        with self._condition:
            self._pending_edits.append((frame_depth, out_var, tensor))

    def release(self) -> None:
        """
        Let the worker execute the next instruction, from the consumer.
        """

        # The advance flag wakes the waiting worker
        with self._condition:
            self._advance = True
            self._condition.notify_all()

    def close(self) -> None:
        """
        Close the stepped run, unwinding the worker at its next step.
        """

        # The close flag makes the next wait raise the abort signal
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class StepInfo:
    """
    One executed instruction of a stepped schema run.

    A step info is yielded by SchemaInterpreter.step_run after each executed
    instruction (control flow instructions before their body), or published
    to a raw step hook. It carries the original variable names, the live
    scope frames of the execution and the produced value, and its edit
    method patches a scope frame before the next instruction.
    """

    def __init__(
        self,
        step_index: int,
        kind: str,
        out_var: str,
        op_name: str,
        block: ModelBlockSchema,
        scope: dict[str, Any],
        scope_frames: list[dict[str, Any]],
        frame_depth: int,
        value: Any
    ) -> None:
        """
        Init function.

        Args:
            step_index (int): The 1-based chronological index of the step,
                deterministic for a schema, input and control path.
            kind (str): The kind of the instruction, one of "op",
                "block_call", "if", "for", "while", "try", "return",
                "continue" or "break".
            out_var (str): The original variable name the result is bound
                to, "" when the instruction binds nothing.
            op_name (str): The display name of the instruction, the
                qualified name of the torch callable for an operation or
                the block reference path for a sub-block call.
            block (ModelBlockSchema): The block schema owning the
                instruction.
            scope (dict[str, Any]): The scope of the block executing the
                instruction (the innermost frame).
            scope_frames (list[dict[str, Any]]): The live stack of the
                executed block scopes, the innermost frame last. The root
                frame has the depth 0.
            frame_depth (int): The index of the innermost frame in the
                scope frames stack.
            value (Any): The value bound to the variable of the step, when
                the instruction binds one.
        """

        # Store here the chronological index of the step
        self.step_index: int = step_index

        # Store here the kind of the executed instruction
        self.kind: str = kind

        # Store here the original variable name of the result
        self.out_var: str = out_var

        # Store here the display name of the instruction
        self.op_name: str = op_name

        # Store here the block schema owning the instruction
        self.block: ModelBlockSchema = block

        # Store here the innermost scope of the execution
        self.scope: dict[str, Any] = scope

        # Store here the live scope frames of the execution
        self.scope_frames: list[dict[str, Any]] = scope_frames

        # Store here the depth of the innermost scope frame
        self.frame_depth: int = frame_depth

        # Store here the value bound by the instruction
        self.value: Any = value

        # The optional stepper routing the edits of the generator mode
        self._stepper: _Stepper | None = None

    def _bind_stepper(self, stepper: _Stepper) -> None:
        """
        Bind the stepper of a generator mode run to this step.

        Args:
            stepper (_Stepper): The stepper owning the executed run.
        """

        # The edits are queued on the stepper and applied by the worker
        self._stepper = stepper

    @property
    def default_shape(self) -> tuple[int, ...]:
        """
        Get the shape of the value bound by the step.

        Returns:
            tuple[int, ...]: The shape of the bound value, of its resolved
            representation, or an empty tuple when unknown.

        Raises:
            ValueError: When neither the value nor its resolved
                representation defines a shape.
        """

        # The tensor value defines its own shape
        if isinstance(self.value, Tensor):
            return tuple(int(dim) for dim in self.value.shape)

        # The resolved representation of the instruction defines the shape
        instruction_resolved: TensorRepr | None = self._instruction_resolved()
        if instruction_resolved is not None and instruction_resolved.shape != ():
            return tuple(int(dim) for dim in instruction_resolved.shape)

        # The shape of the edited variable is unknown
        raise ValueError(
            f"Cannot infer the shape of the edited variable '{self.out_var}', "
            "pass an explicit shape."
        )

    def _instruction_resolved(self) -> TensorRepr | None:
        """
        Get the resolved representation of the instruction of the step.

        Returns:
            TensorRepr | None: The resolved representation, or None when the
            instruction holds none.
        """

        # The resolved representation is searched on the instruction types
        # carrying one, through a small reflection over the block body
        instruction: Instruction
        for instruction in self.block.body:
            if instruction.out_var == self.out_var \
                    and instruction.resolved is not None:
                return instruction.resolved
        return None

    @property
    def default_dtype(self) -> torch.dtype:
        """
        Get the dtype of the value bound by the step.

        Returns:
            torch.dtype: The dtype of the bound value, of its resolved
            representation, or float32 when unknown.
        """

        # The tensor value defines its own dtype
        if isinstance(self.value, Tensor):
            return self.value.dtype

        # The resolved representation of the instruction defines the dtype
        instruction_resolved: TensorRepr | None = self._instruction_resolved()
        if instruction_resolved is not None:
            try:
                return getattr(torch, instruction_resolved.dtype.split(".")[-1])
            except AttributeError:
                pass

        # The float32 default dtype
        return torch.float32

    def edit(
        self,
        values: Any,
        shape: tuple[int, ...] | None = None,
        dtype: torch.dtype | None = None,
        frame_depth: int | None = None,
        out_var: str | None = None
    ) -> None:
        """
        Patch a scope frame with a new tensor before the next instruction.

        In the generator mode (step_run), the edit is applied by the worker
        thread right after the consumer requests the next step, so call it
        between two yields. In the raw hook mode (step_hook), it is applied
        immediately: call it from within the hook for a mid-run effect.

        Args:
            values (Any): The new values of the variable, any tensor-like
                accepted by torch.as_tensor (list, tuple, tensor).
            shape (tuple[int, ...] | None): The shape of the new tensor,
                defaulting to the shape of the edited value.
            dtype (torch.dtype | None): The dtype of the new tensor,
                defaulting to the dtype of the edited value.
            frame_depth (int | None): The depth of the scope frame to patch,
                defaulting to the frame of the step.
            out_var (str | None): The name of the variable to patch,
                defaulting to the variable of the step.

        Raises:
            ValueError: When the step binds no variable, the value count
                does not match the shape, or the shape is unknown.
        """

        # The variable to patch defaults to the variable of the step
        target_frame: int = self.frame_depth if frame_depth is None \
            else frame_depth
        target_var: str = self.out_var if out_var is None else out_var
        if not target_var:
            raise ValueError("This step binds no variable to edit.")

        # The shape and dtype default to the ones of the edited value
        target_shape: tuple[int, ...] = shape if shape is not None \
            else self.default_shape
        target_dtype: torch.dtype = dtype if dtype is not None \
            else self.default_dtype

        # Build the new tensor of the variable
        new_tensor: Tensor = torch.as_tensor(
            values, dtype=target_dtype
        ).reshape(target_shape)

        # The edit is queued on the stepper in the generator mode, and
        # applied by the worker thread before the next instruction
        if self._stepper is not None:
            self._stepper.queue_edit(
                frame_depth=target_frame,
                out_var=target_var,
                tensor=new_tensor
            )
            return

        # The raw hook mode applies the edit immediately
        self.scope_frames[target_frame][target_var] = new_tensor


def step_kind(instruction: Instruction) -> str:
    """
    Get the step kind of an instruction.

    Args:
        instruction (Instruction): The executed instruction.

    Returns:
        str: One of "op", "block_call", "if", "for", "while", "try",
        "return", "continue" or "break".
    """

    # The dispatch mirrors the instruction types of the schema
    if isinstance(instruction, OpInstruction):
        return "op"
    if isinstance(instruction, BlockInstruction):
        return "block_call"
    if isinstance(instruction, ControlInstruction):
        return instruction.kind
    if isinstance(instruction, TryInstruction):
        return "try"
    if isinstance(instruction, ReturnInstruction):
        return "return"
    if isinstance(instruction, ContinueInstruction):
        return "continue"
    if isinstance(instruction, BreakInstruction):
        return "break"

    # The unknown instruction kinds are described by their type
    return type(instruction).__name__


def step_op_name(instruction: Instruction) -> str:
    """
    Get the display name of an executed instruction.

    Args:
        instruction (Instruction): The executed instruction.

    Returns:
        str: The qualified name of the callable for an operation, the block
        reference path for a sub-block call, "" otherwise.
    """

    # The operation instructions carry their qualified name
    if isinstance(instruction, OpInstruction):
        return instruction.qualified_name

    # The block calls carry the path of the called sub-block
    if isinstance(instruction, BlockInstruction):
        return instruction.block_ref.path

    # The other instruction kinds have no display name
    return ""


def make_step_info(
    step_index: int,
    instruction: Instruction,
    block: ModelBlockSchema,
    scope: dict[str, Any],
    scope_frames: list[dict[str, Any]]
) -> StepInfo:
    """
    Build the step info of an executed instruction.

    Args:
        step_index (int): The 1-based chronological index of the step.
        instruction (Instruction): The executed instruction.
        block (ModelBlockSchema): The block schema owning the instruction.
        scope (dict[str, Any]): The scope of the block executing it.
        scope_frames (list[dict[str, Any]]): The live stack of the executed
            block scopes, the innermost frame last.

    Returns:
        StepInfo: The step info of the instruction.
    """

    # The innermost frame is the last of the stack
    frame_depth: int = len(scope_frames) - 1

    # The step info of the instruction
    return StepInfo(
        step_index=step_index,
        kind=step_kind(instruction=instruction),
        out_var=instruction.out_var,
        op_name=step_op_name(instruction=instruction),
        block=block,
        scope=scope,
        scope_frames=scope_frames,
        frame_depth=frame_depth,
        value=scope.get(instruction.out_var) if instruction.out_var else None
    )


# The type of the raw step hooks of the interpreter
StepHook = Callable[[StepInfo], None]
