"""
Recording of the activation and gradient traces of a schema execution.

The ActivationRecorder is a TraceListener allocating a TensorContainer with a
stable container id for every captured tensor of an execution, and building
the ActivationTrace of the forward pass. When capture_grads is enabled, the
live tensors are kept with retain_grad so a backward pass can be followed by
collect_gradients to build the cross-referenced GradientTrace.
"""

# Import Modules
from typing import Any

import torch
from torch import Tensor

from nnschemaparser.core.traces import ActivationTrace, GradientTrace, TraceStep
from nnschemaparser.core.instructions import (
    BlockInstruction,
    Instruction,
    OpInstruction,
    ReturnInstruction
)
from nnschemaparser.core.model_schema import ModelBlockSchema
from nnschemaparser.core.tensor_schema import TensorContainer
from nnschemaparser.interpreter.tracer import TraceListener


# The human readable descriptions of the operations, matched by substring of
# the qualified name, the first match wins
_OP_DESCRIPTIONS: list[tuple[str, str]] = [
    ("linear", "Linear projection"),
    ("addmm", "Tensor multiplication"),
    ("bmm", "Batched tensor multiplication"),
    ("matmul", "Tensor multiplication"),
    ("layer_norm", "Layer normalization"),
    ("batch_norm", "Batch normalization"),
    ("group_norm", "Group normalization"),
    ("instance_norm", "Instance normalization"),
    ("relu", "ReLU activation"),
    ("gelu", "GELU activation"),
    ("silu", "SiLU activation"),
    ("sigmoid", "Sigmoid activation"),
    ("tanh", "Tanh activation"),
    ("softmax", "Softmax normalization"),
    ("log_softmax", "Log softmax normalization"),
    ("dropout", "Dropout"),
    ("conv", "Convolution"),
    ("embedding", "Embedding lookup"),
    ("cat", "Tensor concatenation"),
    ("split", "Tensor split"),
    ("chunk", "Tensor chunk"),
    ("gather", "Tensor gather"),
    ("cumsum", "Cumulative sum"),
    ("clamp", "Tensor clamp"),
    ("where", "Conditional selection"),
    ("reshape", "Tensor reshape"),
    ("view", "Tensor view"),
    ("transpose", "Tensor transpose"),
    ("permute", "Tensor permutation"),
    ("flatten", "Tensor flattening"),
    ("unsqueeze", "Dimension insertion"),
    ("squeeze", "Dimension removal"),
    ("expand", "Tensor expansion"),
    ("repeat", "Tensor repetition"),
    ("pad", "Tensor padding"),
    ("pool", "Pooling"),
    ("sum", "Tensor sum"),
    ("mean", "Tensor mean"),
    ("max", "Tensor maximum"),
    ("min", "Tensor minimum"),
    ("pow", "Power"),
    ("sqrt", "Square root"),
    ("rsqrt", "Reciprocal square root"),
    ("exp", "Exponential"),
    ("log", "Logarithm"),
    ("add", "Elementwise addition"),
    ("sub", "Elementwise subtraction"),
    ("mul", "Elementwise multiplication"),
    ("div", "Elementwise division"),
    ("to", "Dtype or device cast")
]


def describe_op(qualified_name: str) -> str:
    """
    Get the human readable description of an operation.

    Args:
        qualified_name (str): The display name of the operation (e.g.
            "torch.nn.functional.softmax").

    Returns:
        str: E.g. "Softmax normalization", or "Call to {qualified_name}"
        when the operation is unknown.
    """

    # The lowercase name of the operation without its module prefixes
    name: str = qualified_name.lower()
    suffix: str = name.rsplit(".", 1)[-1]

    # The first matching description wins
    for pattern, description in _OP_DESCRIPTIONS:
        if pattern in name or pattern in suffix:
            return description

    # The fallback description names the operation itself
    return f"Call to {qualified_name}"


class ActivationRecorder(TraceListener):
    """
    Recorder of the activation and gradient traces of a schema execution.

    The recorder allocates a TensorContainer with a stable container id for
    every captured tensor: the root block inputs ("input_0001", ...), then
    one container per tensor produced by an operation or a sub-block call
    ("{block_path}_{step}", ...). With capture_grads=True, the live tensors
    are kept with retain_grad so a backward pass followed by
    collect_gradients produces the cross-referenced GradientTrace.
    """

    def __init__(self, capture_grads: bool = False) -> None:
        """
        Init function.

        Args:
            capture_grads (bool): When True, the live tensors are retained
                for the gradient collection after a backward pass.
        """

        # Store here whether the gradients must be capturable
        self.capture_grads: bool = capture_grads

        # The activation trace filled during the forward pass
        self.trace: ActivationTrace = ActivationTrace()

        # The nesting depth of the executed blocks
        self._depth: int = 0

        # The stack of the entered blocks with their attribute paths
        self._stack: list[tuple[ModelBlockSchema, str]] = []

        # The live tensors kept for the gradient collection, as tuples of
        # (container id, activation step, live tensor)
        self._live: list[tuple[str, TraceStep, Tensor]] = []

    def _current_path(self) -> str:
        """
        Get the attribute path of the block being executed.

        Returns:
            str: The attribute path of the innermost entered block, or ""
            outside of any block.
        """

        # The path of the innermost entered block
        if self._stack:
            return self._stack[-1][1]
        return ""

    def _block_path(self, block: ModelBlockSchema) -> str:
        """
        Get the attribute path of an entered block from its parent.

        Args:
            block (ModelBlockSchema): The block being entered.

        Returns:
            str: The key of the block in its parent sub-blocks or methods,
            or the module class name for the root block.
        """

        # The root block is named after its module class
        if not self._stack:
            if block.module is not None:
                return type(block.module).__name__
            return ""

        # The nested blocks are searched by identity in their parent
        parent: ModelBlockSchema = self._stack[-1][0]
        candidates: dict[str, ModelBlockSchema] = {
            **parent.sub_blocks,
            **parent.methods
        }
        key: str
        sub_block: ModelBlockSchema
        for key, sub_block in candidates.items():
            if sub_block is block:
                return key

        # The fallback name is the module class name
        if block.module is not None:
            return type(block.module).__name__
        return ""

    def _record(
        self,
        description: str,
        slot_type: str,
        value: Tensor,
        out_var: str,
        op_name: str
    ) -> None:
        """
        Allocate a container and record one step of the activation trace.

        Args:
            description (str): The human readable description of the step.
            slot_type (str): The role of the tensor, "input" or "activation".
            value (Tensor): The live tensor produced by the execution.
            out_var (str): The original variable name of the tensor.
            op_name (str): The display name of the producing operation.
        """

        # The next chronological step number of the trace
        step_number: int = len(self.trace.steps) + 1

        # The container id is prefixed by the block path of the capture
        block_path: str = self._current_path()
        prefix: str = slot_type if slot_type in ("input", "output") \
            else (block_path.replace(".", "_") or "tensor")
        container_id: str = f"{prefix}_{step_number:04d}"

        # The depth of the scope frame holding the variable of the capture
        frame_depth: int = max(0, len(self._stack) - 1)

        # The container holding the detached snapshot of the tensor
        container: TensorContainer = TensorContainer(
            shape=tuple(int(dim) for dim in value.shape),
            dtype=str(value.dtype),
            value=value.detach(),
            container_id=container_id,
            slot_type=slot_type
        )

        # The trace step referencing the container
        step: TraceStep = TraceStep(
            step=step_number,
            container_id=container_id,
            description=description,
            slot_type=slot_type,
            out_var=out_var,
            op_name=op_name,
            block_path=block_path,
            tensor=container,
            frame_depth=frame_depth
        )
        self.trace.steps.append(step)

        # The live tensors requiring grad are kept for the gradient capture
        if self.capture_grads and torch.is_grad_enabled() and value.requires_grad:

            # Only the non leaf tensors need an explicit retain_grad, the
            # gradients of the leaf inputs are populated by autograd itself
            if not value.is_leaf:
                value.retain_grad()
            self._live.append((container_id, step, value))

    def on_block_enter(self, block: ModelBlockSchema, scope: dict[str, Any]) -> None:
        """
        Record the root block inputs and push the block on the path stack.

        Args:
            block (ModelBlockSchema): The block being entered.
            scope (dict[str, Any]): The scope of the block execution.
        """

        # Push the block with its attribute path on the stack, before the
        # inputs are recorded so their frame depth is the root frame depth
        self._stack.append((block, self._block_path(block=block)))

        # The inputs of the root block open the trace
        if self._depth == 0:
            input_name: str
            for input_name in block.inputs:
                value: Any = scope.get(input_name)
                if isinstance(value, Tensor):
                    self._record(
                        description="Initial input tensor",
                        slot_type="input",
                        value=value,
                        out_var=input_name,
                        op_name="input"
                    )

        self._depth += 1

    def on_block_exit(self, block: ModelBlockSchema, result: list[Any]) -> None:
        """
        Pop the exited block from the path stack and record the outputs of
        the root block.

        Args:
            block (ModelBlockSchema): The block being exited.
            result (list[Any]): The values returned by the block.
        """

        # The outputs of the root block close the trace
        if self._depth == 1:
            self._record_root_outputs(block=block, result=result)

        # The block path stack is popped on every block exit
        if self._stack:
            self._stack.pop()
        self._depth = max(0, self._depth - 1)

    def _record_root_outputs(
        self,
        block: ModelBlockSchema,
        result: list[Any]
    ) -> None:
        """
        Record the output tensors of the root block as output containers.

        Args:
            block (ModelBlockSchema): The root block being exited.
            result (list[Any]): The values returned by the block.
        """

        # The original variable names of the returned values
        out_vars: list[str] = self._return_vars(block=block)

        # Each returned tensor is recorded as an output container
        index: int
        value: Any
        for index, value in enumerate(result):
            if not isinstance(value, Tensor):
                continue
            self._record(
                description="Model output tensor",
                slot_type="output",
                value=value,
                out_var=out_vars[index] if index < len(out_vars) else "",
                op_name="output"
            )

    def _return_vars(self, block: ModelBlockSchema) -> list[str]:
        """
        Get the returned variable names of a block from its body.

        Args:
            block (ModelBlockSchema): The block to inspect.

        Returns:
            list[str]: The out_vars of the last return instruction, empty
            when the block has no parsed return instruction.
        """

        # Search the body of the block for the last return instruction
        instruction: Instruction
        for instruction in reversed(block.body):
            if isinstance(instruction, ReturnInstruction):
                return instruction.out_vars

        # The block has no parsed return instruction
        return []

    def on_op(self, instruction: OpInstruction, scope: dict[str, Any]) -> None:
        """
        Record the tensor produced by an operation instruction.

        Args:
            instruction (OpInstruction): The executed operation.
            scope (dict[str, Any]): The scope holding the result.
        """

        # Only the tensor results are recorded
        value: Any = scope.get(instruction.out_var)
        if not isinstance(value, Tensor):
            return

        # The step describes the operation and the bound variable
        self._record(
            description=describe_op(qualified_name=instruction.qualified_name),
            slot_type="activation",
            value=value,
            out_var=instruction.out_var,
            op_name=instruction.qualified_name
        )

    def on_block_call(
        self,
        instruction: BlockInstruction,
        sub_block: ModelBlockSchema | None,
        scope: dict[str, Any]
    ) -> None:
        """
        Record the tensor produced by a sub-block call instruction.

        Args:
            instruction (BlockInstruction): The executed block call.
            sub_block (ModelBlockSchema | None): The called sub-block, None
                when the call was re-dispatched to a real module.
            scope (dict[str, Any]): The scope holding the result.
        """

        # Only the tensor results are recorded
        value: Any = scope.get(instruction.out_var)
        if not isinstance(value, Tensor):
            return

        # The description names the called sub-block or real module
        called: str = instruction.block_ref.path
        if sub_block is not None and sub_block.module is not None:
            called = f"{called} ({type(sub_block.module).__name__})"
        self._record(
            description=f"Call to sub-block {called}",
            slot_type="activation",
            value=value,
            out_var=instruction.out_var,
            op_name=instruction.block_ref.path
        )

    def collect_gradients(self) -> GradientTrace:
        """
        Build the gradient trace of the last recorded forward pass.

        The method must be called after the backward pass of a loss computed
        from the outputs of the recorded execution (with capture_grads=True
        at recorder construction). The gradient steps mirror the activation
        steps in the same order: the container id of a gradient step is the
        container id of its activation step prefixed with "grad_" and its
        source_container_id points back to the activation step.

        Returns:
            GradientTrace: The gradients of the recorded activations, empty
            when the backward pass has not been run or no tensor required
            grad.
        """

        # The gradient trace mirrors the activation trace
        trace: GradientTrace = GradientTrace()

        # The retained live tensors hold their gradient after the backward
        container_id: str
        activation_step: TraceStep
        live: Tensor
        for container_id, activation_step, live in self._live:

            # The tensors that did not contribute to the loss have no grad
            grad: Tensor | None = live.grad
            if grad is None:
                continue

            # The gradient step is numbered in activation order
            step_number: int = len(trace.steps) + 1
            gradient_id: str = f"grad_{container_id}"

            # The container holding the detached gradient snapshot
            container: TensorContainer = TensorContainer(
                shape=tuple(int(dim) for dim in grad.shape),
                dtype=str(grad.dtype),
                value=grad.detach(),
                container_id=gradient_id,
                slot_type="gradient"
            )

            # The gradient step cross-references its activation step
            step: TraceStep = TraceStep(
                step=step_number,
                container_id=gradient_id,
                description=f"Gradient of {activation_step.description}",
                slot_type="gradient",
                out_var=activation_step.out_var,
                op_name=activation_step.op_name,
                block_path=activation_step.block_path,
                tensor=container,
                source_container_id=container_id
            )
            trace.steps.append(step)

        # The cross-referenced gradient trace
        return trace
