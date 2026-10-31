"""
Schema interpreter of NNSchemaParser.

Executes the body of a ModelBlockSchema instruction by instruction, by
re-dispatching the recorded calls to the real torch functions. It is used in
two modes:

- parse time, under FakeTensorMode, to resolve the shapes and dtypes;
- WebUI time, with real tensors, to capture every intermediate value.
"""

# Import Modules
from typing import Any, Callable, Generator

import inspect
import queue
import threading

from torch import Tensor

from nnschemaparser.core.model_schema import ModelBlockSchema
from nnschemaparser.forward import resolve_forward
from nnschemaparser.core.instructions import (
    BlockInstruction,
    BlockRef,
    BreakInstruction,
    ContinueInstruction,
    ControlInstruction,
    Instruction,
    OpInstruction,
    ReturnInstruction,
    TryInstruction,
    VarRef
)
from nnschemaparser.interpreter.tracer import TraceListener
from nnschemaparser.interpreter.stepping import StepInfo, _Stepper, make_step_info
from nnschemaparser.interpreter.control_flow import (
    ControlFlowExecutor,
    _LoopBreak,
    _LoopContinue
)
from nnschemaparser.interpreter.execution import InstructionExecutor


class SchemaInterpreter(InstructionExecutor, ControlFlowExecutor):
    """
    Executes the body of a ModelBlockSchema block.
    """

    def __init__(
        self,
        capture_values: bool = False,
        listeners: list[TraceListener] | None = None
    ) -> None:
        """
        Init function.

        Args:
            capture_values (bool): If True, every tensor result is kept as a
                TensorReprWithValue inside the instruction resolved field.
            listeners (list[TraceListener] | None): The optional listeners
                notified of every event of the schema executions.
        """

        # Store here whether the intermediate values must be captured
        self.capture_values: bool = capture_values

        # Store here the listeners of the interpreter events
        self.listeners: list[TraceListener] = listeners or []

        # The stack of the executed block scopes, the innermost frame last.
        # Maintained by run() so the step hooks and the consumers can locate
        # the frame depth of any variable of the execution
        self._scope_stack: list[dict[str, Any]] = []

        # The counter of the published instruction steps of the executions
        self._step_counter: int = 0

        # The optional raw hook called after each executed instruction (and
        # before the body of the control flow and try instructions). The
        # hook may mutate the scope frames of the yielded StepInfo for a
        # mid-run effect, and must not run the interpreter recursively
        self.step_hook: Callable[[StepInfo], None] | None = None

        # The outputs of the last completed stepped run, also returned as
        # the StopIteration value of the step_run generator
        self.step_outputs: list[Any] | None = None

    def run(self, block: ModelBlockSchema, scope: dict[str, Any]) -> list[Any]:
        """
        Execute the body of a block schema.

        Args:
            block (ModelBlockSchema): The block schema to execute.
            scope (dict[str, Any]): The initial variables of the block (its
                inputs), modified in place during the execution.

        Returns:
            list[Any]: The values returned by the block.
        """

        # Seed the scope with the weights of the block
        self._seed_weights(block=block, scope=scope)

        # Seed the scope with the source module of the block
        if block.module is not None:
            scope.setdefault("$self", block.module)

        # Seed the scope with the defaults of the forward arguments
        if block.module is not None:
            self._seed_forward_defaults(block=block, scope=scope)

        # Resolve the placeholder representations of the block inputs
        self._resolve_block_inputs(block=block, scope=scope)

        # Push the scope of the block on the frame stack for the whole run
        self._scope_stack.append(scope)
        try:

            # Notify the listeners of the block entry
            self._notify_block_enter(block=block, scope=scope)

            # Execute all the instructions of the block body in order
            instruction: Instruction
            for instruction in block.body:

                # Execute the current instruction of the block body
                result: list[Any] | None = self._execute_instruction(
                    instruction=instruction,
                    block=block,
                    scope=scope
                )

                # A returned result stops the execution of the block
                if result is not None:
                    self._notify_block_exit(block=block, result=result)
                    return result

            # If no return instruction was found, the block returns nothing
            self._notify_block_exit(block=block, result=[])
            return []
        finally:

            # Pop the scope frame of the block whatever the outcome
            self._scope_stack.pop()

    def _publish_step(
        self,
        instruction: Instruction,
        block: ModelBlockSchema,
        scope: dict[str, Any]
    ) -> None:
        """
        Publish a StepInfo of an executed instruction to the raw step hook.

        Args:
            instruction (Instruction): The executed instruction.
            block (ModelBlockSchema): The block schema owning it.
            scope (dict[str, Any]): The scope of the block executing it.
        """

        # No step hook, no step publication
        if self.step_hook is None:
            return

        # The next chronological index of the step stream
        self._step_counter += 1

        # The step info of the instruction, with the live scope frames
        step_info: StepInfo = make_step_info(
            step_index=self._step_counter,
            instruction=instruction,
            block=block,
            scope=scope,
            scope_frames=self._scope_stack
        )

        # The hook may mutate the scope frames of the step info
        self.step_hook(step_info)

    def step_run(
        self,
        block: ModelBlockSchema,
        scope: dict[str, Any]
    ) -> "Generator[StepInfo, None, list[Any]]":
        """
        Execute a block schema step by step as a generator of StepInfo.

        One StepInfo is yielded per executed instruction, in execution order
        (per-iteration steps for the loop bodies, nested sub-block
        instructions included, control flow and try instructions published
        before their body). The consumer may call the edit method of a
        yielded step between two iterations to patch a scope frame before
        the next instruction. The outputs of the run are the return value
        of the generator, also available in the step_outputs attribute, and
        the scope is mutated in place like for the run method.

        The execution runs on a worker thread gated by the generator: the
        interpreter itself stays synchronous and no listener is required.
        Only one stepped run may be active at a time per interpreter
        instance. Abandoning the generator (break, close) unwinds the run
        at the next step without executing the remaining instructions.

        Args:
            block (ModelBlockSchema): The block schema to execute.
            scope (dict[str, Any]): The initial variables of the block (its
                inputs), mutated in place during the execution.

        Returns:
            Generator[StepInfo, None, list[Any]]: The generator of the step
            infos, returning the outputs of the block.
        """

        # The gate of the worker thread and the queue of the yielded steps
        stepper: _Stepper = _Stepper()
        steps: queue.Queue = queue.Queue()
        holder: dict[str, Any] = {}

        def sink(step_info: StepInfo) -> None:
            """Publish the step then apply the queued edits on wake."""

            # Bind the stepper so the edits of the consumer are queued
            step_info._bind_stepper(stepper=stepper)

            # Publish the step to the generator and wait for the release
            steps.put(step_info)
            edits: list[tuple[int, str, Tensor]] = stepper.wait()

            # Apply the queued edits to the live scope frames; a closed
            # stepper raises the abort signal from the wait call itself
            edit_frame_depth: int
            edit_out_var: str
            edit_tensor: Tensor
            for edit_frame_depth, edit_out_var, edit_tensor in edits:
                step_info.scope_frames[edit_frame_depth][edit_out_var] = edit_tensor

        def worker() -> None:
            """Run the schema on this thread with the step sink installed."""

            # Install the sink and restore the previous hook afterwards
            previous_hook: Callable[[StepInfo], None] | None = self.step_hook
            self.step_hook = sink
            try:
                holder["outputs"] = self.run(block=block, scope=scope)
            except BaseException as error:  # noqa: BLE001 - forwarded below
                holder["error"] = error
            finally:
                self.step_hook = previous_hook

                # The end sentinel of the step queue, whatever the outcome
                steps.put(None)

        # Reset the step counter so forks replay the same step indices
        self._step_counter = 0

        # Start the worker thread gating the instruction executions
        worker_thread: threading.Thread = threading.Thread(
            target=worker,
            daemon=True
        )
        worker_thread.start()

        # The generator proper: yield the steps, release on each resume
        try:
            while True:

                # The next step of the worker, or its end sentinel
                item: StepInfo | None = steps.get()
                if item is None:
                    break

                # The consumer may edit the scope frames of the step
                yield item

                # Let the worker apply the queued edits and continue
                stepper.release()

            # Forward the worker error to the consumer
            if "error" in holder:
                raise holder["error"]

            # The outputs of the run, as the generator return value
            self.step_outputs = holder.get("outputs")
            return self.step_outputs
        finally:

            # An abandoned generator unwinds the worker at its next step
            stepper.close()

    def _notify_block_enter(self, block: ModelBlockSchema, scope: dict[str, Any]) -> None:
        """
        Notify every listener of a block execution entry.

        Args:
            block (ModelBlockSchema): The block being entered.
            scope (dict[str, Any]): The scope of the block execution.
        """

        # Dispatch the event to all the registered listeners
        listener: TraceListener
        for listener in self.listeners:
            listener.on_block_enter(block=block, scope=scope)

    def _notify_block_exit(self, block: ModelBlockSchema, result: list[Any]) -> None:
        """
        Notify every listener of a block execution exit.

        Args:
            block (ModelBlockSchema): The block being exited.
            result (list[Any]): The values returned by the block.
        """

        # Dispatch the event to all the registered listeners
        listener: TraceListener
        for listener in self.listeners:
            listener.on_block_exit(block=block, result=result)

    def _seed_weights(self, block: ModelBlockSchema, scope: dict[str, Any]) -> None:
        """
        Seed the scope with the weights of the block.

        Args:
            block (ModelBlockSchema): The block schema being executed.
            scope (dict[str, Any]): The initial variables of the block.
        """

        # Each weight is bound by its name in the block scope. The live
        # attribute of the source module is preferred over the parse time
        # snapshot so that runtime mutations of the weights are visible
        weight_name: str
        weight: Any
        for weight_name, weight in block.weights.items():
            live: Any | None = getattr(block.module, weight_name, None) \
                if block.module is not None else None
            scope.setdefault(
                weight_name,
                live if live is not None else weight.value
            )

    def _seed_forward_defaults(self, block: ModelBlockSchema, scope: dict[str, Any]) -> None:
        """
        Seed the scope with the default values of the forward arguments.

        Args:
            block (ModelBlockSchema): The block schema being executed.
            scope (dict[str, Any]): The initial variables of the block.
        """

        # The method blocks take the defaults of their own method
        if block.method_name is not None:
            callable_ = getattr(block.module, block.method_name)
        else:
            callable_ = resolve_forward(module=block.module)

        # Get the signature of the executed callable of the block
        signature: inspect.Signature = inspect.signature(callable_)
        parameter_name: str
        parameter: inspect.Parameter
        for parameter_name, parameter in signature.parameters.items():

            # Skip the self parameter of the forward method
            if parameter_name == "self":
                continue

            # Skip the arguments already provided by the caller
            if parameter_name in scope:
                continue

            # The var-keyword parameter receives an empty dict when unbound
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                scope.setdefault(parameter_name, {})
                continue

            # Seed the scope with the default value of the argument
            if parameter.default is not inspect.Parameter.empty:
                scope.setdefault(parameter_name, parameter.default)

    def _execute_instruction(
        self,
        instruction: Instruction,
        block: ModelBlockSchema,
        scope: dict[str, Any]
    ) -> list[Any] | None:
        """
        Dispatch the execution of an instruction to its specific handler.

        Args:
            instruction (Instruction): The instruction to execute.
            block (ModelBlockSchema): The block schema owning the instruction.
            scope (dict[str, Any]): The current variables of the block.

        Returns:
            list[Any] | None: The outputs of the block when the instruction is
                a return instruction, else None.
        """

        # The stepping hook is called around the instruction execution, with
        # the control flow, try and loop signal instructions published before
        # their execution so the stream follows the program order
        stepping: bool = self.step_hook is not None
        pre_published: bool = False
        if stepping and isinstance(
            instruction,
            (ControlInstruction, TryInstruction, ContinueInstruction, BreakInstruction)
        ):
            self._publish_step(instruction=instruction, block=block, scope=scope)
            pre_published = True

        # Dispatch to the execution of the specific instruction type,
        # propagating a return instruction reached in a nested body
        result: list[Any] | None = None
        if isinstance(instruction, OpInstruction):
            self._execute_op(
                instruction=instruction,
                block=block,
                scope=scope
            )

        elif isinstance(instruction, BlockInstruction):
            self._execute_block_call(
                instruction=instruction,
                block=block,
                scope=scope
            )

        elif isinstance(instruction, TryInstruction):
            result = self._execute_try(
                instruction=instruction,
                block=block,
                scope=scope
            )

        elif isinstance(instruction, ControlInstruction):
            result = self._execute_control(
                instruction=instruction,
                block=block,
                scope=scope
            )

        # A return instruction produces the outputs of the block
        elif isinstance(instruction, ReturnInstruction):
            result = self._execute_return(
                instruction=instruction,
                block=block,
                scope=scope
            )

        elif isinstance(instruction, ContinueInstruction):
            raise _LoopContinue()

        elif isinstance(instruction, BreakInstruction):
            raise _LoopBreak()

        else:
            raise TypeError(
                f"Unsupported instruction type: {type(instruction).__name__}"
            )

        # Publish the executed instruction when not already published
        if stepping and not pre_published:
            self._publish_step(instruction=instruction, block=block, scope=scope)

        # No return instruction was executed, or its result is propagated
        return result

    def _resolve_value(self, value: VarRef | Any, scope: dict[str, Any]) -> Any:
        """
        Resolve a call argument from the current block scope.

        Args:
            value (VarRef | Any): The argument, a variable reference or a literal.
            scope (dict[str, Any]): The current variables of the block.

        Returns:
            Any: The resolved value of the argument.
        """

        # If the value is a variable reference, we look it up in the scope
        if isinstance(value, VarRef):

            # Unknown variables are an error of the schema
            if value.name not in scope:
                raise KeyError(f"Unbound variable in block scope: {value.name}")

            # Return the value of the variable
            return scope[value.name]

        # Else the value is a literal, returned as is
        return value

    def _get_sub_block(
        self,
        block: ModelBlockSchema,
        block_ref: BlockRef,
        scope: dict[str, Any]
    ) -> ModelBlockSchema:
        """
        Get the sub-block schema targeted by a block reference.

        Args:
            block (ModelBlockSchema): The block schema owning the sub-block.
            block_ref (BlockRef): The reference to the targeted sub-block.
            scope (dict[str, Any]): The current variables of the block.

        Returns:
            ModelBlockSchema: The targeted sub-block schema.
        """

        # A direct sub-block is a simple dictionary lookup
        if block_ref.index_var is None:
            return block.sub_blocks[block_ref.path]

        # Get the container index value from the scope
        index: Any = scope[block_ref.index_var]

        # A loop variable bound to a sub-block schema is used directly
        if isinstance(index, ModelBlockSchema):
            return index

        # The self container entries are keyed by their index only
        if block_ref.path == "$self":
            return block.sub_blocks[str(index)]

        # The sub-block is a container entry, its key is "path.index"
        entry_key: str = f"{block_ref.path}.{index}"

        # Get the container entry from the computed key
        return block.sub_blocks[entry_key]

