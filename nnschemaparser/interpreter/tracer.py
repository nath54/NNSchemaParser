"""
Tracing of the schema interpreter executions.

A TraceListener receives one callback per interpreter event (block enter and
exit, operations, sub-block calls, control flows and try statements). The
listeners are registered on a SchemaInterpreter at construction time and are
called synchronously during its run, in execution order.
"""

# Import Modules
from typing import Any

from nnschemaparser.core.instructions import (
    BlockInstruction,
    ControlInstruction,
    OpInstruction,
    TryInstruction
)
from nnschemaparser.core.model_schema import ModelBlockSchema
from nnschemaparser.dump import MAX_LINES, describe_ref, describe_value


def _module_type_name(block: ModelBlockSchema) -> str:
    """
    Get the display name of the module class of a block.

    Args:
        block (ModelBlockSchema): The block to name.

    Returns:
        str: The module class name, or "?" without a source module.
    """

    # Without a source module there is no name to display
    if block.module is None:
        return "?"

    # The class name of the source module, with the method name if any
    module_type: str = type(block.module).__name__
    if block.method_name is not None:
        return f"{module_type}.{block.method_name}"
    return module_type


class TraceListener:
    """
    Base class of the schema execution listeners.

    Every hook is a no-op by default, so a subclass only implements the
    events it needs. The hooks are called from the interpreter during its
    run and must not modify the executed scope.
    """

    def on_block_enter(self, block: ModelBlockSchema, scope: dict[str, Any]) -> None:
        """
        Called when the execution of a block starts.

        Args:
            block (ModelBlockSchema): The block being entered.
            scope (dict[str, Any]): The scope of the block execution.
        """

        # Do nothing by default

    def on_block_exit(self, block: ModelBlockSchema, result: list[Any]) -> None:
        """
        Called when the execution of a block ends.

        Args:
            block (ModelBlockSchema): The block being exited.
            result (list[Any]): The values returned by the block.
        """

        # Do nothing by default

    def on_op(self, instruction: OpInstruction, scope: dict[str, Any]) -> None:
        """
        Called after an operation instruction has been executed.

        Args:
            instruction (OpInstruction): The executed operation.
            scope (dict[str, Any]): The scope holding the result.
        """

        # Do nothing by default

    def on_block_call(
        self,
        instruction: BlockInstruction,
        sub_block: ModelBlockSchema | None,
        scope: dict[str, Any]
    ) -> None:
        """
        Called after a sub-block call instruction has been executed.

        Args:
            instruction (BlockInstruction): The executed block call.
            sub_block (ModelBlockSchema | None): The called sub-block, None
                when the call was re-dispatched to a real module of a skipped
                container entry.
            scope (dict[str, Any]): The scope holding the result.
        """

        # Do nothing by default

    def on_control_enter(self, instruction: ControlInstruction) -> None:
        """
        Called before a control flow instruction is executed.

        Args:
            instruction (ControlInstruction): The control flow instruction.
        """

        # Do nothing by default

    def on_control_exit(self, instruction: ControlInstruction) -> None:
        """
        Called after a control flow instruction has been executed.

        Args:
            instruction (ControlInstruction): The control flow instruction.
        """

        # Do nothing by default

    def on_try_enter(self, instruction: TryInstruction) -> None:
        """
        Called before a try instruction is executed.

        Args:
            instruction (TryInstruction): The try instruction.
        """

        # Do nothing by default

    def on_try_exit(self, instruction: TryInstruction) -> None:
        """
        Called after a try instruction has been executed.

        Args:
            instruction (TryInstruction): The try instruction.
        """

        # Do nothing by default


class TraceRecorder(TraceListener):
    """
    Recorder of a chronological text trace of a schema execution.

    One line is recorded per event, indented by the current nesting depth,
    with the value signatures of every intermediate tensor. The trace is
    written to a text file by the save method.
    """

    def __init__(self, cap: int = MAX_LINES) -> None:
        """
        Init function.

        Args:
            cap (int): The maximum number of recorded lines.
        """

        # The recorded lines, the nesting depth and the line budget
        self.lines: list[str] = []
        self.depth: int = 0
        self.count: int = 0
        self.cap: int = cap

    def _log(self, text: str) -> None:
        """
        Record one line of the trace with the current indentation.

        Args:
            text (str): The line to record.
        """

        # Skip the line when the budget is reached
        if self.count >= self.cap:
            return
        self.count += 1
        self.lines.append(f"[{self.count:05d}] {'  ' * self.depth}{text}")

    def on_block_enter(self, block: ModelBlockSchema, scope: dict[str, Any]) -> None:
        """
        Record the entry of a block with its input values.

        Args:
            block (ModelBlockSchema): The block being entered.
            scope (dict[str, Any]): The scope of the block execution.
        """

        # The header of the block, then its input values
        self._log_block_header(block=block)
        self.depth += 1
        input_name: str
        for input_name in block.inputs:
            if input_name in scope:
                self._log(f"in {input_name} = {describe_value(scope[input_name])}")

    def _log_block_header(self, block: ModelBlockSchema) -> None:
        """
        Record the header line of a block entry.

        Args:
            block (ModelBlockSchema): The block being entered.
        """

        # The header with the module class name
        self._log(f"ENTER BLOCK {_module_type_name(block=block)}")

    def on_block_exit(self, block: ModelBlockSchema, result: list[Any]) -> None:
        """
        Record the exit of a block with its output values.

        Args:
            block (ModelBlockSchema): The block being exited.
            result (list[Any]): The values returned by the block.
        """

        # The outputs of the block and the dedent
        self.depth = max(0, self.depth - 1)
        self._log(
            f"EXIT BLOCK {_module_type_name(block=block)} "
            f"-> {describe_value(tuple(result))}"
        )

    def on_op(self, instruction: OpInstruction, scope: dict[str, Any]) -> None:
        """
        Record the result of an operation instruction.

        Args:
            instruction (OpInstruction): The executed operation.
            scope (dict[str, Any]): The scope holding the result.
        """

        # The target, arguments and result of the operation
        args: str = ", ".join(describe_ref(arg) for arg in instruction.args)
        kwargs: str = ", ".join(
            f"{key}={describe_ref(value)}"
            for key, value in instruction.kwargs.items()
        )
        all_args: str = ", ".join(part for part in (args, kwargs) if part)
        self._log(
            f"{instruction.out_var} = {instruction.qualified_name}({all_args})"
            f"  [{describe_value(scope.get(instruction.out_var))}]"
        )

    def on_block_call(
        self,
        instruction: BlockInstruction,
        sub_block: ModelBlockSchema | None,
        scope: dict[str, Any]
    ) -> None:
        """
        Record the result of a sub-block call instruction.

        Args:
            instruction (BlockInstruction): The executed block call.
            sub_block (ModelBlockSchema | None): The called sub-block, None
                for a real module call of a skipped container entry.
            scope (dict[str, Any]): The scope holding the result.
        """

        # The call reference depends on the instruction kind: a method call
        # is an operation instruction, a sub-block call a block instruction
        if isinstance(instruction, OpInstruction):
            call_ref: str = instruction.qualified_name
        else:
            call_ref = instruction.block_ref.path

        # The called module type, arguments and result of the call
        module_type: str = _module_type_name(block=sub_block) \
            if sub_block is not None else "?"
        args: str = ", ".join(describe_ref(arg) for arg in instruction.args)
        self._log(
            f"{instruction.out_var} = CALL {call_ref}({args})"
            f" [{module_type}]"
            f"  [{describe_value(scope.get(instruction.out_var))}]"
        )

    def on_control_enter(self, instruction: ControlInstruction) -> None:
        """
        Record the entry of a control flow instruction.

        Args:
            instruction (ControlInstruction): The control flow instruction.
        """

        # The if marker or the for header, then the indent of the body
        if instruction.kind == "if":
            self._log("IF")
        else:
            self._log(
                f"FOR {instruction.bind_var} IN "
                f"{instruction.iter_container.path if instruction.iter_container else '?'}"
            )
        self.depth += 1

    def on_control_exit(self, instruction: ControlInstruction) -> None:
        """
        Record the exit of a control flow instruction.

        Args:
            instruction (ControlInstruction): The control flow instruction.
        """

        # The dedent of the control flow body
        self.depth = max(0, self.depth - 1)

    def on_try_enter(self, instruction: TryInstruction) -> None:
        """
        Record the entry of a try instruction.

        Args:
            instruction (TryInstruction): The try instruction.
        """

        # The try marker, then the indent of the body
        self._log("TRY")
        self.depth += 1

    def on_try_exit(self, instruction: TryInstruction) -> None:
        """
        Record the exit of a try instruction.

        Args:
            instruction (TryInstruction): The try instruction.
        """

        # The dedent of the try body
        self.depth = max(0, self.depth - 1)

    def save(self, path: str) -> None:
        """
        Write the recorded trace to a text file.

        Args:
            path (str): The path of the output text file.
        """

        # The full recorded trace
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(self.lines) + "\n")
