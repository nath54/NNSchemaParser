"""
Trace recorder of the parser robustness sweep.

The recorder hooks the schema interpreter and writes a chronological
operation per operation log of an inference with the value signatures of
every intermediate tensor. The pure schema dumpers live in the package
(nnschemaparser.dump) and are re-exported here for convenience.
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
from nnschemaparser.dump import MAX_LINES, describe_value, dump_schema  # noqa: F401
from nnschemaparser.interpreter import SchemaInterpreter


class TraceRecorder:
    """
    Recorder of the chronological inference trace of a schema execution.
    """

    def __init__(self, cap: int = MAX_LINES) -> None:
        """
        Init function.

        Args:
            cap (int): The maximum number of recorded lines.
        """

        # The recorded lines and the current nesting depth
        self.lines: list[str] = []
        self.depth: int = 0
        self.count: int = 0
        self.cap: int = cap

    def log(self, text: str) -> None:
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

    def enter_block(self, block: ModelBlockSchema, scope: dict[str, Any]) -> None:
        """
        Record the entry of a block with its input values.

        Args:
            block (ModelBlockSchema): The entered block.
            scope (dict[str, Any]): The scope of the block.
        """

        # The header with the module class
        module_type: str = type(block.module).__name__ \
            if block.module is not None else "root"
        self.log(f"ENTER BLOCK {module_type}")
        self.depth += 1

        # The input values of the block
        input_name: str
        for input_name in block.inputs:
            if input_name in scope:
                self.log(f"in {input_name} = {describe_value(scope[input_name])}")

    def exit_block(self, block: ModelBlockSchema, result: list[Any]) -> None:
        """
        Record the exit of a block with its output values.

        Args:
            block (ModelBlockSchema): The exited block.
            result (list[Any]): The values returned by the block.
        """

        # The outputs and the dedent
        self.depth = max(0, self.depth - 1)
        module_type: str = type(block.module).__name__ \
            if block.module is not None else "root"
        self.log(f"EXIT BLOCK {module_type} -> {describe_value(tuple(result))}")

    def log_op(
        self,
        instruction: OpInstruction,
        scope: dict[str, Any]
    ) -> None:
        """
        Record the result of an operation instruction.

        Args:
            instruction (OpInstruction): The executed operation.
            scope (dict[str, Any]): The scope holding the result.
        """

        # The target, arguments and result of the operation
        args: str = ", ".join(
            value.name if hasattr(value, "name") else repr(value)[:40]
            for value in instruction.args
        )
        self.log(
            f"{instruction.out_var} = {instruction.qualified_name}({args})"
            f"  [{describe_value(scope.get(instruction.out_var))}]"
        )

    def log_block_call(
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
                for a skipped container entry re-dispatched at runtime.
            scope (dict[str, Any]): The scope holding the result.
        """

        # The reference and the result of the call
        module_type: str = type(sub_block.module).__name__ \
            if sub_block is not None and sub_block.module is not None else "?"
        self.log(
            f"{instruction.out_var} = CALL "
            f"{instruction.block_ref.path} [{module_type}]"
            f"  [{describe_value(scope.get(instruction.out_var))}]"
        )

    def log_control(self, instruction: ControlInstruction) -> None:
        """
        Record the entry of a control flow instruction.

        Args:
            instruction (ControlInstruction): The control instruction.
        """

        # The if and for kinds
        if instruction.kind == "if":
            self.log("IF")
        else:
            self.log(
                f"FOR {instruction.bind_var} IN "
                f"{instruction.iter_container.path if instruction.iter_container else '?'}"
            )

    def log_try(self) -> None:
        """Record the entry of a try instruction."""

        # The try marker
        self.log("TRY")

    def save(self, path: str) -> None:
        """
        Write the recorded trace to a text file.

        Args:
            path (str): The path of the output text file.
        """

        # The full recorded trace
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(self.lines) + "\n")


# The currently installed tracer, and the original interpreter methods
_ACTIVE_TRACER: TraceRecorder | None = None
_ORIG_RUN = SchemaInterpreter.run
_ORIG_OP = SchemaInterpreter._execute_op
_ORIG_BLOCK = SchemaInterpreter._execute_block_call
_ORIG_CONTROL = SchemaInterpreter._execute_control
_ORIG_TRY = SchemaInterpreter._execute_try


def _traced_run(self: SchemaInterpreter, block: ModelBlockSchema,
                scope: dict[str, Any]) -> list[Any]:
    """
    Traced replacement of the interpreter run method.

    Args:
        self (SchemaInterpreter): The interpreter instance.
        block (ModelBlockSchema): The block being executed.
        scope (dict[str, Any]): The scope of the block.

    Returns:
        list[Any]: The values returned by the block.
    """

    # Enter, run and exit the block
    tracer: TraceRecorder | None = _ACTIVE_TRACER
    if tracer is None:
        return _ORIG_RUN(self, block=block, scope=scope)
    try:
        tracer.enter_block(block=block, scope=scope)
    except Exception:  # noqa: BLE001 - the trace must never break the run
        pass
    result: list[Any] = _ORIG_RUN(self, block=block, scope=scope)
    try:
        tracer.exit_block(block=block, result=result)
    except Exception:  # noqa: BLE001 - the trace must never break the run
        pass
    return result


def _traced_op(self: SchemaInterpreter, instruction: OpInstruction,
               scope: dict[str, Any]) -> None:
    """
    Traced replacement of the operation execution.

    Args:
        self (SchemaInterpreter): The interpreter instance.
        instruction (OpInstruction): The operation being executed.
        scope (dict[str, Any]): The current scope.
    """

    # Run then record the result
    _ORIG_OP(self, instruction=instruction, scope=scope)
    tracer: TraceRecorder | None = _ACTIVE_TRACER
    if tracer is None:
        return
    try:
        tracer.log_op(instruction=instruction, scope=scope)
    except Exception:  # noqa: BLE001 - the trace must never break the run
        pass


def _traced_block_call(self: SchemaInterpreter, instruction: BlockInstruction,
                       block: ModelBlockSchema, scope: dict[str, Any]) -> None:
    """
    Traced replacement of the sub-block call execution.

    Args:
        self (SchemaInterpreter): The interpreter instance.
        instruction (BlockInstruction): The block call being executed.
        block (ModelBlockSchema): The block owning the call.
        scope (dict[str, Any]): The current scope.
    """

    # Resolve the called sub-block for the description, run then record
    tracer: TraceRecorder | None = _ACTIVE_TRACER
    sub_block: ModelBlockSchema | None = None
    if instruction.block_ref.index_var is not None:
        bound: Any = scope.get(instruction.block_ref.index_var)
        if bound is not None and not isinstance(bound, ModelBlockSchema):
            # A skipped container entry bound to its real module
            sub_block = None
        else:
            sub_block = self._get_sub_block(
                block=block,
                block_ref=instruction.block_ref,
                scope=scope
            )
    else:
        sub_block = self._get_sub_block(
            block=block,
            block_ref=instruction.block_ref,
            scope=scope
        )
    _ORIG_BLOCK(self, instruction=instruction, block=block, scope=scope)
    if tracer is None:
        return
    try:
        tracer.log_block_call(
            instruction=instruction,
            sub_block=sub_block,
            scope=scope
        )
    except Exception:  # noqa: BLE001 - the trace must never break the run
        pass


def _traced_control(self: SchemaInterpreter, instruction: ControlInstruction,
                    block: ModelBlockSchema, scope: dict[str, Any]) -> Any:
    """
    Traced replacement of the control flow execution.

    Args:
        self (SchemaInterpreter): The interpreter instance.
        instruction (ControlInstruction): The control flow being executed.
        block (ModelBlockSchema): The block owning the control flow.
        scope (dict[str, Any]): The current scope.

    Returns:
        Any: The propagated return of the control flow, if any.
    """

    # Record the entry then run the control flow
    tracer: TraceRecorder | None = _ACTIVE_TRACER
    if tracer is not None:
        try:
            tracer.log_control(instruction=instruction)
            tracer.depth += 1
        except Exception:  # noqa: BLE001
            pass
    result: Any = _ORIG_CONTROL(self, instruction=instruction, block=block,
                                scope=scope)
    if tracer is not None:
        tracer.depth = max(0, tracer.depth - 1)
    return result


def _traced_try(self: SchemaInterpreter, instruction: TryInstruction,
                block: ModelBlockSchema, scope: dict[str, Any]) -> Any:
    """
    Traced replacement of the try execution.

    Args:
        self (SchemaInterpreter): The interpreter instance.
        instruction (TryInstruction): The try being executed.
        block (ModelBlockSchema): The block owning the try.
        scope (dict[str, Any]): The current scope.

    Returns:
        Any: The propagated return of the try, if any.
    """

    # Record the entry then run the try
    tracer: TraceRecorder | None = _ACTIVE_TRACER
    if tracer is not None:
        try:
            tracer.log_try()
            tracer.depth += 1
        except Exception:  # noqa: BLE001
            pass
    result: Any = _ORIG_TRY(self, instruction=instruction, block=block,
                            scope=scope)
    if tracer is not None:
        tracer.depth = max(0, tracer.depth - 1)
    return result


def install_tracer(tracer: TraceRecorder) -> None:
    """
    Install a tracer on the schema interpreter.

    Args:
        tracer (TraceRecorder): The recorder receiving the trace lines.
    """

    # Store the active tracer and patch the interpreter methods
    global _ACTIVE_TRACER
    _ACTIVE_TRACER = tracer
    SchemaInterpreter.run = _traced_run  # type: ignore[method-assign]
    SchemaInterpreter._execute_op = _traced_op  # type: ignore[method-assign]
    SchemaInterpreter._execute_block_call = _traced_block_call  # type: ignore[method-assign]
    SchemaInterpreter._execute_control = _traced_control  # type: ignore[method-assign]
    SchemaInterpreter._execute_try = _traced_try  # type: ignore[method-assign]


def uninstall_tracer() -> None:
    """Restore the original interpreter methods."""

    # Restore the originals and drop the active tracer
    global _ACTIVE_TRACER
    _ACTIVE_TRACER = None
    SchemaInterpreter.run = _ORIG_RUN  # type: ignore[method-assign]
    SchemaInterpreter._execute_op = _ORIG_OP  # type: ignore[method-assign]
    SchemaInterpreter._execute_block_call = _ORIG_BLOCK  # type: ignore[method-assign]
    SchemaInterpreter._execute_control = _ORIG_CONTROL  # type: ignore[method-assign]
    SchemaInterpreter._execute_try = _ORIG_TRY  # type: ignore[method-assign]
