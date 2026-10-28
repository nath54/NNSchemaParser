"""
Schema interpreter of NNViewer.

Executes the body of a ModelBlockSchema instruction by instruction, by
re-dispatching the recorded calls to the real torch functions. It is used in
two modes:

- parse time, under FakeTensorMode, to resolve the shapes and dtypes;
- WebUI time, with real tensors, to capture every intermediate value.
"""

# Import Modules
from typing import Any

import inspect

from src.core.model_schema import ModelBlockSchema
from src.core.instructions import (
    BlockInstruction,
    BlockRef,
    ControlInstruction,
    Instruction,
    OpInstruction,
    ReturnInstruction,
    TryInstruction,
    VarRef
)
from src.model_parser.interpreter.control_flow import ControlFlowExecutor
from src.model_parser.interpreter.execution import InstructionExecutor


class SchemaInterpreter(InstructionExecutor, ControlFlowExecutor):
    """
    Executes the body of a ModelBlockSchema block.
    """

    def __init__(self, capture_values: bool = False) -> None:
        """
        Init function.

        Args:
            capture_values (bool): If True, every tensor result is kept as a
                TensorReprWithValue inside the instruction resolved field.
        """

        # Store here whether the intermediate values must be captured
        self.capture_values: bool = capture_values

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
                return result

        # If no return instruction was found, the block returns nothing
        return []

    def _seed_weights(self, block: ModelBlockSchema, scope: dict[str, Any]) -> None:
        """
        Seed the scope with the weights of the block.

        Args:
            block (ModelBlockSchema): The block schema being executed.
            scope (dict[str, Any]): The initial variables of the block.
        """

        # Each weight is bound by its name in the block scope
        weight_name: str
        weight: Any
        for weight_name, weight in block.weights.items():
            scope.setdefault(weight_name, weight.value)

    def _seed_forward_defaults(self, block: ModelBlockSchema, scope: dict[str, Any]) -> None:
        """
        Seed the scope with the default values of the forward arguments.

        Args:
            block (ModelBlockSchema): The block schema being executed.
            scope (dict[str, Any]): The initial variables of the block.
        """

        # Get the signature of the forward method of the block
        signature: inspect.Signature = inspect.signature(block.module.forward)
        parameter_name: str
        parameter: inspect.Parameter
        for parameter_name, parameter in signature.parameters.items():

            # Skip the self parameter of the forward method
            if parameter_name == "self":
                continue

            # Skip the arguments already provided by the caller
            if parameter_name in scope:
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

        # Dispatch to the execution of the specific instruction type,
        # propagating a return instruction reached in a nested body
        if isinstance(instruction, OpInstruction):
            self._execute_op(instruction=instruction, scope=scope)

        elif isinstance(instruction, BlockInstruction):
            self._execute_block_call(
                instruction=instruction,
                block=block,
                scope=scope
            )

        elif isinstance(instruction, TryInstruction):
            return self._execute_try(
                instruction=instruction,
                block=block,
                scope=scope
            )

        elif isinstance(instruction, ControlInstruction):
            return self._execute_control(
                instruction=instruction,
                block=block,
                scope=scope
            )

        # A return instruction produces the outputs of the block
        elif isinstance(instruction, ReturnInstruction):
            return self._execute_return(
                instruction=instruction,
                block=block,
                scope=scope
            )

        else:
            raise TypeError(
                f"Unsupported instruction type: {type(instruction).__name__}"
            )

        # No return instruction was executed
        return None

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

        # The sub-block is a container entry, its key is "path.index"
        entry_key: str = f"{block_ref.path}.{index}"

        # Get the container entry from the computed key
        return block.sub_blocks[entry_key]

