"""
Execution of the control flow instructions of a block schema.

Covers the if statements, the for loops over iterables and over sub-block
containers, and the try statements with their exception handlers.
"""

# Import Modules
from typing import Any

import torch.nn as nn

from src.core.instructions import ControlInstruction, Instruction, TryInstruction
from src.core.model_schema import ModelBlockSchema
from src.model_parser.container_keys import (
    container_entry_keys,
    container_entry_sort_key
)


class ControlFlowExecutor:
    """
    Executes the control flow instructions of a block schema.

    The concrete interpreter inherits this class and provides the instruction
    dispatch and the execution of the operations and block calls.
    """

    def _execute_control(
        self,
        instruction: ControlInstruction,
        block: ModelBlockSchema,
        scope: dict[str, Any]
    ) -> list[Any] | None:
        """
        Execute a control flow instruction, an if statement or a for loop.

        Args:
            instruction (ControlInstruction): The control flow instruction.
            block (ModelBlockSchema): The block schema owning the instruction.
            scope (dict[str, Any]): The current variables of the block.

        Returns:
            list[Any] | None: The outputs of a return instruction reached in
                the executed body, else None.
        """

        # An if statement executes the body matching its condition value
        if instruction.kind == "if":
            return self._execute_if(
                instruction=instruction,
                block=block,
                scope=scope
            )

        # A for loop iterates over the values of its iterable expression
        if instruction.kind == "for":
            return self._execute_for(
                instruction=instruction,
                block=block,
                scope=scope
            )

        # Any other control flow kind is an error of the schema
        raise ValueError(f"Unsupported control flow kind: {instruction.kind}")

    def _execute_if(
        self,
        instruction: ControlInstruction,
        block: ModelBlockSchema,
        scope: dict[str, Any]
    ) -> list[Any] | None:
        """
        Execute an if instruction by running the body matching its condition.

        Args:
            instruction (ControlInstruction): The if instruction to execute.
            block (ModelBlockSchema): The block schema owning the instruction.
            scope (dict[str, Any]): The current variables of the block.

        Returns:
            list[Any] | None: The outputs of a return instruction reached in
                the executed body, else None.
        """

        # The condition of an if statement must be defined
        if instruction.condition is None:
            raise ValueError("An if instruction must define a condition.")

        # Evaluate the condition expression of the if statement
        self._execute_op(instruction=instruction.condition, scope=scope)
        condition_value: Any = scope[instruction.condition.out_var]

        # Execute the body matching the condition value and propagate
        # a return instruction reached in the executed body
        if condition_value:
            return self._run_body(body=instruction.body, block=block, scope=scope)

        # Else the else body of the if statement is executed
        return self._run_body(body=instruction.else_body, block=block, scope=scope)

    def _execute_for(
        self,
        instruction: ControlInstruction,
        block: ModelBlockSchema,
        scope: dict[str, Any]
    ) -> list[Any] | None:
        """
        Execute a for instruction over its iterable or container.

        Args:
            instruction (ControlInstruction): The for instruction to execute.
            block (ModelBlockSchema): The block schema owning the instruction.
            scope (dict[str, Any]): The current variables of the block.

        Returns:
            list[Any] | None: The outputs of a return instruction reached in
                the loop body, else None.
        """

        # The loop variable name must be defined to bind each value
        if instruction.bind_var is None:
            raise ValueError("A for loop instruction must define a bind_var.")

        # A for loop over a sub-block container binds the sub-block schemas
        if instruction.iter_container is not None:
            return self._execute_container_loop(
                instruction=instruction,
                block=block,
                scope=scope
            )

        # Else the iterable expression defines the values of the loop
        if instruction.condition is not None:
            return self._execute_iterable_loop(
                instruction=instruction,
                block=block,
                scope=scope
            )

        # A for loop must define a condition or a container to iterate
        raise ValueError(
            "A for loop instruction must define a condition or a container."
        )

    def _execute_iterable_loop(
        self,
        instruction: ControlInstruction,
        block: ModelBlockSchema,
        scope: dict[str, Any]
    ) -> list[Any] | None:
        """
        Execute a for loop over the values of its iterable expression.

        Args:
            instruction (ControlInstruction): The for instruction to execute.
            block (ModelBlockSchema): The block schema owning the instruction.
            scope (dict[str, Any]): The current variables of the block.

        Returns:
            list[Any] | None: The outputs of a return instruction reached in
                the loop body, else None.
        """

        # Evaluate the iterable expression of the loop
        self._execute_op(instruction=instruction.condition, scope=scope)
        iterable: Any = scope[instruction.condition.out_var]

        # Execute the loop body once per value of the iterable
        value: Any
        for value in iterable:

            # Bind the loop variable to the current iteration value
            scope[instruction.bind_var] = value

            # Execute the body of the loop for the current iteration and
            # propagate a return instruction reached in the body
            result: list[Any] | None = self._run_body(
                body=instruction.body,
                block=block,
                scope=scope
            )
            if result is not None:
                return result

        # No return instruction was reached in the loop body
        return None

    def _execute_container_loop(
        self,
        instruction: ControlInstruction,
        block: ModelBlockSchema,
        scope: dict[str, Any]
    ) -> list[Any] | None:
        """
        Execute a for loop over the entries of a sub-block container.

        Args:
            instruction (ControlInstruction): The container for loop instruction.
            block (ModelBlockSchema): The block schema owning the instruction.
            scope (dict[str, Any]): The current variables of the block.

        Returns:
            list[Any] | None: The outputs of a return instruction reached in
                the loop body, else None.
        """

        # Get the ordered entry keys of the iterated container
        container_path: str = instruction.iter_container.path

        # The special self path iterates over all the sub-blocks of the block
        if container_path == "$self":
            entry_keys: list[str] = sorted(
                block.sub_blocks.keys(),
                key=container_entry_sort_key
            )

        # Else the entry keys are the prefixed keys of the container
        else:
            entry_keys = container_entry_keys(
                sub_block_names=block.sub_blocks,
                container_name=container_path
            )

            # The entries skipped at parse time are missing from the parsed
            # sub-blocks: the real container is iterated to include them
            real_container: Any | None = None
            if block.module is not None:
                real_container = getattr(block.module, container_path, None)
            if isinstance(real_container, (nn.ModuleList, nn.Sequential)):
                missing_index: int
                for missing_index in range(len(real_container)):
                    missing_key: str = f"{container_path}.{missing_index}"
                    if missing_key not in entry_keys:
                        entry_keys.append(missing_key)
                entry_keys.sort(key=container_entry_sort_key)
            elif isinstance(real_container, nn.ModuleDict):
                missing_key_entry: str
                for missing_key_entry in real_container.keys():
                    missing_key = f"{container_path}.{missing_key_entry}"
                    if missing_key not in entry_keys:
                        entry_keys.append(missing_key)
                entry_keys.sort(key=container_entry_sort_key)

        # Execute the loop body once per entry of the container
        entry_index: int
        entry_key: str
        for entry_index, entry_key in enumerate(entry_keys):

            # Bind the loop variable to the entry sub-block schema, or to
            # the real module for an entry skipped at parse time
            parsed_entry: ModelBlockSchema | None = block.sub_blocks.get(entry_key)
            if parsed_entry is not None:
                scope[instruction.bind_var] = parsed_entry
            else:
                entry_name: str = entry_key.rsplit(".", 1)[1]
                scope[instruction.bind_var] = real_container[
                    int(entry_name) if entry_name.isdigit() else entry_name
                ]

            # Bind the enumerate index variable when defined
            if instruction.index_var is not None:
                scope[instruction.index_var] = entry_index

            # Execute the body of the loop for the current iteration and
            # propagate a return instruction reached in the loop body
            result: list[Any] | None = self._run_body(
                body=instruction.body,
                block=block,
                scope=scope
            )
            if result is not None:
                return result

        # No return instruction was reached in the loop body
        return None

    def _execute_try(
        self,
        instruction: TryInstruction,
        block: ModelBlockSchema,
        scope: dict[str, Any]
    ) -> list[Any] | None:
        """
        Execute a try instruction with its exception handlers and finally.

        Args:
            instruction (TryInstruction): The try instruction to execute.
            block (ModelBlockSchema): The block schema owning the instruction.
            scope (dict[str, Any]): The current variables of the block.

        Returns:
            list[Any] | None: The outputs of a return instruction reached in
                the body or a matched handler, else None.
        """

        # Execute the body of the try instruction
        try:
            return self._run_body(body=instruction.body, block=block, scope=scope)

        # Execute the first matching handler of the caught exception
        except Exception as error:

            # Search for a handler matching the type of the caught error
            handler_index: int
            handler_types: tuple[type[BaseException], ...]
            handler_body: list[Instruction]
            for handler_index, (handler_types, handler_body) in enumerate(
                instruction.handlers
            ):

                # Run the body of the first handler matching the error type
                if isinstance(error, handler_types):

                    # Bind the caught exception when the handler names it
                    handler_name: str | None = instruction.handler_names[handler_index]
                    if handler_name is not None:
                        scope[handler_name] = error

                    # Run the body of the matched handler and propagate
                    # a return instruction reached in the handler body
                    return self._run_body(
                        body=handler_body,
                        block=block,
                        scope=scope
                    )

            # Re-raise the error when no handler matches its type
            else:
                raise

        # Execute the finally body after the try or the matched handler
        finally:
            self._run_body(
                body=instruction.final_body,
                block=block,
                scope=scope
            )

    def _run_body(
        self,
        body: list[Instruction],
        block: ModelBlockSchema,
        scope: dict[str, Any]
    ) -> list[Any] | None:
        """
        Execute a list of instructions of a control flow body.

        Args:
            body (list[Instruction]): The instructions to execute.
            block (ModelBlockSchema): The block schema owning the instructions.
            scope (dict[str, Any]): The current variables of the block.

        Returns:
            list[Any] | None: The outputs of a return instruction reached in
                the body, else None.
        """

        # Execute all the instructions of the body in order
        body_instruction: Instruction
        for body_instruction in body:

            # Execute the current instruction of the body
            result: list[Any] | None = self._execute_instruction(
                instruction=body_instruction,
                block=block,
                scope=scope
            )

            # A returned result stops the execution of the body
            if result is not None:
                return result

        # No return instruction was reached in the body
        return None
