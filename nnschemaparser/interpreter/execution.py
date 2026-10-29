"""
Execution of the operation and block instructions of a block schema.

Covers the operation calls re-dispatched to the real torch functions, the
sub-block calls recursively executed, the returns and the resolution of the
placeholder tensor representations of the reached blocks.
"""

# Import Modules
from typing import Any

import inspect

from torch import Tensor

import torch.nn as nn

from nnschemaparser.core.instructions import (
    BlockInstruction,
    OpInstruction,
    ReturnInstruction,
    VarRef
)
from nnschemaparser.core.model_schema import ModelBlockSchema
from nnschemaparser.core.tensor_schema import TensorRepr, TensorReprWithValue
from nnschemaparser.forward import resolve_forward
from nnschemaparser.interpreter.scope import (
    PLACEHOLDER_DTYPE,
    resolve_placeholder
)
from nnschemaparser.interpreter.tracer import TraceListener


class InstructionExecutor:
    """
    Executes the operation and block instructions of a block schema.

    The concrete interpreter inherits this class and adds the execution of the
    control flow instructions on top of it.
    """

    def _notify_op(self, instruction: OpInstruction, scope: dict[str, Any]) -> None:
        """
        Notify every listener of an executed operation instruction.

        Args:
            instruction (OpInstruction): The executed operation.
            scope (dict[str, Any]): The scope holding the result.
        """

        # Dispatch the event to all the registered listeners
        listener: TraceListener
        for listener in self.listeners:
            listener.on_op(instruction=instruction, scope=scope)

    def _notify_block_call(
        self,
        instruction: BlockInstruction,
        sub_block: ModelBlockSchema | None,
        scope: dict[str, Any]
    ) -> None:
        """
        Notify every listener of an executed sub-block call instruction.

        Args:
            instruction (BlockInstruction): The executed block call.
            sub_block (ModelBlockSchema | None): The called sub-block, None
                when the call was re-dispatched to a real module.
            scope (dict[str, Any]): The scope holding the result.
        """

        # Dispatch the event to all the registered listeners
        listener: TraceListener
        for listener in self.listeners:
            listener.on_block_call(
                instruction=instruction,
                sub_block=sub_block,
                scope=scope
            )

    def _execute_op(self, instruction: OpInstruction, scope: dict[str, Any]) -> None:
        """
        Execute an operation instruction by calling its recorded callable.

        Args:
            instruction (OpInstruction): The operation instruction to execute.
            scope (dict[str, Any]): The current variables of the block.
        """

        # Resolve all the positional arguments of the call from the scope
        resolved_args: list[Any] = [
            self._resolve_value(value=arg, scope=scope) for arg in instruction.args
        ]

        # Resolve all the keyword arguments of the call from the scope
        resolved_kwargs: dict[str, Any] = {
            key: self._resolve_value(value=value, scope=scope)
            for key, value in instruction.kwargs.items()
        }

        # Merge the variable unpacked as the extra keyword arguments
        if instruction.kwargs_var is not None:
            unpacked: Any = scope.get(instruction.kwargs_var)
            if unpacked:
                resolved_kwargs.update(unpacked)

        # Extend the positional arguments with the unpacked variable
        if instruction.args_var is not None:
            star_args: Any = scope.get(instruction.args_var)
            if star_args is not None:
                resolved_args.extend(star_args)

        # Resolve the target callable from the scope if it is a variable
        call_target: Any = instruction.target
        if isinstance(call_target, VarRef):
            call_target = self._resolve_value(value=call_target, scope=scope)

        # Call the recorded torch function with the resolved arguments
        result: Any = call_target(*resolved_args, **resolved_kwargs)

        # Bind the result to the output variable of the instruction
        scope[instruction.out_var] = result

        # Fill the resolved representation of the instruction
        instruction.resolve(value=result, capture_values=self.capture_values)

        # Notify the listeners of the executed operation
        self._notify_op(instruction=instruction, scope=scope)

    def _execute_block_call(
        self,
        instruction: BlockInstruction,
        block: ModelBlockSchema,
        scope: dict[str, Any]
    ) -> None:
        """
        Execute a sub-block call instruction by running the sub-block schema.

        Args:
            instruction (BlockInstruction): The block call instruction.
            block (ModelBlockSchema): The block schema owning the instruction.
            scope (dict[str, Any]): The current variables of the block.
        """

        # A loop variable bound to a real module is a skipped container
        # entry: the real module is called directly like an operation
        if instruction.block_ref.index_var is not None:
            bound: Any = scope.get(instruction.block_ref.index_var)
            if isinstance(bound, nn.Module):
                self._execute_real_module_call(
                    instruction=instruction,
                    module=bound,
                    scope=scope
                )
                self._notify_block_call(
                    instruction=instruction,
                    sub_block=None,
                    scope=scope
                )
                return

        # A container entry skipped at parse time is dispatched to its
        # real module through the parent of the block
        if instruction.block_ref.index_var is not None and \
                f"{instruction.block_ref.path}.{scope.get(instruction.block_ref.index_var)}" \
                not in block.sub_blocks and block.module is not None:
            container: Any = getattr(
                block.module, instruction.block_ref.path, None
            )
            index: Any = scope.get(instruction.block_ref.index_var)
            if container is not None:
                try:
                    entry: Any = container[index]
                except Exception:  # noqa: BLE001 - fall through to the lookup
                    entry = None
                if isinstance(entry, nn.Module):
                    self._execute_real_module_call(
                        instruction=instruction,
                        module=entry,
                        scope=scope
                    )
                    self._notify_block_call(
                        instruction=instruction,
                        sub_block=None,
                        scope=scope
                    )
                    return

        # Get the target sub-block schema from the reference
        sub_block: ModelBlockSchema = self._get_sub_block(
            block=block,
            block_ref=instruction.block_ref,
            scope=scope
        )

        # Build the sub scope of the sub-block from its declared inputs
        sub_scope: dict[str, Any] = self._build_sub_scope(
            instruction=instruction,
            sub_block=sub_block,
            scope=scope
        )

        # Recursively execute the body of the sub-block
        result: list[Any] = self.run(block=sub_block, scope=sub_scope)

        # Bind the outputs like python: a single output is bound directly
        # and several outputs are bound as a tuple
        if len(result) == 1:
            scope[instruction.out_var] = result[0]
        else:
            scope[instruction.out_var] = tuple(result)

        # Fill the resolved representation of the instruction
        if len(result) == 1:
            instruction.resolve(value=result[0], capture_values=self.capture_values)

        # Notify the listeners of the executed sub-block call
        self._notify_block_call(
            instruction=instruction,
            sub_block=sub_block,
            scope=scope
        )

    def _execute_real_module_call(
        self,
        instruction: BlockInstruction,
        module: nn.Module,
        scope: dict[str, Any]
    ) -> None:
        """
        Execute a block call bound to a real module of a skipped container
        entry, re-dispatching to the module like an operation.

        Args:
            instruction (BlockInstruction): The block call instruction.
            module (nn.Module): The real module bound to the loop variable.
            scope (dict[str, Any]): The current variables of the block.
        """

        # Resolve all the positional arguments of the call from the scope
        resolved_args: list[Any] = [
            self._resolve_value(value=arg, scope=scope) for arg in instruction.args
        ]

        # Resolve all the keyword arguments of the call from the scope
        resolved_kwargs: dict[str, Any] = {
            key: self._resolve_value(value=value, scope=scope)
            for key, value in instruction.kwargs.items()
        }

        # Call the real module with the resolved arguments
        result: Any = module(*resolved_args, **resolved_kwargs)

        # Bind the result to the output variable of the instruction
        scope[instruction.out_var] = result

        # Fill the resolved representation of the instruction
        instruction.resolve(value=result, capture_values=self.capture_values)

    def _build_sub_scope(
        self,
        instruction: BlockInstruction,
        sub_block: ModelBlockSchema,
        scope: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Build the sub scope of a sub-block call from its declared inputs.

        Args:
            instruction (BlockInstruction): The block call instruction.
            sub_block (ModelBlockSchema): The called sub-block schema.
            scope (dict[str, Any]): The current variables of the caller block.

        Returns:
            dict[str, Any]: The scope of the sub-block execution.
        """

        # Get the declared input names of the sub-block
        input_names: list[str] = list(sub_block.inputs.keys())

        # Resolve the positional arguments of the call, splat when needed
        positional_args: list[Any] = self._resolve_positional_args(
            instruction=instruction,
            scope=scope
        )

        # Map the declared input names to the positional arguments
        sub_scope: dict[str, Any] = {
            input_name: arg
            for input_name, arg in zip(input_names, positional_args)
        }

        # Bind the keyword arguments of the call to the sub-block inputs
        self._bind_call_kwargs(
            instruction=instruction,
            sub_block=sub_block,
            scope=scope,
            sub_scope=sub_scope
        )

        # Fill the missing arguments with the defaults of the sub-block forward
        # (only the declared inputs not bound positionally or by keyword, else
        # the defaults would silently overwrite the bound keyword arguments)
        bound_names: set[str] = set(sub_scope) | set(instruction.kwargs)
        missing_names: list[str] = [
            input_name
            for input_name in input_names
            if input_name not in bound_names
        ]
        if missing_names:
            sub_scope.update(
                self._sub_block_defaults(
                    sub_block=sub_block,
                    missing_names=missing_names
                )
            )

        # Return the built sub scope of the sub-block
        return sub_scope

    def _resolve_positional_args(
        self,
        instruction: BlockInstruction,
        scope: dict[str, Any]
    ) -> list[Any]:
        """
        Resolve the positional arguments of a sub-block call.

        Args:
            instruction (BlockInstruction): The block call instruction.
            scope (dict[str, Any]): The current variables of the caller block.

        Returns:
            list[Any]: The resolved positional arguments of the call.
        """

        # A splat call unpacks the argument variable tuple as the arguments
        if instruction.splat_args:
            splat_value: Any = self._resolve_value(
                value=instruction.args[0],
                scope=scope
            )

            # A tuple result is unpacked as several arguments, a single result
            # is passed as the only argument
            return (
                list(splat_value) if isinstance(splat_value, tuple)
                else [splat_value]
            )

        # Else the declared arguments of the instruction are the arguments
        resolved: list[Any] = [
            self._resolve_value(value=arg, scope=scope)
            for arg in instruction.args
        ]

        # Extend the resolved arguments with the unpacked variable
        if instruction.args_var is not None:
            star_args: Any = scope.get(instruction.args_var)
            if star_args is not None:
                resolved.extend(star_args)

        # Return the resolved positional arguments of the call
        return resolved

    def _bind_call_kwargs(
        self,
        instruction: BlockInstruction,
        sub_block: ModelBlockSchema,
        scope: dict[str, Any],
        sub_scope: dict[str, Any]
    ) -> None:
        """
        Bind the keyword arguments of a sub-block call to its declared inputs.

        The explicit keyword arguments are merged with the variable unpacked
        as the extra keyword arguments (the **kwargs of the call). The keys
        matching a declared input are bound by name, the other keys are
        gathered into the var-keyword parameter of the sub-block forward.

        Args:
            instruction (BlockInstruction): The block call instruction.
            sub_block (ModelBlockSchema): The called sub-block schema.
            scope (dict[str, Any]): The current variables of the caller block.
            sub_scope (dict[str, Any]): The scope of the sub-block execution.

        Raises:
            KeyError: When a keyword argument matches no declared input and
                the sub-block forward has no var-keyword parameter.
        """

        # The explicit keyword arguments of the call resolved from the scope
        resolved: dict[str, Any] = {
            key: self._resolve_value(value=value, scope=scope)
            for key, value in instruction.kwargs.items()
        }

        # The variable unpacked as the extra keyword arguments, if any
        if instruction.kwargs_var is not None:
            unpacked: Any = scope.get(instruction.kwargs_var)
            if unpacked:
                resolved.update(unpacked)

        # The name of the var-keyword parameter of the sub-block forward
        var_kw_name: str | None = self._var_keyword_name(sub_block=sub_block)

        # Bind each keyword argument to the declared input or the extra dict
        extra: dict[str, Any] = {}
        key: str
        value: Any
        for key, value in resolved.items():

            # The declared inputs are bound by name
            if key in sub_block.inputs:
                sub_scope[key] = value
            elif var_kw_name is not None:
                extra[key] = value
            else:
                raise KeyError(
                    f"Unknown keyword argument for sub-block: {key}"
                )

        # Bind the var-keyword dict of the sub-block forward, even when
        # empty, so the body references of the sub-block find the variable
        if var_kw_name is not None:
            sub_scope[var_kw_name] = extra

    def _var_keyword_name(self, sub_block: ModelBlockSchema) -> str | None:
        """
        Get the name of the var-keyword parameter of a sub-block forward.

        Args:
            sub_block (ModelBlockSchema): The called sub-block schema.

        Returns:
            str | None: The name of the **kwargs parameter, or None when the
                sub-block has no module or no var-keyword parameter.
        """

        # Without a source module there is no signature to inspect
        if sub_block.module is None:
            return None

        # Find the var-keyword parameter of the forward method
        parameter: inspect.Parameter
        for name, parameter in inspect.signature(
            resolve_forward(module=sub_block.module)
        ).parameters.items():
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                return name
        return None

    def _execute_return(
        self,
        instruction: ReturnInstruction,
        block: ModelBlockSchema,
        scope: dict[str, Any]
    ) -> list[Any]:
        """
        Execute a return instruction and resolve the block outputs.

        Args:
            instruction (ReturnInstruction): The return instruction.
            block (ModelBlockSchema): The block schema owning the instruction.
            scope (dict[str, Any]): The current variables of the block.

        Returns:
            list[Any]: The values returned by the block.
        """

        # Resolve the returned values from the scope
        outputs: list[Any] = [
            self._resolve_value(value=VarRef(name=out_var), scope=scope)
            for out_var in instruction.out_vars
        ]

        # Resolve the placeholder representations of the block outputs
        self._resolve_block_outputs(block=block, values=outputs)

        # Return the values of the block
        return outputs

    def _resolve_block_inputs(self, block: ModelBlockSchema, scope: dict[str, Any]) -> None:
        """
        Fill the placeholder shape and dtype of the block inputs from the scope.

        Args:
            block (ModelBlockSchema): The block schema being executed.
            scope (dict[str, Any]): The current variables of the block.
        """

        # Check each declared input of the block
        input_name: str
        input_repr: TensorRepr
        for input_name, input_repr in block.inputs.items():

            # Get the value of the input from the scope
            value: Any = scope.get(input_name)

            # Only tensor values fill the representation
            if not isinstance(value, Tensor):
                continue

            # Fill the placeholder shape and dtype of the input
            resolve_placeholder(repr=input_repr, value=value)

    def _resolve_block_outputs(self, block: ModelBlockSchema, values: list[Any]) -> None:
        """
        Fill the placeholder shape and dtype of the block outputs from the values.

        Args:
            block (ModelBlockSchema): The block schema being executed.
            values (list[Any]): The values returned by the block execution.
        """

        # Check each returned value against its output representation
        index: int
        value: Any
        for index, value in enumerate(values):

            # The outputs of the block only cover the known return count
            if index >= len(block.outputs):
                continue

            # Only placeholder representations are updated
            output_repr: TensorRepr = block.outputs[index]
            if output_repr.shape != () or output_repr.dtype != PLACEHOLDER_DTYPE:
                continue

            # Tensor values fill the shape and the dtype of the representation
            if isinstance(value, Tensor):
                output_repr.shape = tuple(int(dim) for dim in value.shape)
                output_repr.dtype = str(value.dtype)

            # The other values (model output containers, tuples, python
            # scalars) are described by their type name
            else:
                output_repr.dtype = type(value).__name__

    def _sub_block_defaults(
        self,
        sub_block: ModelBlockSchema,
        missing_names: list[str]
    ) -> dict[str, Any]:
        """
        Get the default values of the missing arguments of a sub-block call.

        Args:
            sub_block (ModelBlockSchema): The called sub-block schema.
            missing_names (list[str]): The names of the missing arguments.

        Returns:
            dict[str, Any]: The default values by argument name.
        """

        # Without a source module, no default is available
        if sub_block.module is None:
            return {}

        # Get the signature of the sub-block forward method
        signature: inspect.Signature = inspect.signature(
            resolve_forward(module=sub_block.module)
        )

        # Collect the defaults of the missing arguments
        defaults: dict[str, Any] = {}
        missing_name: str
        for missing_name in missing_names:

            # Only the parameters of the forward method provide defaults
            if missing_name not in signature.parameters:
                continue

            # Only the parameters with a default value provide a default
            parameter: inspect.Parameter = signature.parameters[missing_name]
            if parameter.default is not inspect.Parameter.empty:
                defaults[missing_name] = parameter.default

        # Return the collected defaults
        return defaults
