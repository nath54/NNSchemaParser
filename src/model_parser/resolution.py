"""
Shape and dtype resolution of a parsed ModelBlockSchema.

The resolution executes the schema once: under a fake tensor mode to avoid any
memory usage, or with real tensors when the model uses data dependent python
code or when example input values are given.
"""

# Import Modules
from typing import Any

import torch
from torch import Tensor
from torch._subclasses.fake_tensor import FakeTensorMode

from src.core.model_schema import ModelBlockSchema
from src.model_parser.interpreter import SchemaInterpreter
from src.model_parser.interpreter.scope import forward_signature_defaults


class ShapeResolver:
    """
    Resolves the shapes and dtypes of a parsed block schema by execution.
    """

    @staticmethod
    def resolve(
        block: ModelBlockSchema,
        inputs_tensors_shapes: dict[str, tuple[int, ...]],
        inputs_tensors_dtypes: dict[str, str],
        inputs_tensors_values: dict[str, Tensor] | None
    ) -> None:
        """
        Execute the block schema to resolve shapes and dtypes.

        A first execution is done under a fake tensor mode to avoid any memory
        usage. If the model uses data dependent python code (e.g. tensor.item),
        the resolution falls back to a real execution without gradients, using
        the example values of the inputs when they are given.

        The execution fills the resolved representations of all the
        instructions, and the inputs and outputs of all the reached blocks.

        Args:
            block (ModelBlockSchema): The block schema to resolve.
            inputs_tensors_shapes (dict[str, tuple[int, ...]]): The shapes of
                the input tensors of the forward method.
            inputs_tensors_dtypes (dict[str, str]): The torch dtypes of the
                input tensors of the forward method.
            inputs_tensors_values (dict[str, Tensor] | None): The optional
                example values of the input tensors of the forward method.
        """

        # The example values allow a direct real execution of the schema
        if inputs_tensors_values is not None:
            ShapeResolver._resolve_with_values(
                block=block,
                inputs_tensors_shapes=inputs_tensors_shapes,
                inputs_tensors_dtypes=inputs_tensors_dtypes,
                inputs_tensors_values=inputs_tensors_values
            )
            return

        # A first resolution is attempted under a fake tensor mode
        try:
            ShapeResolver._resolve_with_fake_mode(
                block=block,
                inputs_tensors_shapes=inputs_tensors_shapes,
                inputs_tensors_dtypes=inputs_tensors_dtypes
            )

        # Else the resolution falls back to a real execution without gradients
        except Exception:
            ShapeResolver._resolve_with_real_tensors(
                block=block,
                inputs_tensors_shapes=inputs_tensors_shapes,
                inputs_tensors_dtypes=inputs_tensors_dtypes
            )

    @staticmethod
    def _resolve_with_values(
        block: ModelBlockSchema,
        inputs_tensors_shapes: dict[str, tuple[int, ...]],
        inputs_tensors_dtypes: dict[str, str],
        inputs_tensors_values: dict[str, Tensor]
    ) -> None:
        """
        Resolve the shapes with a real execution on the example values.

        Args:
            block (ModelBlockSchema): The block schema to resolve.
            inputs_tensors_shapes (dict[str, tuple[int, ...]]): The shapes of
                the input tensors of the forward method.
            inputs_tensors_dtypes (dict[str, str]): The torch dtypes of the
                input tensors of the forward method.
            inputs_tensors_values (dict[str, Tensor]): The example values of
                the input tensors of the forward method.
        """

        # Build the scope of the resolution from the example values
        scope: dict[str, Any] = ShapeResolver._input_scope(
            inputs_tensors_shapes=inputs_tensors_shapes,
            inputs_tensors_dtypes=inputs_tensors_dtypes
        )
        input_name: str
        input_value: Tensor
        for input_name, input_value in inputs_tensors_values.items():
            scope[input_name] = input_value.detach().clone()

        # Seed the defaults of the forward arguments without a given value,
        # without overwriting the example values by the default None values
        defaults: dict[str, Any] = forward_signature_defaults(
            module=block.module
        )
        default_name: str
        default_value: Any
        for default_name, default_value in defaults.items():
            scope.setdefault(default_name, default_value)

        # Execute the schema with the real input tensors
        with torch.no_grad():
            interpreter: SchemaInterpreter = SchemaInterpreter(
                capture_values=False
            )
            interpreter.run(block=block, scope=scope)

    @staticmethod
    def _resolve_with_fake_mode(
        block: ModelBlockSchema,
        inputs_tensors_shapes: dict[str, tuple[int, ...]],
        inputs_tensors_dtypes: dict[str, str]
    ) -> None:
        """
        Resolve the shapes with an execution under a fake tensor mode.

        Args:
            block (ModelBlockSchema): The block schema to resolve.
            inputs_tensors_shapes (dict[str, tuple[int, ...]]): The shapes of
                the input tensors of the forward method.
            inputs_tensors_dtypes (dict[str, str]): The torch dtypes of the
                input tensors of the forward method.

        Raises:
            Exception: When the schema execution fails under the fake mode.
        """

        # Execute the schema with fake input tensors and real module weights
        with FakeTensorMode(allow_non_fake_inputs=True):
            scope: dict[str, Any] = ShapeResolver._build_scope(
                block=block,
                inputs_tensors_shapes=inputs_tensors_shapes,
                inputs_tensors_dtypes=inputs_tensors_dtypes
            )
            interpreter: SchemaInterpreter = SchemaInterpreter(
                capture_values=False
            )
            interpreter.run(block=block, scope=scope)

    @staticmethod
    def _resolve_with_real_tensors(
        block: ModelBlockSchema,
        inputs_tensors_shapes: dict[str, tuple[int, ...]],
        inputs_tensors_dtypes: dict[str, str]
    ) -> None:
        """
        Resolve the shapes with a real execution on neutral input tensors.

        Args:
            block (ModelBlockSchema): The block schema to resolve.
            inputs_tensors_shapes (dict[str, tuple[int, ...]]): The shapes of
                the input tensors of the forward method.
            inputs_tensors_dtypes (dict[str, str]): The torch dtypes of the
                input tensors of the forward method.
        """

        # Build the scope of the resolution with neutral input tensors
        scope: dict[str, Any] = ShapeResolver._build_scope(
            block=block,
            inputs_tensors_shapes=inputs_tensors_shapes,
            inputs_tensors_dtypes=inputs_tensors_dtypes
        )

        # Execute the schema with the real input tensors
        with torch.no_grad():
            interpreter: SchemaInterpreter = SchemaInterpreter(
                capture_values=False
            )
            interpreter.run(block=block, scope=scope)

    @staticmethod
    def _build_scope(
        block: ModelBlockSchema,
        inputs_tensors_shapes: dict[str, tuple[int, ...]],
        inputs_tensors_dtypes: dict[str, str]
    ) -> dict[str, Any]:
        """
        Build the input scope of a resolution pass.

        Args:
            block (ModelBlockSchema): The block schema to resolve.
            inputs_tensors_shapes (dict[str, tuple[int, ...]]): The shapes of
                the input tensors of the forward method.
            inputs_tensors_dtypes (dict[str, str]): The torch dtypes of the
                input tensors of the forward method.

        Returns:
            dict[str, Any]: The scope of the input tensors of the forward.
        """

        # Create the input tensors of the resolution pass
        scope: dict[str, Any] = ShapeResolver._input_scope(
            inputs_tensors_shapes=inputs_tensors_shapes,
            inputs_tensors_dtypes=inputs_tensors_dtypes
        )

        # Seed the defaults of the forward arguments without a given shape,
        # without overwriting the input tensors by the default None values
        defaults: dict[str, Any] = forward_signature_defaults(
            module=block.module
        )
        default_name: str
        default_value: Any
        for default_name, default_value in defaults.items():
            scope.setdefault(default_name, default_value)

        # Return the scope of the resolution pass
        return scope

    @staticmethod
    def _input_scope(
        inputs_tensors_shapes: dict[str, tuple[int, ...]],
        inputs_tensors_dtypes: dict[str, str]
    ) -> dict[str, Any]:
        """
        Create the input tensors of a resolution pass.

        Args:
            inputs_tensors_shapes (dict[str, tuple[int, ...]]): The shapes of
                the input tensors of the forward method.
            inputs_tensors_dtypes (dict[str, str]): The torch dtypes of the
                input tensors of the forward method.

        Returns:
            dict[str, Any]: The scope of the input tensors of the forward.
        """

        # Create an input tensor per declared shape
        scope: dict[str, Any] = {}
        parameter_name: str
        shape: tuple[int, ...]
        for parameter_name, shape in inputs_tensors_shapes.items():
            dtype_name: str = inputs_tensors_dtypes.get(parameter_name, "float32")
            # Accept both the short ("float32") and the full ("torch.float32") names
            dtype_name = dtype_name.split(".")[-1]
            scope[parameter_name] = torch.empty(
                size=shape,
                dtype=getattr(torch, dtype_name)
            )

        # Return the scope of the input tensors
        return scope
