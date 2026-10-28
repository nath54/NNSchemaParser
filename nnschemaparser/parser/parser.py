"""
Parser converting torch nn.Module models into the NNSchemaParser ModelBlockSchema.

The parser walks a module recursively, lowers the AST of each forward method
into instructions, then delegates the shape and dtype resolution to the
ShapeResolver.
"""

# Import Modules
from typing import Any

import inspect

from torch import Tensor, nn

from nnschemaparser.core.instructions import Instruction
from nnschemaparser.core.model_schema import ModelBlockSchema
from nnschemaparser.core.tensor_schema import TensorRepr, TensorReprWithValue
from nnschemaparser.parser.children import collect_sub_blocks
from nnschemaparser.forward import resolve_forward
from nnschemaparser.parser.lowering import BodyLowering
from nnschemaparser.parser.lowering.errors import UnsupportedSyntaxError
from nnschemaparser.parser.resolution import ShapeResolver


class NNParser:
    """
    Parser converting torch nn.Module models into ModelBlockSchema.
    """

    @staticmethod
    def parse_model(
        model: nn.Module,
        inputs_tensors_shapes: dict[str, tuple[int, ...]],
        inputs_tensors_dtypes: dict[str, str] | None = None,
        inputs_tensors_values: dict[str, Tensor] | None = None
    ) -> ModelBlockSchema:
        """
        Parse a torch module into a ModelBlockSchema.

        Args:
            model (nn.Module): The model to parse.
            inputs_tensors_shapes (dict[str, tuple[int, ...]]): The shapes of
                the input tensors of the forward method.
            inputs_tensors_dtypes (dict[str, str] | None): The optional torch
                dtypes of the input tensors, float32 by default.
            inputs_tensors_values (dict[str, Tensor] | None): The optional
                example values of the input tensors, used by the resolution
                when the model has data dependent shapes.

        Returns:
            ModelBlockSchema: The parsed model schema.
        """

        # Check the forward signature for unsupported argument kinds
        signature: inspect.Signature = inspect.signature(
            resolve_forward(module=model)
        )
        NNParser._check_signature(signature=signature)

        # Collect the direct parameters and buffers of the model
        model_weights: dict[str, TensorReprWithValue] = NNParser._collect_weights(
            model=model
        )

        # Collect the sub-blocks and the container names of the model
        model_sub_blocks: dict[str, ModelBlockSchema]
        container_names: set[str]
        dict_container_names: set[str]
        model_sub_blocks, container_names, dict_container_names = collect_sub_blocks(
            model=model,
            parse_model=NNParser.parse_model
        )

        # Collect the plain python attributes of the module as literals
        literals: dict[str, Any] = NNParser._collect_literals(
            model=model,
            model_weights=model_weights
        )

        # Lower the body of the forward method into instructions
        body: list[Instruction]
        output_vars: list[str]
        body, output_vars = NNParser._lower_forward(
            model=model,
            sub_block_names=set(model_sub_blocks.keys()),
            container_names=container_names,
            dict_container_names=dict_container_names,
            weight_names=set(model_weights.keys()),
            literals=literals
        )

        # Build the schema of the model from its parsed parts
        schema: ModelBlockSchema = NNParser._build_schema(
            model=model,
            signature=signature,
            inputs_tensors_shapes=inputs_tensors_shapes,
            model_weights=model_weights,
            model_sub_blocks=model_sub_blocks,
            body=body,
            output_vars=output_vars
        )

        # Resolve the shapes and dtypes by executing the schema
        if inputs_tensors_shapes:
            ShapeResolver.resolve(
                block=schema,
                inputs_tensors_shapes=inputs_tensors_shapes,
                inputs_tensors_dtypes=inputs_tensors_dtypes or {},
                inputs_tensors_values=inputs_tensors_values
            )

        # Return the parsed model schema
        return schema

    @staticmethod
    def _check_signature(signature: inspect.Signature) -> None:
        """
        Check that the forward signature has no unsupported argument kind.

        Args:
            signature (inspect.Signature): The signature of the forward.

        Raises:
            UnsupportedSyntaxError: When the forward has variadic positional
                arguments.
        """

        # A variadic positional argument cannot be mapped to the declared
        # block inputs, while a variadic keyword argument is simply ignored:
        # it binds nothing in the body and receives no input value
        parameter: inspect.Parameter
        for parameter in signature.parameters.values():
            if parameter.kind == parameter.VAR_POSITIONAL:
                raise UnsupportedSyntaxError(
                    f"Unsupported argument kind in forward: {parameter.kind}",
                    signature
                )

    @staticmethod
    def _lower_forward(
        model: nn.Module,
        sub_block_names: set[str],
        container_names: set[str],
        dict_container_names: set[str],
        weight_names: set[str],
        literals: dict[str, Any]
    ) -> tuple[list[Instruction], list[str]]:
        """
        Lower the forward method of a model into its body instructions.

        Args:
            model (nn.Module): The model whose forward is lowered.
            sub_block_names (set[str]): The names of the direct sub-blocks.
            container_names (set[str]): The names of the sub-block containers.
            dict_container_names (set[str]): The names of the sub-block
                dictionaries, whose calls are re-dispatched at run time.
            weight_names (set[str]): The names of the parameters and buffers.
            literals (dict[str, Any]): The plain python attribute values.

        Returns:
            tuple[list[Instruction], list[str]]: The instructions of the body
                and the names of the returned variables.
        """

        # Create the lowering of the forward method
        lowering: BodyLowering = BodyLowering(
            sub_block_names=sub_block_names,
            container_names=container_names,
            dict_container_names=dict_container_names,
            weight_names=weight_names,
            literals=literals
        )

        # Lower the body of the forward method into instructions
        return lowering.lower_forward(model=model)

    @staticmethod
    def _build_schema(
        model: nn.Module,
        signature: inspect.Signature,
        inputs_tensors_shapes: dict[str, tuple[int, ...]],
        model_weights: dict[str, TensorReprWithValue],
        model_sub_blocks: dict[str, ModelBlockSchema],
        body: list[Instruction],
        output_vars: list[str]
    ) -> ModelBlockSchema:
        """
        Build the block schema of a model from its parsed parts.

        Args:
            model (nn.Module): The parsed model.
            signature (inspect.Signature): The signature of the forward.
            inputs_tensors_shapes (dict[str, tuple[int, ...]]): The shapes of
                the input tensors of the forward method.
            model_weights (dict[str, TensorReprWithValue]): The weights of the
                model collected by name.
            model_sub_blocks (dict[str, ModelBlockSchema]): The sub-blocks of
                the model collected by name.
            body (list[Instruction]): The lowered body of the forward.
            output_vars (list[str]): The names of the returned variables.

        Returns:
            ModelBlockSchema: The built schema of the model.
        """

        # Collect the inputs of the model from the forward signature
        model_inputs: dict[str, TensorRepr] = {}
        parameter_name: str
        for parameter_name in signature.parameters:

            # The self parameter is not an input of the model
            if parameter_name == "self":
                continue

            # The variadic keyword argument is not an input of the model
            if signature.parameters[parameter_name].kind == \
                    inspect.Parameter.VAR_KEYWORD:
                continue

            # The input shapes are unknown until the resolution pass
            model_inputs[parameter_name] = TensorRepr(
                shape=inputs_tensors_shapes.get(parameter_name, ()),
                dtype=""
            )

        # Collect the outputs of the model from the lowered return instruction
        model_outputs: list[TensorRepr] = [
            TensorRepr(shape=(), dtype="") for _ in output_vars
        ]

        # Return the schema of the model from its parsed parts
        return ModelBlockSchema(
            inputs=model_inputs,
            outputs=model_outputs,
            weights=model_weights,
            sub_blocks=model_sub_blocks,
            body=body,
            module=model
        )

    @staticmethod
    def _collect_weights(model: nn.Module) -> dict[str, TensorReprWithValue]:
        """
        Collect the direct parameters and buffers of a module.

        Args:
            model (nn.Module): The module whose weights are collected.

        Returns:
            dict[str, TensorReprWithValue]: The collected weights by name.
        """

        # Initialize the collected weights of the model
        model_weights: dict[str, TensorReprWithValue] = {}

        # Collect the direct parameters of the model
        weight_name: str
        weight: Tensor
        for weight_name, weight in model.named_parameters():
            if "." not in weight_name:
                model_weights[weight_name] = TensorReprWithValue(
                    shape=tuple(weight.shape),
                    dtype=str(weight.dtype),
                    value=weight.data
                )

        # Collect the direct buffers of the model
        for weight_name, weight in model.named_buffers():
            if "." not in weight_name:
                model_weights[weight_name] = TensorReprWithValue(
                    shape=tuple(weight.shape),
                    dtype=str(weight.dtype),
                    value=weight.data
                )

        # Return the collected weights of the model
        return model_weights

    @staticmethod
    def _collect_literals(
        model: nn.Module,
        model_weights: dict[str, TensorReprWithValue]
    ) -> dict[str, Any]:
        """
        Collect the plain python attributes of a module as literal values.

        Args:
            model (nn.Module): The module whose attributes are collected.
            model_weights (dict[str, TensorReprWithValue]): The already
                collected weights of the module.

        Returns:
            dict[str, Any]: The collected literal attributes by name.
        """

        # Initialize the collected literals of the module
        literals: dict[str, Any] = {}

        # Check each public attribute of the module
        attr_name: str
        for attr_name in dir(model):

            # Private attributes are not part of the forward logic
            if attr_name.startswith("_") or attr_name in model_weights:
                continue

            # Get the value of the attribute from the module
            attr_value: Any = getattr(model, attr_name)

            # Modules and parameters are handled elsewhere
            if isinstance(attr_value, (nn.Module, nn.Parameter, Tensor)):
                continue

            # Methods and types are not values
            if callable(attr_value) or isinstance(attr_value, type):
                continue

            # Only the plain python values are embedded as literals
            if isinstance(attr_value, (int, float, bool, str, type(None))):
                literals[attr_name] = attr_value

        # Return the collected literals of the module
        return literals
