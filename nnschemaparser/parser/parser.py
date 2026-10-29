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

from nnschemaparser.core.instructions import Instruction, OpInstruction, VarRef
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

        # Parse the other methods of the module called by the forward
        methods: dict[str, ModelBlockSchema] = NNParser._parse_methods(
            model=model,
            body=body,
            parsed_names=set()
        )

        # Build the schema of the model from its parsed parts
        schema: ModelBlockSchema = NNParser._build_schema(
            model=model,
            signature=signature,
            inputs_tensors_shapes=inputs_tensors_shapes,
            inputs_tensors_values=inputs_tensors_values,
            model_weights=model_weights,
            model_sub_blocks=model_sub_blocks,
            body=body,
            output_vars=output_vars,
            methods=methods
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
    def _parse_methods(
        model: nn.Module,
        body: list[Instruction],
        parsed_names: set[str]
    ) -> dict[str, ModelBlockSchema]:
        """
        Parse the other methods of a module called by a lowered body.

        Args:
            model (nn.Module): The module owning the called methods.
            body (list[Instruction]): The lowered body calling the methods.
            parsed_names (set[str]): The already parsed method names, updated
                in place to break the recursion of the mutual calls.

        Returns:
            dict[str, ModelBlockSchema]: The parsed method schemas by name.
        """

        # Collect the names of the methods called on self in the body
        method_names: set[str] = NNParser._collect_method_names(body=body)

        # Parse each new method of the module class
        methods: dict[str, ModelBlockSchema] = {}
        method_name: str
        for method_name in sorted(method_names):

            # Skip the methods already parsed by another body
            if method_name in parsed_names:
                continue
            parsed_names.add(method_name)

            # Skip the attributes that are not methods of the module class
            if not NNParser._is_class_method(model=model, name=method_name):
                continue

            # Parse the method, recursing on its own self method calls
            method_schema: ModelBlockSchema | None = NNParser._parse_method(
                model=model,
                method_name=method_name,
                parsed_names=parsed_names
            )

            # Keep the parsed method when its body could be lowered
            if method_schema is not None:
                methods[method_name] = method_schema

        # Return the parsed method schemas by name
        return methods

    @staticmethod
    def _collect_method_names(body: list[Instruction]) -> set[str]:
        """
        Collect the names of the methods called on self in a lowered body.

        Args:
            body (list[Instruction]): The lowered body to scan.

        Returns:
            set[str]: The names of the self method calls of the body.
        """

        # A self method call is lowered to a getattr of the self reference
        method_names: set[str] = set()
        instruction: Instruction
        for instruction in body:
            if not isinstance(instruction, OpInstruction):
                continue
            if instruction.qualified_name != "getattr" or \
                    len(instruction.args) != 2:
                continue
            target: VarRef | Any = instruction.args[0]
            attribute: VarRef | Any = instruction.args[1]
            if isinstance(target, VarRef) and target.name == "$self" and \
                    isinstance(attribute, str):
                method_names.add(attribute)

        # Return the collected method names
        return method_names

    @staticmethod
    def _is_class_method(model: nn.Module, name: str) -> bool:
        """
        Check whether a name is a plain method defined by the module class.

        Args:
            model (nn.Module): The module owning the checked attribute.
            name (str): The name of the checked attribute.

        Returns:
            bool: Whether the attribute is a parseable method of the class.
        """

        # Only the functions defined by the class itself are parsed, the
        # inherited helpers and the non function attributes are re-dispatched
        attribute: Any = type(model).__dict__.get(name)
        return inspect.isfunction(attribute) and name != "forward"

    @staticmethod
    def _parse_method(
        model: nn.Module,
        method_name: str,
        parsed_names: set[str]
    ) -> ModelBlockSchema | None:
        """
        Parse one method of a module into a block schema.

        Args:
            model (nn.Module): The module owning the parsed method.
            method_name (str): The name of the parsed method.
            parsed_names (set[str]): The already parsed method names.

        Returns:
            ModelBlockSchema | None: The parsed method schema, or None when
                the method body could not be lowered.
        """

        # Collect the parts shared with the forward schema of the module
        model_weights: dict[str, TensorReprWithValue] = NNParser._collect_weights(
            model=model
        )
        model_sub_blocks: dict[str, ModelBlockSchema]
        container_names: set[str]
        dict_container_names: set[str]
        model_sub_blocks, container_names, dict_container_names = \
            collect_sub_blocks(model=model, parse_model=NNParser.parse_model)
        literals: dict[str, Any] = NNParser._collect_literals(
            model=model,
            model_weights=model_weights
        )

        # Lower the body of the method into instructions
        try:
            method_body: list[Instruction]
            output_vars: list[str]
            method_body, output_vars = NNParser._lower_method(
                model=model,
                method_name=method_name,
                sub_block_names=set(model_sub_blocks.keys()),
                container_names=container_names,
                dict_container_names=dict_container_names,
                weight_names=set(model_weights.keys()),
                literals=literals
            )
        except Exception:  # noqa: BLE001 - an unparsable method is skipped
            return None

        # Parse the methods called by the method itself
        method_methods: dict[str, ModelBlockSchema] = NNParser._parse_methods(
            model=model,
            body=method_body,
            parsed_names=parsed_names
        )

        # Collect the inputs of the method from its signature
        signature: inspect.Signature = inspect.signature(
            getattr(model, method_name)
        )
        method_inputs: dict[str, TensorRepr] = {
            parameter_name: TensorRepr(shape=(), dtype="")
            for parameter_name, parameter in signature.parameters.items()
            if parameter_name != "self"
            and parameter.kind != inspect.Parameter.VAR_KEYWORD
        }

        # Collect the outputs of the method from its return instruction
        method_outputs: list[TensorRepr] = [
            TensorRepr(shape=(), dtype="") for _ in output_vars
        ]

        # Return the schema of the method
        return ModelBlockSchema(
            inputs=method_inputs,
            outputs=method_outputs,
            weights=model_weights,
            sub_blocks=model_sub_blocks,
            body=method_body,
            module=model,
            methods=method_methods,
            method_name=method_name
        )

    @staticmethod
    def _lower_method(
        model: nn.Module,
        method_name: str,
        sub_block_names: set[str],
        container_names: set[str],
        dict_container_names: set[str],
        weight_names: set[str],
        literals: dict[str, Any]
    ) -> tuple[list[Instruction], list[str]]:
        """
        Lower another method of a module into its body instructions.

        Args:
            model (nn.Module): The module owning the lowered method.
            method_name (str): The name of the lowered method.
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

        # Create the lowering of the method
        lowering: BodyLowering = BodyLowering(
            sub_block_names=sub_block_names,
            container_names=container_names,
            dict_container_names=dict_container_names,
            weight_names=weight_names,
            literals=literals
        )

        # Lower the body of the method into instructions
        return lowering.lower_method(model=model, method_name=method_name)

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
        inputs_tensors_values: dict[str, Tensor] | None,
        model_weights: dict[str, TensorReprWithValue],
        model_sub_blocks: dict[str, ModelBlockSchema],
        body: list[Instruction],
        output_vars: list[str],
        methods: dict[str, ModelBlockSchema]
    ) -> ModelBlockSchema:
        """
        Build the block schema of a model from its parsed parts.

        Args:
            model (nn.Module): The parsed model.
            signature (inspect.Signature): The signature of the forward.
            inputs_tensors_shapes (dict[str, tuple[int, ...]]): The shapes of
                the input tensors of the forward method.
            inputs_tensors_values (dict[str, Tensor] | None): The optional
                example values of the inputs, describing the scalar inputs.
            model_weights (dict[str, TensorReprWithValue]): The weights of the
                model collected by name.
            model_sub_blocks (dict[str, ModelBlockSchema]): The sub-blocks of
                the model collected by name.
            body (list[Instruction]): The lowered body of the forward.
            output_vars (list[str]): The names of the returned variables.
            methods (dict[str, ModelBlockSchema]): The other methods of the
                module called by the forward, parsed as block schemas.

        Returns:
            ModelBlockSchema: The built schema of the model.
        """

        # Collect the inputs of the model from the forward signature
        model_inputs: dict[str, TensorRepr] = {}
        input_defaults: dict[str, Any] = {}
        parameter_name: str
        for parameter_name in signature.parameters:

            # The self parameter is not an input of the model
            if parameter_name == "self":
                continue

            # The variadic keyword argument is not an input of the model
            if signature.parameters[parameter_name].kind == \
                    inspect.Parameter.VAR_KEYWORD:
                continue

            # The scalar example values describe their input by their type
            example_value: Any | None = (
                inputs_tensors_values or {}).get(parameter_name
            )
            if example_value is not None and not isinstance(example_value, Tensor):
                model_inputs[parameter_name] = TensorRepr(
                    shape=(), dtype=type(example_value).__name__
                )
                continue

            # The scalar annotations describe their input by their type
            scalar_annotations: tuple[type, ...] = (int, float, bool, str)
            if signature.parameters[parameter_name].annotation in scalar_annotations:
                model_inputs[parameter_name] = TensorRepr(
                    shape=(),
                    dtype=signature.parameters[parameter_name].annotation.__name__
                )
                continue

            # The input shapes are unknown until the resolution pass
            model_inputs[parameter_name] = TensorRepr(
                shape=inputs_tensors_shapes.get(parameter_name, ()),
                dtype=""
            )

            # The optional parameters keep their default value
            parameter_default: Any = (
                signature.parameters[parameter_name].default
            )
            if parameter_default is not inspect.Parameter.empty:
                input_defaults[parameter_name] = parameter_default

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
            module=model,
            methods=methods,
            input_defaults=input_defaults
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
