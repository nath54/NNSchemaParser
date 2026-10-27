"""
TODO
"""

# Import Modules
from typing import Optional, Callable, Type, get_args

import inspect

from torch import nn, Tensor

from src.core.model_schema import ModelBlockSchema
from src.core.tensor_schema import TensorRepr, TensorReprWithValue
from src.core.instructions import Instruction


class NNParser:
    """
    TODO
    """

    @staticmethod
    def parse_model(model: nn.Module, inputs_tensors_shapes: dict[str, tuple[int, ...]]) -> ModelBlockSchema:
        """
        TODO

        Args:
            model (nn.Module): The model to parse
            inputs_tensors_shapes (dict[str, tuple[int, ...]]): The input tensors shapes

        Returns:
            ModelSchema: The parsed model schema
        """

        # Initialize the model inputs
        model_inputs: dict[str, TensorRepr] = {}

        # Initialize the model outputs
        model_outputs: list[TensorRepr] = []

        # Initialize the model weights
        model_weights = {
            attr_name: TensorReprWithValue(
                shape=getattr(model, attr_name).shape,
                dtype=str(getattr(model, attr_name).dtype),
                value=getattr(model, attr_name).data
            )
            for attr_name in dir(model) if isinstance(getattr(model, attr_name), nn.Parameter)
        }

        # Initialize the model sub_blocks
        model_sub_blocks: dict[str, ModelBlockSchema] = {
            attr_name: NNParser.parse_model(model=getattr(model, attr_name), inputs_tensors_shapes={})  # TODO: parse the body before to get the tensor shapes
            for attr_name in dir(model) if isinstance(getattr(model, attr_name), nn.Module)
        }

        # Initialize the model body
        model_body: list[Instruction] = []

        # We get the forward function of the model
        model_forward_fn: Callable = model.forward

        # Inspect the signature of the forward method of the model
        sig: inspect.Signature = inspect.signature(model_forward_fn)

        # Loop on all the parameter of the function to get the model inputs
        name: str
        param: inspect.Parameter
        for name, param in sig.parameters.items():

            # We only care about the Tensor inputs
            if param.annotation != Tensor:
                continue

            model_inputs[name] = TensorRepr(
                shape = (),  # TODO
                dtype = ""   # TODO
            )

        # Parse the model output
        out_type = sig.return_annotation

        # If the output_type is only one Tensor, it is easy
        if out_type == Tensor:
            model_outputs = [TensorRepr(
                shape = (),  # TODO
                dtype = ""   # TODO
            )]

        # Else if it is a tuple, we need to search for the sub_types
        elif out_type == tuple:
            sub_types: list[Type] = get_args(out_type)

            sub_type: Type
            for sub_type in sub_types:
                if sub_type == Tensor:
                    model_outputs.append(
                        TensorRepr(
                            shape = (),  # TODO
                            dtype = ""   # TODO
                        )
                    )

        # If no type compatible, raise error
        else:
            raise TypeError(
                f"Model output is not a Tensor or a tuple containing at least one tensor: {out_type}"
            )

        # Return the results
        return ModelBlockSchema(
            inputs=model_inputs,
            outputs=model_outputs,
            weights=model_weights,
            sub_blocks=model_sub_blocks,
            body=model_body
        )
