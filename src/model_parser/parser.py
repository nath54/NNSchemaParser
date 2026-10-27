"""
TODO
"""

# Import Modules
from typing import Callable, Type, get_args

import inspect

from torch import nn, Tensor

from src.core.model_schema import ModelBlockSchema
from src.core.tensor_schema import TensorRepr
from src.core.instructions import Instruction


class NNParser:
    """
    TODO
    """

    @staticmethod
    def parse_model(model: nn.Module) -> ModelBlockSchema:
        """
        TODO

        Args:
            model (nn.Module): The model to parse

        Returns:
            ModelSchema: The parsed model schema
        """

        # Initialize the model inputs
        model_inputs: dict[str, TensorRepr] = {}

        # Initialize the model outputs
        model_outputs: list[TensorRepr] = []

        # Initialize the model sub_blocks
        model_sub_blocks: dict[str, ModelBlockSchema] = {
            attr_name: NNParser.parse_model(model=getattr(model, attr_name))
            for attr_name in dir(model) if getattr(model, attr_name) is nn.Module
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
            sub_blocks=model_sub_blocks,
            body=model_body
        )
