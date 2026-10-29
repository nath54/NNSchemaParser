"""
Tests of the method schemas and the scalar inputs of the parser.

Run with: python -m tests.test_methods_and_scalars
"""

# Import Modules
from typing import Any

import torch
from torch import Tensor, nn

from nnschemaparser.core.model_schema import ModelBlockSchema
from nnschemaparser.interpreter import SchemaInterpreter
from nnschemaparser.parser import NNParser


class Projector(nn.Module):
    """
    Model delegating part of its forward to another method.
    """

    def __init__(self) -> None:
        """
        Init function.
        """

        # Init parent
        super().__init__()

        # Create the weights of the model
        self.w1: nn.Parameter = nn.Parameter(torch.randn(4, 4) * 0.1)
        self.w2: nn.Parameter = nn.Parameter(torch.randn(4, 4) * 0.1)

    def project_twice(self, x: Tensor) -> Tensor:
        """
        Apply the second projection to the input.

        Args:
            x (Tensor): The input tensor of the projection.

        Returns:
            Tensor: The projected tensor.
        """

        # Project the input with the second weight
        return (x @ self.w2).relu()

    def forward(self, x: Tensor, times: int = 2, tag: str = "a") -> Tensor:
        """
        Compute the model output from the input and the scalar options.

        Args:
            x (Tensor): The input tensor of the model.
            times (int): The number of times the input is repeated.
            tag (str): A tag selecting an extra transformation.

        Returns:
            Tensor: The model output.
        """

        # Project the input with the first weight
        hidden: Tensor = (x @ self.w1).relu()

        # Delegate the rest of the computation to the other method
        output: Tensor = self.project_twice(hidden)

        # Apply the scalar options of the forward
        output = output * times
        if tag == "double":
            output = output * 2

        # Return the computed output
        return output


def test_method_schema() -> None:
    """
    Test the parsing and the execution of a method called by the forward.
    """

    # Parse the model with its methods
    model: Projector = Projector().eval()
    x: Tensor = torch.randn(2, 4)
    schema: ModelBlockSchema = NNParser.parse_model(
        model=model,
        inputs_tensors_shapes={"x": (2, 4)},
        inputs_tensors_dtypes={"x": "torch.float32"},
        inputs_tensors_values={"x": x}
    )

    # The method called by the forward is parsed into the schema
    assert "project_twice" in schema.methods
    method_schema: ModelBlockSchema = schema.methods["project_twice"]
    assert method_schema.method_name == "project_twice"

    # The execution through the method schema matches the real forward
    outputs: list[Any] = SchemaInterpreter().run(
        block=schema,
        scope={"x": x.clone(), "times": 3, "tag": "double"}
    )
    reference: Tensor = model(x, times=3, tag="double")
    assert torch.equal(outputs[0], reference), "method schema run mismatch"

    # The defaults of the forward are used when the scalars are unbound
    outputs = SchemaInterpreter().run(block=schema, scope={"x": x.clone()})
    assert torch.equal(outputs[0], model(x)), "default scalars run mismatch"


def test_scalar_inputs() -> None:
    """
    Test the typing of the scalar inputs from their annotations and values.
    """

    # Parse the model with its scalar inputs
    model: Projector = Projector().eval()
    x: Tensor = torch.randn(2, 4)
    schema: ModelBlockSchema = NNParser.parse_model(
        model=model,
        inputs_tensors_shapes={"x": (2, 4)},
        inputs_tensors_dtypes={"x": "torch.float32"},
        inputs_tensors_values={"x": x}
    )

    # The scalar inputs are typed by their annotations
    assert schema.inputs["times"].dtype == "int"
    assert schema.inputs["tag"].dtype == "str"

    # The scalar inputs passed at run time are used as they are
    outputs: list[Any] = SchemaInterpreter().run(
        block=schema,
        scope={"x": x.clone(), "times": 5, "tag": "simple"}
    )
    assert torch.equal(outputs[0], model(x, times=5, tag="simple"))


if __name__ == "__main__":

    # Run all the tests of the module
    test_method_schema()
    test_scalar_inputs()
    print("tests.test_methods_and_scalars: ALL OK")
