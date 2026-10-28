"""
Core schema types of NNSchemaParser.

A parsed model is a tree of ModelBlockSchema blocks, each holding its inputs,
outputs, weights, sub-blocks and the instruction list of its forward body.
"""

# Import Modules
from typing import Any

from nnschemaparser.core.instructions import Instruction
from nnschemaparser.core.tensor_schema import TensorRepr, TensorReprWithValue


class ModelBlockSchema:
    """
    Schema of one block of a parsed model.

    A block mirrors one nn.Module of the source model: its forward parameters,
    its returned values, its direct parameters and buffers, its child modules
    as sub-blocks, and the lowered body of its forward method.
    """

    def __init__(
        self,
        inputs: dict[str, TensorRepr],
        outputs: list[TensorRepr],
        weights: dict[str, TensorReprWithValue],
        sub_blocks: dict[str, "ModelBlockSchema"],
        body: list[Instruction],
        module: Any | None = None
    ) -> None:
        """
        Init function.

        Args:
            inputs (dict[str, TensorRepr]): The list of all named input values of the model
                                            (often one X value)
            outputs (list[TensorRepr]): The list of all named output values of the model
                                            (often one Y value)
            weights: dict[str, TensorReprWithValue]: The weights of the current block.
            sub_blocks (dict[str, ModelBlockSchema]): The list of all the sub block of the model
                                            (all the other nn.Modules in the model class attribute)
            body (list[Instruction]): What is happening inside the current model block
                                            (the graph that will be shown in the webUI)
            module (Any | None): The source module of the block, used by the
                                            interpreter to resolve dynamic attributes.
        """

        # Store here the model input(s) representation(s)
        self.inputs: dict[str, TensorRepr] = inputs

        # Store here the model output(s) representation(s)
        self.outputs: list[TensorRepr] = outputs

        # Store here the weights of the block
        self.weights: dict[str, TensorReprWithValue] = weights

        # Store here the model sub blocks
        self.sub_blocks: dict[str, ModelBlockSchema] = sub_blocks

        # Store here the model body list of instructions / expressions
        self.body: list[Instruction] = body

        # Store here the source module of the block
        self.module: Any | None = module
