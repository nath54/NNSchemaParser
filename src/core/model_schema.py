"""
TODO
"""

# Import Modules
from src.core.tensor_schema import TensorRepr
from src.core.instructions import Instruction


class ModelBlockSchema:
    """
    TODO
    """

    def __init__(
        self,
        inputs: dict[str, TensorRepr],
        outputs: list[TensorRepr],
        sub_blocks: dict[str, "ModelBlockSchema"],
        body: list[Instruction]
    ) -> None:
        """
        Init function.

        Args:
            inputs (dict[str, TensorRepr]): The list of all named input values of the model
                                            (often one X value)
            outputs (list[TensorRepr]): The list of all named output values of the model
                                            (often one Y value)
            sub_blocks (dict[str, ModelBlockSchema]): The list of all the sub block of the model
                                            (all the other nn.Modules in the model class attribute)
            body (list[Instruction]): What is happening inside the current model block
                                            (the graph that will be shown in the webUI)
        """

        # Store here the model input(s) representation(s)
        self.inputs: dict[str, TensorRepr] = inputs

        # Store here the model output(s) representation(s)
        self.outputs: list[TensorRepr] = outputs

        # Store here the model sub blocks
        self.sub_blocks: dict[str, ModelBlockSchema] = sub_blocks

        # Store here the model body list of instructions / expressions
        self.body: list[Instruction] = body
