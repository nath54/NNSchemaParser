"""
TODO
"""

# Import Modules
from src.core.tensor_schema import TensorRepr


class ModelSchema:
    """
    TODO
    """

    def __init__(
        self,
        inputs: dict[str, TensorRepr],
        outputs: dict[str, TensorRepr]
    ) -> None:
        """
        Init function.

        Args:
            inputs (dict[str, TensorRepr]): The list of all named input values of the model
                                            (often one X value)
            outputs (dict[str, TensorRepr]): The list of all named output values of the model
                                            (often one Y value)
        """

        # Store here the model input(s) representation(s)
        self.inputs: dict[str, TensorRepr] = inputs

        # Store here the model output(s) representation(s)
        self.outputs: dict[str, TensorRepr] = outputs
