"""
Tensor representation types of NNSchemaParser.

A TensorRepr describes a tensor flowing through the graph by its shape and its
dtype, and TensorReprWithValue additionally holds the value itself.
"""

# Import Modules
from torch import Tensor

# The placeholder dtype of unresolved tensor representations
PLACEHOLDER_DTYPE: str = ""


class TensorRepr:
    """
    Representation of a tensor flowing through the graph, without its value.
    """

    def __init__(self, shape: tuple[int, ...], dtype: str) -> None:
        """
        Init function.

        Args:
            shape (tuple[int | str, ...]): shape of the tensor, the str type is for symbolic values
            dtype (str): The type of the tensor's values
        """

        # Store here the shape of the tensor
        self.shape: tuple[int, ...] = shape

        # Store here the type of the tensor
        self.dtype: str = dtype


class TensorReprWithValue(TensorRepr):
    """
    Representation of a tensor keeping a snapshot of its value.
    """

    def __init__(self, shape: tuple[int, ...], dtype: str, value: Tensor) -> None:
        """
        Init function.

        Args:
            shape (tuple[int, ...]): Shape of the tensor.
            dtype (str): The type of the tensor's values.
            value (Tensor): The value of the tensor.
        """

        # Init parent
        super().__init__(shape=shape, dtype=dtype)

        # Store here the value of the tensor
        self.value: Tensor = value
