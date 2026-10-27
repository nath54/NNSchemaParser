"""
TODO
"""

# Import Modules
from torch import Tensor


class TensorRepr:
    """
    TODO
    """

    def __init__(self, shape: tuple[int, ...], dtype: str) -> None:
        """
        Init function.

        Args:
            shape (tuple[int | str, ...]): shape of the tensor, the str type is for symbolic values
            dtype (str): The type of the tensor's values
        """

        # Store here the shape of the tensor
        self.shape: tuple[int | str, ...] = shape

        # Store here the type of the tensor
        self.dtype: str = dtype


class TensorReprWithValue(TensorRepr):
    """
    TODO
    """

    def __init__(self, shape: tuple[int, ...], dtype: str, value: Tensor) -> None:
        """
        Init function.

        Args:
            shape (tuple[int, ...]): shape of the tensor
            dtype (str): The type of the tensor's values
        """

        # Init parent
        super().__init__(shape=shape, dtype=dtype)

        # Store here the value of the tensor
        self.value: Tensor = value
