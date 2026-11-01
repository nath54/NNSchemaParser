"""
Tensor representation types of NNSchemaParser.

A TensorRepr describes a tensor flowing through the graph by its shape and its
dtype, TensorReprWithValue additionally holds the value itself, and
TensorContainer identifies a captured value by a stable container id and a
slot type for the activation and gradient traces.
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

    def __repr__(self) -> str:
        """
        Get the developer representation of the tensor representation.

        Returns:
            str: E.g. "TensorRepr(shape=(2, 16), dtype='torch.float32')".
        """

        # The shape and the dtype of the represented tensor
        return f"TensorRepr(shape={self.shape!r}, dtype={self.dtype!r})"

    def __str__(self) -> str:
        """
        Get the readable representation of the tensor representation.

        Returns:
            str: E.g. "torch.float32[2, 16]", or "bool" for scalar values.
        """

        # The scalar python values have no shape brackets
        if self.shape == ():
            return self.dtype
        return f"{self.dtype}{list(self.shape)}"


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

    def __repr__(self) -> str:
        """
        Get the developer representation of the value holding representation.

        Returns:
            str: E.g. "TensorReprWithValue(shape=(2, 16), dtype='torch.float32',
            value=Tensor)". The tensor itself is not printed, only its type.
        """

        # The shape, the dtype and the type of the held value (not its content)
        return (
            f"TensorReprWithValue(shape={self.shape!r}, dtype={self.dtype!r}, "
            f"value={type(self.value).__name__})"
        )


class TensorContainer(TensorReprWithValue):
    """
    Captured tensor identified by a stable container id and a slot type.

    Tensor containers are the payload of the activation and gradient traces:
    every captured tensor is referenced by a unique container id (e.g.
    "input_0001" or "layers_0_0002") and by a slot type describing its role
    ("input", "activation", "output" or "gradient"), so a viewer can attach
    the values to the nodes of a drawn graph.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: str,
        value: Tensor,
        container_id: str,
        slot_type: str
    ) -> None:
        """
        Init function.

        Args:
            shape (tuple[int, ...]): Shape of the tensor.
            dtype (str): The type of the tensor's values.
            value (Tensor): The value of the tensor.
            container_id (str): The stable unique id of the container (e.g.
                "input_0001" or "layers_0_0002").
            slot_type (str): The role of the tensor in the execution, one of
                "input", "activation", "output" or "gradient".
        """

        # Init parent
        super().__init__(shape=shape, dtype=dtype, value=value)

        # Store here the stable container id of the captured tensor
        self.container_id: str = container_id

        # Store here the role of the tensor in the execution
        self.slot_type: str = slot_type

    def __repr__(self) -> str:
        """
        Get the developer representation of the captured tensor.

        Returns:
            str: E.g. "TensorContainer(container_id='input_0001',
            slot_type='input', shape=(2, 16), dtype='torch.float32')".
        """

        # The container identity first, then the tensor description
        return (
            f"TensorContainer(container_id={self.container_id!r}, "
            f"slot_type={self.slot_type!r}, shape={self.shape!r}, "
            f"dtype={self.dtype!r})"
        )

    def __str__(self) -> str:
        """
        Get the readable representation of the captured tensor.

        Returns:
            str: E.g. "input_0001 (input): torch.float32[2, 16]".
        """

        # The container id and slot type, followed by the tensor description
        return f"{self.container_id} ({self.slot_type}): {TensorRepr.__str__(self)}"
