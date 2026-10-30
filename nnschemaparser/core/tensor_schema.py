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
