"""
Scope utilities of the schema interpreter.

Covers the placeholder tensor representations and the default values of the
forward arguments of a block.
"""

# Import Modules
from typing import Any

import inspect

from torch import Tensor

from src.core.tensor_schema import TensorRepr


# The placeholder dtype of unresolved tensor representations
PLACEHOLDER_DTYPE: str = ""


def resolve_placeholder(repr: TensorRepr, value: Tensor) -> None:
    """
    Fill the placeholder shape and dtype of a tensor representation.

    Args:
        repr (TensorRepr): The representation to fill in place.
        value (Tensor): The tensor value of the representation.
    """

    # The dtype is filled when the representation has none
    if repr.dtype == PLACEHOLDER_DTYPE:
        repr.dtype = str(value.dtype)

    # The shape is filled when the representation has none
    if repr.shape == ():
        repr.shape = tuple(int(dim) for dim in value.shape)


def is_placeholder(repr: TensorRepr) -> bool:
    """
    Check whether a tensor representation is still an unresolved placeholder.

    Args:
        repr (TensorRepr): The representation to check.

    Returns:
        bool: Whether the representation has no shape nor dtype yet.
    """

    # A placeholder has an empty shape and the placeholder dtype
    return repr.shape == () and repr.dtype == PLACEHOLDER_DTYPE


def forward_signature_defaults(module: Any) -> dict[str, Any]:
    """
    Get the default values of the forward arguments of a module.

    Args:
        module (Any): The module whose forward signature is inspected.

    Returns:
        dict[str, Any]: The default values by argument name.
    """

    # Get the signature of the forward method of the module
    signature: inspect.Signature = inspect.signature(module.forward)

    # Collect the defaults of the forward arguments
    defaults: dict[str, Any] = {}
    parameter_name: str
    parameter: inspect.Parameter
    for parameter_name, parameter in signature.parameters.items():

        # Skip the self parameter of the forward method
        if parameter_name == "self":
            continue

        # Only the parameters with a default value provide a default
        if parameter.default is not inspect.Parameter.empty:
            defaults[parameter_name] = parameter.default

    # Return the collected defaults
    return defaults
