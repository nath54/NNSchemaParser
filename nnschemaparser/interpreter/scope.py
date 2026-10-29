"""
Scope utilities of the schema interpreter.

Covers the placeholder tensor representations and the default values of the
forward arguments of a block.
"""

# Import Modules
from typing import Any

import inspect
from typing import Any

from torch import Tensor

from nnschemaparser.core.model_schema import ModelBlockSchema
from nnschemaparser.core.tensor_schema import PLACEHOLDER_DTYPE, TensorRepr
from nnschemaparser.forward import resolve_forward



# The neutral values of the scalar input types, used by the resolution
SCALAR_NEUTRAL_VALUES: dict[str, Any] = {
    "int": 0,
    "float": 0.0,
    "bool": False,
    "str": ""
}


def seed_scalar_inputs(block: ModelBlockSchema, scope: dict[str, Any]) -> None:
    """
    Seed the scope with the neutral values of the scalar block inputs.

    A scalar input declared by its type (e.g. an int or a str forward
    parameter) is seeded with a neutral value when it has no given value and
    no default, so the resolution can execute the data dependent branches.

    Args:
        block (ModelBlockSchema): The block schema being resolved.
        scope (dict[str, Any]): The scope of the resolution, updated in place.
    """

    # Seed each declared scalar input that is not bound yet
    input_name: str
    input_repr: TensorRepr
    for input_name, input_repr in block.inputs.items():
        if input_repr.shape == () and input_repr.dtype in SCALAR_NEUTRAL_VALUES:
            scope.setdefault(input_name, SCALAR_NEUTRAL_VALUES[input_repr.dtype])


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
    signature: inspect.Signature = inspect.signature(
        resolve_forward(module=module)
    )

    # Collect the defaults of the forward arguments
    defaults: dict[str, Any] = {}
    parameter_name: str
    parameter: inspect.Parameter
    for parameter_name, parameter in signature.parameters.items():

        # Skip the self parameter of the forward method
        if parameter_name == "self":
            continue

        # The var-keyword parameter receives an empty dict when unbound
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            defaults[parameter_name] = {}
            continue

        # Only the parameters with a default value provide a default
        if parameter.default is not inspect.Parameter.empty:
            defaults[parameter_name] = parameter.default

    # Return the collected defaults
    return defaults
