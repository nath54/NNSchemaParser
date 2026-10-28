"""
Name resolution utilities of the lowering.

Resolves the dotted names of the calls (e.g. torch.nn.functional.softmax) from
the globals of the forward method, the known namespaces and the builtins.
"""

# Import Modules
from typing import Any

import ast
import builtins
import operator

import torch
from torch import nn

from src.model_parser.lowering.errors import UnsupportedSyntaxError


# Sentinel used to distinguish a missing name from a None value
_MISSING: Any = object()


# Namespace used to resolve the dotted names of calls (e.g. torch.nn.functional.softmax)
RESOLVE_NAMESPACE: dict[str, Any] = {
    "torch": torch,
    "F": torch.nn.functional,
    "nn": nn,
    "functional": nn.functional,
    "operator": operator
}


def is_known_root(name: str, forward_globals: dict[str, Any]) -> bool:
    """
    Check whether a name root can be resolved without the forward globals.

    Args:
        name (str): The root name of a dotted name (e.g. "torch").
        forward_globals (dict[str, Any]): The globals of the forward method.

    Returns:
        bool: Whether the root is a known namespace, builtin or global.
    """

    # The root is known when it resolves from any of the name sources
    return (
        name in forward_globals
        or name in RESOLVE_NAMESPACE
        or hasattr(builtins, name)
    )


def resolve_dotted_name(dotted: str, node: ast.AST, forward_globals: dict[str, Any]) -> Any:
    """
    Resolve a dotted name to the object it designates.

    Args:
        dotted (str): The dotted name to resolve (e.g. "torch.relu").
        node (ast.AST): The node the name comes from, for error reporting.
        forward_globals (dict[str, Any]): The globals of the forward method.

    Returns:
        Any: The resolved object.

    Raises:
        UnsupportedSyntaxError: When the dotted name cannot be resolved.
    """

    # Split the dotted name into its parts
    parts: list[str] = dotted.split(".")

    # Resolve the root of the name from the forward globals
    current: Any = forward_globals.get(parts[0], _MISSING)

    # Else resolve the root from the known namespace or the builtins
    if current is _MISSING:
        current = RESOLVE_NAMESPACE.get(parts[0], _MISSING)
    if current is _MISSING:
        current = getattr(builtins, parts[0], _MISSING)

    # If the root cannot be resolved, the name is not supported
    if current is _MISSING:
        raise UnsupportedSyntaxError(f"Unsupported callable name: {dotted}", node)

    # Walk the remaining parts of the dotted name
    part: str
    for part in parts[1:]:
        current = getattr(current, part, _MISSING)

        # If a part cannot be resolved, the name is not supported
        if current is _MISSING:
            raise UnsupportedSyntaxError(f"Unsupported callable name: {dotted}", node)

    # Return the resolved object
    return current


def dotted_name(expr: ast.expr) -> str | None:
    """
    Get the dotted name of a name or attribute chain expression.

    Args:
        expr (ast.expr): The expression to get the dotted name of.

    Returns:
        str | None: The dotted name, or None if not a name chain.
    """

    # An attribute chain is joined with dots
    if isinstance(expr, ast.Attribute):
        parent: str | None = dotted_name(expr=expr.value)
        if parent is None:
            return None
        return f"{parent}.{expr.attr}"

    # A root name is the start of the chain
    if isinstance(expr, ast.Name):
        return expr.id

    # Any other expression has no dotted name
    return None


class GlobalCallable:
    """
    A dotted name resolved lazily from the globals of the forward method.

    Used for the callables of the failed optional imports (e.g. flash_attn),
    which must only fail when the corresponding code path is executed.
    """

    def __init__(self, dotted: str, forward_globals: dict[str, Any]) -> None:
        """
        Init function.

        Args:
            dotted (str): The dotted name of the callable.
            forward_globals (dict[str, Any]): The globals of the forward.
        """

        # Store here the dotted name of the callable
        self.dotted: str = dotted

        # Store here the globals of the forward method
        self.forward_globals: dict[str, Any] = forward_globals

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """
        Resolve the dotted name then call the resolved object.

        Args:
            *args (Any): The positional arguments of the call.
            **kwargs (Any): The keyword arguments of the call.

        Returns:
            Any: The result of the resolved callable.

        Raises:
            RuntimeError: When the dotted name cannot be resolved.
        """

        # Resolve the dotted name at call time
        resolved: Any = resolve_dotted_name(
            dotted=self.dotted,
            node=self,
            forward_globals=self.forward_globals
        )

        # Only the failed optional imports are None at this point
        if resolved is None:
            raise RuntimeError(
                f"The callable {self.dotted} is not available "
                "(failed optional import)"
            )

        # Call the resolved object with the given arguments
        return resolved(*args, **kwargs)
