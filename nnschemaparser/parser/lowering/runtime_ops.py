"""
Runtime helper functions and operator tables of the lowering.

These helpers are the targets of the emitted operation instructions: they are
called by the interpreter at execution time with the resolved arguments.
"""

# Import Modules
from typing import Any, Callable

import ast
import operator


# Mapping of the python binary operators to their operator module function
BINARY_OPERATORS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.MatMult: operator.matmul,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.BitXor: operator.xor,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift
}


# Mapping of the python comparison operators to their operator module function
COMPARISON_OPERATORS: dict[type, Callable[[Any, Any], Any]] = {
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Is: operator.is_,
    ast.IsNot: operator.is_not
}


def check(value: Any) -> None:
    """
    Raise an assertion error when the given value is falsy.

    Args:
        value (Any): The value checked by the assertion.

    Raises:
        AssertionError: When the given value is falsy.
    """

    # Raise the assertion error when the value is falsy
    if not value:
        raise AssertionError("Assertion failed during the schema execution")


def raise_exception(exception_type: type[BaseException], *args: Any) -> None:
    """
    Raise an exception of the given type with the given arguments.

    Args:
        exception_type (type[BaseException]): The type of the exception.
        *args (Any): The arguments of the exception constructor.

    Raises:
        BaseException: Always, an exception of the given type.
    """

    # Raise the exception of the given type
    raise exception_type(*args)


def raise_exception_instance(exception: BaseException) -> None:
    """
    Raise the given exception instance.

    Args:
        exception (BaseException): The exception instance to raise.

    Raises:
        BaseException: Always, the given exception instance.
    """

    # Raise the given exception instance
    raise exception


def make_tuple(*args: Any) -> tuple:
    """
    Build a tuple from the given arguments.

    Args:
        *args (Any): The elements of the tuple.

    Returns:
        tuple: The built tuple.
    """

    # Return the tuple of the given arguments
    return args


def make_list(*args: Any) -> list:
    """
    Build a list from the given arguments.

    Args:
        *args (Any): The elements of the list.

    Returns:
        list: The built list.
    """

    # Return the list of the given arguments
    return list(args)


def identity(value: Any) -> Any:
    """
    Return the given value unchanged.

    Args:
        value (Any): The value to return.

    Returns:
        Any: The given value, unchanged.
    """

    # Return the value unchanged
    return value


def not_contains(sequence: Any, item: Any) -> bool:
    """
    Return whether the item is not in the sequence.

    Args:
        sequence (Any): The sequence tested for the absence of the item.
        item (Any): The item tested for absence.

    Returns:
        bool: Whether the item is not in the sequence.
    """

    # Return the negated membership test
    return item not in sequence


def bool_and(left: Any, right: Any) -> Any:
    """
    Return the logical and of two values.

    Args:
        left (Any): The left value of the operation.
        right (Any): The right value of the operation.

    Returns:
        Any: The logical and of the two values.
    """

    # Return the logical and of the two values
    return left and right


def bool_or(left: Any, right: Any) -> Any:
    """
    Return the logical or of two values.

    Args:
        left (Any): The left value of the operation.
        right (Any): The right value of the operation.

    Returns:
        Any: The logical or of the two values.
    """

    # Return the logical or of the two values
    return left or right


def format_part(value: Any, conversion: int = -1, format_spec: str | None = None) -> str:
    """
    Format a value as the part of an f-string.

    Args:
        value (Any): The value to format.
        conversion (int): The conversion code of the f-string part.
        format_spec (str | None): The format spec of the f-string part.

    Returns:
        str: The formatted value.
    """

    # Apply the conversion of the f-string part when defined
    if conversion == 115:
        value = str(value)
    elif conversion == 114:
        value = repr(value)
    elif conversion == 97:
        value = ascii(value)

    # Apply the format spec of the f-string part when defined
    if format_spec:
        return format(value, format_spec)
    if conversion != -1:
        return value

    # Else the value is formatted with its default format
    return format(value)


def join_str(*parts: Any) -> str:
    """
    Join the parts of an f-string into a single string.

    Args:
        *parts (Any): The parts of the f-string.

    Returns:
        str: The joined string of the f-string parts.
    """

    # Join the string parts of the f-string
    return "".join(
        part if isinstance(part, str) else str(part) for part in parts
    )


def setitem(obj: Any, key: Any, value: Any) -> None:
    """
    Set the item of an object, the lowering of a subscript assignment.

    Args:
        obj (Any): The object whose item is set.
        key (Any): The index or the slice key of the item.
        value (Any): The value assigned to the item.
    """

    # Set the item on the object
    obj[key] = value


def make_dict(*args: Any) -> dict:
    """
    Build a dict from alternating keys and values.

    Args:
        *args (Any): The keys and the values of the dict, alternating.

    Returns:
        dict: The built dict.
    """

    # Pair the arguments two by two into a dict
    return dict(zip(args[0::2], args[1::2]))
