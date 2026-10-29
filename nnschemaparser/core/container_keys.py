"""
Container entry key utilities shared by the lowering and the interpreter.

The entries of a sub-block container are keyed by "container.index" in the
sub-block dictionaries, and must be iterated in their declaration order.
"""

# Import Modules


def container_entry_sort_key(entry_key: str) -> int:
    """
    Get the numeric sort key of a container entry key.

    Args:
        entry_key (str): The entry key (e.g. "encoders.2").

    Returns:
        int: The numeric suffix of the key, or -1 when not numeric.
    """

    # Split the key into its container prefix and its suffix
    parts: list[str] = entry_key.rsplit(".", 1)

    # A numeric suffix is sorted by its value
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])

    # A non numeric suffix is sorted last
    return -1


def container_entry_keys(
    sub_block_names: set[str] | dict[str, str],
    container_name: str
) -> list[str]:
    """
    Get the ordered entry keys of a container in the parsed sub-blocks.

    Args:
        sub_block_names (set[str] | dict[str, str]): The names of the parsed
            sub-blocks, either a set of names or the sub-block dictionary.
        container_name (str): The name of the container.

    Returns:
        list[str]: The entry keys sorted by their numeric suffix.
    """

    # The special self path holds the direct numeric entries of the module
    if container_name == "$self":
        return sorted(
            [
                sub_block_name for sub_block_name in sub_block_names
                if sub_block_name.isdigit()
            ],
            key=container_entry_sort_key
        )

    # Collect the sub-block keys of the container entries
    keys: list[str] = [
        sub_block_name for sub_block_name in sub_block_names
        if sub_block_name.startswith(f"{container_name}.")
    ]

    # Sort the entry keys by their numeric suffix
    keys.sort(key=container_entry_sort_key)

    # Return the ordered entry keys of the container
    return keys
