"""
Collection of the sub-blocks of a module for the NNSchemaParser parser.

The children of a module are parsed recursively into ModelBlockSchema. A
child whose forward cannot be lowered (e.g. torch ParametrizationList with
its self-indexed calls) is skipped: it is never called explicitly by any
forward, only reached through a runtime getattr of the parent module, which
re-dispatches to the real module.
"""

# Import Modules
from typing import Callable

import torch.nn as nn

from nnschemaparser.core.model_schema import ModelBlockSchema
from nnschemaparser.parser.lowering.errors import UnsupportedSyntaxError

# The type of the parse_model function received to avoid a circular import
ParseModel = Callable[..., ModelBlockSchema]


def parse_child(
    name: str,
    child: nn.Module,
    parse_model: ParseModel,
    skipped: dict[str, str]
) -> ModelBlockSchema | None:
    """
    Parse a direct child module, returning None when it is not parsable.

    Args:
        name (str): The attribute name of the child in its parent.
        child (nn.Module): The child module to parse.
        parse_model (ParseModel): The parser entry point used recursively.
        skipped (dict[str, str]): The skipped children recorded so far,
            updated in place with the name and the reason of this child.

    Returns:
        ModelBlockSchema | None: The parsed schema of the child, or None
            when the child forward uses unsupported syntax.
    """

    # Try to parse the child and skip it on unsupported syntax
    try:
        return parse_model(model=child, inputs_tensors_shapes={})
    except UnsupportedSyntaxError as error:
        print(f"WARNING: skipped unparsable sub-block '{name}': {error}")
        skipped[name] = str(error)
        return None


def parse_container(
    name: str,
    module: nn.Module,
    parse_model: ParseModel,
    skipped: dict[str, str]
) -> dict[str, ModelBlockSchema]:
    """
    Parse a sub-block container entry by entry.

    The containers (nn.ModuleList, nn.ModuleDict, nn.Sequential) are expanded
    into indexed sub-block entries keyed by "name.key".

    Args:
        name (str): The attribute name of the container in its parent.
        module (nn.Module): The container module to parse.
        parse_model (ParseModel): The parser entry point used recursively.
        skipped (dict[str, str]): The skipped children recorded so far,
            updated in place with the skipped entries of the container.

    Returns:
        dict[str, ModelBlockSchema]: The parsed entries keyed by "name.key".
    """

    # Initialize the parsed entries of the container
    parsed: dict[str, ModelBlockSchema] = {}

    # A module list is parsed entry by entry with integer keys
    entry: nn.Module
    if isinstance(module, (nn.ModuleList, nn.Sequential)):
        index: int
        for index, entry in enumerate(module):
            parsed_entry: ModelBlockSchema | None = parse_child(
                name=f"{name}.{index}",
                child=entry,
                parse_model=parse_model,
                skipped=skipped
            )
            if parsed_entry is not None:
                parsed[f"{name}.{index}"] = parsed_entry

    # A module dict is parsed entry by entry with string keys
    elif isinstance(module, nn.ModuleDict):
        key: str
        for key, entry in module.items():
            parsed_entry = parse_child(
                name=f"{name}.{key}",
                child=entry,
                parse_model=parse_model,
                skipped=skipped
            )
            if parsed_entry is not None:
                parsed[f"{name}.{key}"] = parsed_entry

    # Return the parsed entries of the container
    return parsed


def collect_sub_blocks(
    model: nn.Module,
    parse_model: ParseModel
) -> tuple[dict[str, ModelBlockSchema], set[str]]:
    """
    Collect the sub-blocks and the container names of a model.

    Args:
        model (nn.Module): The model whose children are collected.
        parse_model (ParseModel): The parser entry point used recursively.

    Returns:
        tuple[dict[str, ModelBlockSchema], set[str], set[str],
            dict[str, str]]: The parsed sub-blocks by name, the names of the
            sub-block containers, the names of the sub-block dictionaries and
            the skipped children by name with their skip reasons.
    """

    # Initialize the skipped children of the model
    skipped: dict[str, str] = {}

    # Initialize the names of the containers of the model
    container_names: set[str] = set()

    # Initialize the names of the sub-block dictionaries of the model
    dict_container_names: set[str] = set()

    # Initialize the parsed sub-blocks of the model
    model_sub_blocks: dict[str, ModelBlockSchema] = {}

    # Check each direct child of the model
    child_name: str
    child: nn.Module
    for child_name, child in model.named_children():

        # A container is parsed entry by entry with indexed keys
        if isinstance(child, (nn.ModuleList, nn.ModuleDict, nn.Sequential)):
            container_names.add(child_name)
            if isinstance(child, nn.ModuleDict):
                dict_container_names.add(child_name)
            model_sub_blocks.update(
                parse_container(
                    name=child_name,
                    module=child,
                    parse_model=parse_model,
                    skipped=skipped
                )
            )

        # A direct sub-block is parsed recursively
        else:
            parsed_child: ModelBlockSchema | None = parse_child(
                name=child_name,
                child=child,
                parse_model=parse_model,
                skipped=skipped
            )
            if parsed_child is not None:
                model_sub_blocks[child_name] = parsed_child

    # Return the parsed sub-blocks, the container names and the skipped
    return model_sub_blocks, container_names, dict_container_names, skipped
