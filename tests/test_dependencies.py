"""
Tests of the instruction dependencies and the method call sites.

Run with: python tests/test_dependencies.py
"""

# Import Modules
from typing import Any

import torch
from torch import Tensor, nn

from nnschemaparser import NNParser
from nnschemaparser.core.dependencies import (
    Dependency,
    instruction_dependencies,
    method_call_sites
)
from nnschemaparser.core.instructions import (
    BlockInstruction,
    OpInstruction,
    ReturnInstruction
)


class Scale(nn.Module):
    """
    A sub-block scaling its input by a learned factor.
    """

    def __init__(self) -> None:
        """
        Init function.
        """

        # Init parent
        super().__init__()

        # Create the factor of the scale
        self.factor: nn.Parameter = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward function.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The scaled tensor.
        """

        # The single scaling of the sub-block
        return x * self.factor


class Model(nn.Module):
    """
    The test model with a sub-block, a weight, a method and a branch.
    """

    def __init__(self) -> None:
        """
        Init function.
        """

        # Init parent
        super().__init__()

        # Create the sub-block and the weight of the model
        self.scale: Scale = Scale()
        self.bias: nn.Parameter = nn.Parameter(torch.zeros(1))

    def helper(self, x: Tensor) -> Tensor:
        """
        A helper method called by the forward.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The shifted tensor.
        """

        # The single shift of the helper
        return x + 1.0

    def forward(self, x: Tensor, flag: bool = True) -> Tensor:
        """
        Forward function.

        Args:
            x (Tensor): The input tensor.
            flag (bool): The branch selector.

        Returns:
            Tensor: The output tensor.
        """

        # The sub-block call, the two method calls and the weighted product
        s: Tensor = self.scale(x)
        h1: Tensor = self.helper(s)
        h2: Tensor = self.helper(h1)
        t: Tensor = s * self.bias

        # The branch mixing the helper and product results
        if flag:
            r: Tensor = h1 + t
        else:
            r = h2
        return r


def main() -> None:
    """
    Run the dependency tests.
    """

    # The parsed schema of the test model
    model: Model = Model().eval()
    schema = NNParser.parse_model(
        model=model,
        inputs_tensors_shapes={"x": (1, 4)},
        inputs_tensors_dtypes={"x": "torch.float32"},
    )

    # The method call sites: the helper is called twice at the top level
    sites: dict[str, list[int]] = method_call_sites(block=schema)
    assert set(sites.keys()) == {"helper"}, sites
    assert len(sites["helper"]) == 2, sites
    assert sites["helper"][0] < sites["helper"][1]

    # The dependencies of the body instructions
    dependencies: dict[str, list[Dependency]] = instruction_dependencies(
        block=schema
    )

    # The sub-block call reads only the declared input of the model
    block_paths: list[str] = [
        path
        for path, instruction in _instruction_items(block=schema)
        if isinstance(instruction, BlockInstruction)
    ]
    scale_deps: list[Dependency] = dependencies[block_paths[0]]
    assert len(scale_deps) == 1
    assert scale_deps[0].name == "x" and scale_deps[0].kind == "input" \
        and scale_deps[0].origin == "x"

    # The product reads the sub-block output and the bias weight directly
    mul_paths: list[str] = [
        path
        for path, instruction in _instruction_items(block=schema)
        if isinstance(instruction, OpInstruction)
        and instruction.qualified_name == "mul"
    ]
    assert len(mul_paths) == 1
    mul_deps: list[Dependency] = dependencies[mul_paths[0]]
    by_name: dict[str, Dependency] = {dep.name: dep for dep in mul_deps}

    # The s variable points back to the sub-block call instruction
    assert by_name["s"].kind == "instruction" and by_name["s"].origin == "0"

    # The bias variable is a weight of the model block
    assert by_name["bias"].kind == "weight" and by_name["bias"].origin == "bias"

    # The branch add reads the helper result and the product, both bound by
    # top level instructions of the body
    add_paths: list[str] = [
        path
        for path, instruction in _instruction_items(block=schema)
        if isinstance(instruction, OpInstruction)
        and instruction.qualified_name == "add"
        and "/body/" in path
    ]
    assert len(add_paths) == 1
    add_deps: list[Dependency] = dependencies[add_paths[0]]
    assert all(dep.kind == "instruction" for dep in add_deps), add_deps
    assert all("/" not in dep.origin for dep in add_deps), add_deps

    # The condition of the if reads the declared flag input; the lowering
    # hoists the condition op into the body, so its path is its body index
    if_index: int = int(add_paths[0].split("/")[0])
    control: Any = schema.body[if_index]
    condition_path: str = next(
        path
        for path, instruction in _instruction_items(block=schema)
        if instruction is control.condition
    )
    condition_deps: list[Dependency] = dependencies[condition_path]
    assert any(
        dep.name == "flag" and dep.kind == "input" for dep in condition_deps
    ), condition_deps

    # The return reads the r variable last bound in the else body
    return_paths: list[str] = [
        path
        for path, instruction in _instruction_items(block=schema)
        if isinstance(instruction, ReturnInstruction)
    ]
    assert len(return_paths) == 1
    return_deps: list[Dependency] = dependencies[return_paths[0]]
    assert return_deps[0].name == "r" and return_deps[0].kind == "instruction"
    assert return_deps[0].origin == f"{if_index}/else/0"

    # All the assertions passed
    print("tests/test_dependencies.py: all assertions passed")
    print("  method sites:", sites)
    for path in sorted(dependencies, key=_path_sort_key):
        print(f"  {path}: {dependencies[path]}")


def _instruction_items(
    block: Any
) -> list[tuple[str, Any]]:
    """
    Flatten the instructions of a block schema with their paths.

    Args:
        block (Any): The block schema to flatten.

    Returns:
        list[tuple[str, Any]]: The (path, instruction) pairs of the whole
        body tree, in walk order.
    """

    # The flattened pairs of the body tree
    pairs: list[tuple[str, Any]] = []

    # The recursive walk of the instruction lists
    def _walk(instructions: Any, prefix: str) -> None:
        for index, instruction in enumerate(instructions):
            path: str = f"{prefix}/{index}" if prefix else str(index)
            pairs.append((path, instruction))
            if hasattr(instruction, "body"):
                _walk(instruction.body, f"{path}/body")
            if hasattr(instruction, "else_body"):
                _walk(instruction.else_body, f"{path}/else")

    # The walk of the top level body
    _walk(block.body, "")
    return pairs


def _path_sort_key(path: str) -> tuple:
    """
    Get the numeric sort key of an instruction path.

    Args:
        path (str): The instruction path to sort.

    Returns:
        tuple: The numeric parts of the path segments.
    """

    # The numeric key of each slash separated segment
    key: list = []
    segment: str
    for segment in path.split("/"):
        key.append(int(segment) if segment.isdigit() else segment)
    return tuple(key)


if __name__ == "__main__":
    main()
