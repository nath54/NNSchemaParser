"""
Tests of the sub-block call sites of a schema body.

Run with: python tests/test_call_sites.py
"""

# Import Modules
from typing import Any

import torch
from torch import Tensor, nn

from nnschemaparser import NNParser
from nnschemaparser.core.instructions import ControlInstruction


class Inner(nn.Module):
    """
    A sub-block called once at the top level of the outer body.
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward function.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The transformed tensor.
        """

        # The single transformation of the inner block
        return x * 2.0


class Twin(nn.Module):
    """
    A sub-block called twice at the top level of the outer body.
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward function.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The transformed tensor.
        """

        # The single transformation of the twin block
        return x + 1.0


class Branch(nn.Module):
    """
    A sub-block called only inside a conditional body.
    """

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward function.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The transformed tensor.
        """

        # The single transformation of the branch block
        return x - 1.0


class Expert(nn.Module):
    """
    An expert block iterated by a container loop.
    """

    def __init__(self) -> None:
        """
        Init function.
        """

        # Init parent
        super().__init__()

        # Create the weight of the expert
        self.w: nn.Parameter = nn.Parameter(torch.ones(1))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward function.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The transformed tensor.
        """

        # The single transformation of the expert
        return x * self.w


class Outer(nn.Module):
    """
    The test model calling its sub-blocks in every layout situation.
    """

    def __init__(self) -> None:
        """
        Init function.
        """

        # Init parent
        super().__init__()

        # Create the sub-blocks of the test model
        self.once: Inner = Inner()
        self.twin: Twin = Twin()
        self.branch: Branch = Branch()
        self.experts: nn.ModuleList = nn.ModuleList(
            [Expert() for _ in range(2)]
        )

    def forward(self, x: Tensor, flag: bool = True) -> Tensor:
        """
        Forward function.

        Args:
            x (Tensor): The input tensor.
            flag (bool): The branch selector.

        Returns:
            Tensor: The output tensor.
        """

        # The once called sub-block
        a: Tensor = self.once(x)

        # The twice called sub-block
        b: Tensor = self.twin(a)
        c: Tensor = self.twin(b)

        # The conditionally called sub-block
        if flag:
            d: Tensor = self.branch(c)
        else:
            d = c

        # The container iterated experts
        mixed: Tensor = d
        for expert in self.experts:
            mixed = expert(mixed)
        return mixed


def main() -> None:
    """
    Run the call site tests.
    """

    # The parsed schema of the test model
    model: Outer = Outer().eval()
    schema = NNParser.parse_model(
        model=model,
        inputs_tensors_shapes={"x": (1, 4)},
        inputs_tensors_dtypes={"x": "torch.float32"},
    )

    # The top level indices of the control flow instructions of the body
    control_indices: list[int] = [
        index
        for index, instruction in enumerate(schema.body)
        if isinstance(instruction, ControlInstruction)
    ]
    loop_index: int = control_indices[-1]
    branch_index: int = control_indices[0]

    # The call sites of the body
    sites: dict[str, list[int]] = schema.sub_block_call_sites()

    # The sub-block called once anchors at its own top level index
    assert sites.get("once") is not None and len(sites["once"]) == 1, sites

    # The sub-block called twice has two distinct top level anchors
    assert sites.get("twin") is not None and len(sites["twin"]) == 2, sites
    assert sites["twin"][0] < sites["twin"][1]

    # The conditionally called sub-block anchors at the enclosing if index
    assert sites.get("branch") == [branch_index], sites

    # The container experts anchor at the enclosing for index
    assert sites.get("experts") == [loop_index], sites

    # The schema still executes like the model
    interpreter_outputs = __import__("nnschemaparser").SchemaInterpreter().run(
        block=schema,
        scope={"x": torch.randn(1, 4), "flag": True},
    )
    reference: Tensor = model(torch.randn(1, 4), True)
    assert interpreter_outputs[0].shape == reference.shape

    # All the assertions passed
    print("tests/test_call_sites.py: all assertions passed")
    print("  call sites:", sites)


if __name__ == "__main__":
    main()
