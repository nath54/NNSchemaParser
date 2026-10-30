"""
Tests of the NNParser on a mixture of experts style model.

Run with: python tests/test_parser.py
"""

# Import Modules
from typing import Any

import torch
from torch import Tensor, nn

from nnschemaparser.core.instructions import BlockInstruction, ControlInstruction, OpInstruction
from nnschemaparser.core.model_schema import ModelBlockSchema
from nnschemaparser.parser.parser import NNParser
from nnschemaparser.interpreter import SchemaInterpreter


class Expert(nn.Module):
    """
    Small expert block used as a container entry.
    """

    def __init__(self, hidden: int) -> None:
        """
        Init function.

        Args:
            hidden (int): The hidden size of the expert.
        """

        # Init parent
        super().__init__()

        # Store here the hidden size of the expert
        self.hidden: int = hidden

        # Create the weight of the expert
        self.w: nn.Parameter = nn.Parameter(torch.randn(hidden, hidden) * 0.1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the expert transformation of the input.

        Args:
            x (Tensor): The input tensor of the expert.

        Returns:
            Tensor: The transformed input tensor.
        """

        # Compute the linear transformation of the input
        y: Tensor = x @ self.w

        # Return the activated transformation
        return y.relu()


class MoEBlock(nn.Module):
    """
    Mixture of experts style block with a loop, an if and a sub-block.
    """

    def __init__(self, hidden: int, n_experts: int) -> None:
        """
        Init function.

        Args:
            hidden (int): The hidden size of the block.
            n_experts (int): The number of experts of the block.
        """

        # Init parent
        super().__init__()

        # Store here the configuration of the block
        self.hidden: int = hidden
        self.use_extra: bool = True

        # Create the experts container of the block
        self.experts: nn.ModuleList = nn.ModuleList(
            [Expert(hidden=hidden) for _ in range(n_experts)]
        )

        # Create the gate of the block, a per-token scale
        self.gate: nn.Linear = nn.Linear(hidden, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the mixture of experts forward of the input.

        Args:
            x (Tensor): The input tensor of the block.

        Returns:
            Tensor: The output tensor of the block.
        """

        # Compute the gate of the block
        h: Tensor = self.gate(x)

        # Compute the output of each expert
        outs: list = []
        i: int
        for i in range(2):
            outs.append(self.experts[i](x))

        # Sum the outputs of the experts
        z: Tensor = outs[0] + outs[1]

        # Apply the extra activation conditionally
        if self.use_extra:
            z = z.relu()
        else:
            z = z.sigmoid()

        # Return the output of the block
        return z + h


def main() -> None:
    """
    Run the parser test on the mixture of experts block.
    """

    # Create the model and the input tensor of the test
    model: MoEBlock = MoEBlock(hidden=8, n_experts=2).eval()
    x: Tensor = torch.randn(4, 8)

    # Parse the model into a schema
    schema: ModelBlockSchema = NNParser.parse_model(
        model=model,
        inputs_tensors_shapes={"x": (4, 8)}
    )

    # Check the inputs, outputs, sub-blocks and weights of the schema
    print("inputs:", {k: (v.shape, v.dtype) for k, v in schema.inputs.items()})
    print("outputs:", [(o.shape, o.dtype) for o in schema.outputs])
    print("sub_blocks:", sorted(schema.sub_blocks.keys()))
    print("weights:", sorted(schema.weights.keys()))
    print("gate weights:", sorted(schema.sub_blocks["gate"].weights.keys()))

    # Check the resolved instructions of the schema
    instr: Any
    for instr in schema.body:
        if isinstance(instr, OpInstruction):
            print(f"op {instr.qualified_name:10s} -> {instr.resolved.shape} {instr.resolved.dtype}")
        elif isinstance(instr, BlockInstruction):
            print(
                f"block {instr.block_ref.path}[{instr.block_ref.index_var}]"
                f" -> {instr.resolved.shape}"
            )
        elif isinstance(instr, ControlInstruction):
            ci: Any
            for ci in instr.body:
                if isinstance(ci, BlockInstruction):
                    print(
                        f"loop block {ci.block_ref.path}[{ci.block_ref.index_var}]"
                        f" -> {ci.resolved.shape}"
                    )

    # Check the sub-block inputs and outputs filled by the resolution
    expert0: ModelBlockSchema = schema.sub_blocks["experts.0"]
    print("expert0 inputs:", [(v.shape, v.dtype) for v in expert0.inputs.values()])
    print("expert0 outputs:", [(o.shape, o.dtype) for o in expert0.outputs])
    print(
        "expert0 body:",
        [(i.qualified_name, i.resolved.shape) for i in expert0.body if isinstance(i, OpInstruction)]
    )

    # Check the values of the schema execution against the real forward
    out: list = SchemaInterpreter(capture_values=True).run(
        block=schema,
        scope={"x": x.clone()}
    )
    matches: bool = torch.allclose(out[0], model(x), atol=1e-6)
    print("interpreter == model.forward:", matches)
    assert matches, "The schema execution differs from the real forward."


def test_parser() -> None:
    """
    Run the parser test of the module for the pytest discovery.
    """

    torch.manual_seed(0)
    main()


# Main entry of the test file
if __name__ == "__main__":
    torch.manual_seed(0)
    main()
