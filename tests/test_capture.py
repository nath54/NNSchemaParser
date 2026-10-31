"""
Tests of the capture_run convenience and the ExecutionRecord.

Run with: python tests/test_capture.py
"""

# Import Modules
import torch
from torch import Tensor, nn

from nnschemaparser import NNParser, capture_run


class MLP(nn.Module):
    """
    Small two layers MLP used as the test model.
    """

    def __init__(self) -> None:
        """
        Init function.
        """

        # Init parent
        super().__init__()

        # Create the layers of the MLP
        self.fc1: nn.Linear = nn.Linear(8, 16)
        self.fc2: nn.Linear = nn.Linear(16, 4)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward function.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """

        # The hidden layer with its activation and the output layer
        hidden: Tensor = self.fc1(x)
        activated: Tensor = hidden.relu()
        return self.fc2(activated)


def main() -> None:
    """
    Run the capture run tests.
    """

    # The test model and its schema
    model: MLP = MLP().eval()
    schema = NNParser.parse_model(
        model=model,
        inputs_tensors_shapes={"x": (2, 8)},
        inputs_tensors_dtypes={"x": "torch.float32"}
    )

    # The captured run of a fixed input, with the gradients capturable
    x: Tensor = torch.randn(2, 8)
    record = capture_run(
        block=schema,
        scope={"x": x.clone().requires_grad_(True)},
        capture_grads=True
    )

    # The outputs match the real forward
    assert torch.allclose(record.outputs[0], model(x), atol=1e-6)

    # The activation trace holds the seven steps of the MLP
    assert len(record.trace) == 7, f"steps: {len(record.trace)}"
    assert record.trace.steps[0].description == "Initial input tensor"
    assert record.trace.steps[0].frame_depth == 0
    assert record.trace.steps[1].frame_depth == 1

    # The final scope of the run holds the intermediate variables
    assert "activated" in record.scope

    # The gradients are empty before a backward pass
    assert len(record.gradients) == 0

    # The gradient trace after a backward pass on the recorded outputs
    loss: Tensor = record.outputs[0].pow(2).sum()
    loss.backward()
    gradients = record.gradients

    # The gradient steps mirror the activation steps and carry values
    assert len(gradients) == len(record.trace), \
        f"gradients: {len(gradients)} vs {len(record.trace)}"
    for activation_step, gradient_step in zip(record.trace.steps, gradients.steps):
        assert gradient_step.source_container_id == activation_step.container_id

    # The gradients property is cached and stable
    assert record.gradients is gradients

    # All the assertions passed
    print("tests/test_capture.py: all assertions passed")


if __name__ == "__main__":
    main()
