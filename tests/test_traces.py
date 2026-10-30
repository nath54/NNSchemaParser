"""
Tests of the activation and gradient traces.

Run with: python tests/test_traces.py
"""

# Import Modules
import torch
from torch import Tensor, nn

from nnschemaparser import NNParser, SchemaInterpreter, ActivationRecorder
from nnschemaparser.core.traces import TensorTrace


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
        self.fc1: nn.Linear = nn.Linear(16, 32)
        self.fc2: nn.Linear = nn.Linear(32, 8)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward function.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """

        # The hidden layer with its activation
        hidden: Tensor = self.fc1(x)
        activated: Tensor = hidden.relu()

        # The output layer
        return self.fc2(activated)


def main() -> None:
    """
    Run the activation and gradient trace tests.
    """

    # The test model and its input
    model: MLP = MLP().eval()
    x: Tensor = torch.randn(2, 16)

    # Parse the model into a schema
    schema = NNParser.parse_model(
        model=model,
        inputs_tensors_shapes={"x": (2, 16)},
        inputs_tensors_dtypes={"x": "torch.float32"}
    )

    # Record the activations of an interpreted forward pass, with an input
    # requiring grad so the input gradients are captured too
    recorder: ActivationRecorder = ActivationRecorder(capture_grads=True)
    outputs: list[Tensor] = SchemaInterpreter(
        capture_values=True,
        listeners=[recorder]
    ).run(block=schema, scope={"x": x.clone().requires_grad_(True)})

    # The reference values of the real forward
    reference: Tensor = model(x)

    # The activation trace is chronological, with the nested operations of
    # the sub-blocks recorded between the block call events: the input, the
    # F.linear of fc1, the fc1 call, the relu, the F.linear of fc2, the fc2
    # call and the model output
    trace: TensorTrace = recorder.trace
    assert len(trace) == 7, f"expected 7 activation steps, got {len(trace)}"
    assert trace.steps[0].description == "Initial input tensor"
    assert trace.steps[0].container_id == "input_0001"
    assert trace.steps[0].slot_type == "input"
    assert trace.steps[0].tensor.shape == (2, 16)
    assert trace.steps[1].description == "Linear projection"
    assert trace.steps[1].block_path == "fc1"
    assert trace.steps[2].description == "Call to sub-block fc1 (Linear)"
    assert trace.steps[2].out_var == "hidden"
    assert trace.steps[2].container_id == "MLP_0003"
    assert trace.steps[3].description == "ReLU activation"
    assert trace.steps[3].out_var == "activated"
    assert trace.steps[4].block_path == "fc2"
    assert trace.steps[5].description == "Call to sub-block fc2 (Linear)"
    assert trace.steps[6].description == "Model output tensor"
    assert trace.steps[6].slot_type == "output"
    assert trace.steps[6].container_id == "output_0007"
    assert torch.equal(trace.steps[6].tensor.value, reference)

    # The captured values match the real forward
    assert torch.equal(outputs[0], reference)

    # Run a backward pass and collect the cross-referenced gradients
    loss: Tensor = outputs[0].pow(2).sum()
    loss.backward()
    gradients: TensorTrace = recorder.collect_gradients()

    # The gradient steps mirror the activation steps in the same order
    assert len(gradients) == len(trace), \
        f"expected {len(trace)} gradient steps, got {len(gradients)}"
    for activation_step, gradient_step in zip(trace.steps, gradients.steps):
        assert gradient_step.container_id == f"grad_{activation_step.container_id}"
        assert gradient_step.source_container_id == activation_step.container_id
        assert gradient_step.tensor.shape == activation_step.tensor.shape

    # The gradient of the input matches the autograd reference
    x_grad_ref: Tensor = x.clone().requires_grad_(True)
    model(x_grad_ref).pow(2).sum().backward()
    assert x_grad_ref.grad is not None
    assert torch.allclose(
        gradients.steps[0].tensor.value,
        x_grad_ref.grad,
        atol=1e-6
    )

    # The lookups and the text dump of the traces
    assert trace.by_container("MLP_0003") is trace.steps[2]
    assert trace.by_container("unknown") is None
    trace.save("outputs/activation_trace.txt")
    gradients.save("outputs/gradient_trace.txt")

    # All the assertions passed
    print("tests/test_traces.py: all assertions passed")


if __name__ == "__main__":
    main()
