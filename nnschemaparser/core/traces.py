"""
Activation and gradient traces of the schema interpreter executions.

A trace is the chronological record of one execution: every captured tensor
becomes a TensorContainer referenced by a TraceStep holding its step number,
its container id, a human readable description and its position in the model
tree. The ActivationTrace records the forward activations, the GradientTrace
records the backward gradients of the same tensors, cross-referenced by their
source container ids.
"""

# Import Modules
from nnschemaparser.core.tensor_schema import TensorContainer


class TraceStep:
    """
    One entry of a tensor trace.

    A step describes one captured tensor of an execution: its chronological
    step number, the stable container id of its TensorContainer, a human
    readable description of the operation that produced it, the original
    variable name it was bound to, the operation display name, the path of
    the block that executed it and, for gradients, the container id of the
    activation step it is the gradient of.
    """

    def __init__(
        self,
        step: int,
        container_id: str,
        description: str,
        slot_type: str,
        out_var: str,
        op_name: str,
        block_path: str,
        tensor: TensorContainer,
        source_container_id: str | None = None
    ) -> None:
        """
        Init function.

        Args:
            step (int): The 1-based chronological step number of the capture.
            container_id (str): The stable container id of the captured
                tensor (e.g. "input_0001" or "layers_0_0002").
            description (str): The human readable description of the step
                (e.g. "Initial input tensor", "Tensor multiplication").
            slot_type (str): The role of the tensor, one of "input",
                "activation", "output" or "gradient".
            out_var (str): The original variable name the tensor was bound
                to in the forward source ("" when unnamed).
            op_name (str): The display name of the producing operation, the
                qualified name of the torch callable or the path of the
                called sub-block.
            block_path (str): The attribute path of the block that produced
                the tensor ("" for the root block inputs).
            tensor (TensorContainer): The captured tensor itself.
            source_container_id (str | None): For gradient steps, the
                container id of the activation step the gradient belongs to.
        """

        # Store here the chronological step number of the capture
        self.step: int = step

        # Store here the stable container id of the captured tensor
        self.container_id: str = container_id

        # Store here the human readable description of the step
        self.description: str = description

        # Store here the role of the captured tensor
        self.slot_type: str = slot_type

        # Store here the original variable name of the tensor
        self.out_var: str = out_var

        # Store here the display name of the producing operation
        self.op_name: str = op_name

        # Store here the attribute path of the producing block
        self.block_path: str = block_path

        # Store here the captured tensor of the step
        self.tensor: TensorContainer = tensor

        # Store here the activation container id of a gradient step
        self.source_container_id: str | None = source_container_id


def format_step(step: TraceStep) -> str:
    """
    Format a trace step as one human readable line.

    Args:
        step (TraceStep): The trace step to format.

    Returns:
        str: E.g. "step 0002: Tensor multiplication | container_id:
        layers_0_0002 | torch.float32[2, 256] | var: hidden | block:
        layers.0".
    """

    # The shape and dtype of the captured tensor
    shape: str = ""
    if step.tensor is not None:
        shape = f"{step.tensor.dtype}{list(step.tensor.shape)}"

    # The optional original variable name of the tensor
    var: str = f" | var: {step.out_var}" if step.out_var else ""

    # The optional attribute path of the producing block
    block: str = f" | block: {step.block_path}" if step.block_path else ""

    # The optional source container id of a gradient step
    source: str = ""
    if step.source_container_id is not None:
        source = f" | source: {step.source_container_id}"

    # The full one line description of the step
    return (
        f"step {step.step:04d}: {step.description}"
        f" | container_id: {step.container_id}"
        f" | {shape}{var}{block}{source}"
    )


class TensorTrace:
    """
    Chronological record of the captured tensors of one execution.

    A trace holds the ordered list of the TraceStep entries of an execution,
    each referencing its TensorContainer by its stable container id. The
    concrete ActivationTrace and GradientTrace subclasses set the kind of
    the recorded tensors.
    """

    def __init__(self, kind: str) -> None:
        """
        Init function.

        Args:
            kind (str): The kind of recorded tensors, "activation" or
                "gradient".
        """

        # Store here the kind of the recorded tensors
        self.kind: str = kind

        # Store here the ordered steps of the execution
        self.steps: list[TraceStep] = []

    def __len__(self) -> int:
        """
        Get the number of recorded steps.

        Returns:
            int: The number of steps of the trace.
        """

        # The step count of the trace
        return len(self.steps)

    def by_container(self, container_id: str) -> TraceStep | None:
        """
        Get a step of the trace by its container id.

        Args:
            container_id (str): The stable container id to look up.

        Returns:
            TraceStep | None: The matching step, or None when unknown.
        """

        # Search the steps for the requested container id
        step: TraceStep
        for step in self.steps:
            if step.container_id == container_id:
                return step

        # No step matches the requested container id
        return None

    def save(self, path: str) -> None:
        """
        Write the trace to a text file, one line per step.

        Args:
            path (str): The path of the output text file.
        """

        # The full trace, one formatted line per step
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(
                "\n".join(format_step(step=step) for step in self.steps) + "\n"
            )


class ActivationTrace(TensorTrace):
    """
    Chronological record of the forward activations of an execution.

    The steps are ordered by execution: the inputs of the root block first
    ("Initial input tensor"), then one step per captured tensor produced by
    an operation or a sub-block call.
    """

    def __init__(self) -> None:
        """
        Init function.
        """

        # Init parent with the activation kind
        super().__init__(kind="activation")


class GradientTrace(TensorTrace):
    """
    Chronological record of the backward gradients of an execution.

    The steps mirror the steps of the matching ActivationTrace in the same
    order: the container id of a gradient step is the container id of its
    activation step prefixed with "grad_", and its source_container_id
    points back to the activation step.
    """

    def __init__(self) -> None:
        """
        Init function.
        """

        # Init parent with the gradient kind
        super().__init__(kind="gradient")
