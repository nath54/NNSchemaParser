"""
Tests of the trace comparison of two schema executions.

Run with: python tests/test_diff.py
"""

# Import Modules
from typing import Any, Generator

import torch
from torch import Tensor, nn

from nnschemaparser import (
    NNParser,
    SchemaInterpreter,
    StepInfo,
    TensorContainer,
    TensorTrace,
    TraceStep,
    compare_traces,
    diff_summary
)
from nnschemaparser.core.diff import TraceDiff
from nnschemaparser.interpreter.record import ActivationRecorder


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
        self.fc1: nn.Linear = nn.Linear(4, 8)
        self.fc2: nn.Linear = nn.Linear(8, 2)

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


def build_schema() -> Any:
    """
    Build the test schema.

    Returns:
        Any: The parsed schema of the test model.
    """

    # The schema of the test model
    model: MLP = MLP().eval()
    return NNParser.parse_model(
        model=model,
        inputs_tensors_shapes={"x": (2, 4)},
        inputs_tensors_dtypes={"x": "torch.float32"}
    )


def test_compare_edited_run() -> None:
    """
    Check the comparison of a base run and a mid-run edited fork.
    """

    # The test schema and input
    schema: Any = build_schema()
    x: Tensor = torch.randn(2, 4)

    # The base captured run
    base_record = __import__("nnschemaparser").capture_run(
        block=schema,
        scope={"x": x.clone()}
    )

    # The fork: a captured stepped run editing the activated variable right
    # after the ReLU produced it
    fork_recorder: ActivationRecorder = ActivationRecorder()
    fork_interpreter: SchemaInterpreter = SchemaInterpreter(
        capture_values=True,
        listeners=[fork_recorder]
    )
    edited: list[bool] = [False]
    generator: Generator[StepInfo, None, list[Tensor]] = fork_interpreter.step_run(
        block=schema,
        scope={"x": x.clone()}
    )
    while True:
        try:
            step: StepInfo = next(generator)

            # The edit targets the activated variable of the root frame
            if not edited[0] and step.kind == "op" and step.out_var == "activated" \
                    and isinstance(step.value, Tensor):
                edited[0] = True
                step.edit(values=torch.zeros_like(step.value))
        except StopIteration:
            break

    # The edit was applied once and the traces have the same step count
    assert edited[0]
    assert len(fork_recorder.trace) == len(base_record.trace)

    # The comparison of the base and the fork traces
    diffs: list[TraceDiff] = compare_traces(
        trace_a=base_record.trace,
        trace_b=fork_recorder.trace
    )

    # All the steps are aligned by container id with computable values
    assert len(diffs) == len(base_record.trace)
    assert all(diff.status == "ok" and diff.matched_by == "container_id"
               for diff in diffs)

    # The steps recorded up to the edit are unchanged (the edit applies
    # after the ReLU step was recorded), the downstream steps are changed
    for diff in diffs:
        if diff.container_id_a in ("input_0001", "fc1_0002", "MLP_0003", "MLP_0004"):
            assert diff.changed is False, f"unexpected change on {diff.container_id_a}"
        elif diff.container_id_a in ("fc2_0005", "MLP_0006", "output_0007"):
            assert diff.changed is True, f"missing change on {diff.container_id_a}"

    # The summary counts only the ok status
    assert diff_summary(diffs=diffs) == {"ok": len(diffs)}

    print("test_diff.edited: OK")


def test_compare_manual_traces() -> None:
    """
    Check the statuses of the comparison on hand built traces.
    """

    # A first trace with two steps
    trace_a: TensorTrace = TensorTrace(kind="activation")
    trace_a.steps.append(TraceStep(
        step=1,
        container_id="fc1_0001",
        description="Linear projection",
        slot_type="activation",
        out_var="hidden",
        op_name="F.linear",
        block_path="fc1",
        tensor=TensorContainer(
            shape=(2, 3),
            dtype="torch.float32",
            value=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            container_id="fc1_0001",
            slot_type="activation"
        )
    ))
    trace_a.steps.append(TraceStep(
        step=2,
        container_id="fc2_0002",
        description="Linear projection",
        slot_type="activation",
        out_var="out",
        op_name="F.linear",
        block_path="fc2",
        tensor=TensorContainer(
            shape=(2, 2),
            dtype="torch.float32",
            value=torch.zeros(2, 2),
            container_id="fc2_0002",
            slot_type="activation"
        )
    ))

    # A second trace with a different value, a reshaped step and an extra
    trace_b: TensorTrace = TensorTrace(kind="activation")
    trace_b.steps.append(TraceStep(
        step=1,
        container_id="fc1_0001",
        description="Linear projection",
        slot_type="activation",
        out_var="hidden",
        op_name="F.linear",
        block_path="fc1",
        tensor=TensorContainer(
            shape=(2, 3),
            dtype="torch.float32",
            value=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 9.0]]),
            container_id="fc1_0001",
            slot_type="activation"
        )
    ))
    trace_b.steps.append(TraceStep(
        step=2,
        container_id="fc2_0002",
        description="Linear projection",
        slot_type="activation",
        out_var="out",
        op_name="F.linear",
        block_path="fc2",
        tensor=TensorContainer(
            shape=(2, 3),
            dtype="torch.float32",
            value=torch.zeros(2, 3),
            container_id="fc2_0002",
            slot_type="activation"
        )
    ))
    trace_b.steps.append(TraceStep(
        step=3,
        container_id="head_0003",
        description="Softmax normalization",
        slot_type="activation",
        out_var="probs",
        op_name="F.softmax",
        block_path="head",
        tensor=TensorContainer(
            shape=(2, 2),
            dtype="torch.float32",
            value=torch.zeros(2, 2),
            container_id="head_0003",
            slot_type="activation"
        )
    ))

    # The comparison of the two traces
    diffs: list[TraceDiff] = compare_traces(trace_a=trace_a, trace_b=trace_b, atol=1e-6)

    # The first step is aligned by id with a changed value
    assert diffs[0].status == "ok"
    assert diffs[0].matched_by == "container_id"
    assert diffs[0].changed is True
    assert diffs[0].max_abs_diff == 3.0

    # The second step is aligned but reshaped
    assert diffs[1].status == "shape_changed"
    assert diffs[1].max_abs_diff is None

    # The extra step of the second trace is reported at the end
    assert diffs[2].status == "only_in_b"
    assert diffs[2].container_id_b == "head_0003"

    # The summary counts the statuses
    summary: dict[str, int] = diff_summary(diffs=diffs)
    assert summary == {"ok": 1, "shape_changed": 1, "only_in_b": 1}

    # A comparison in the other direction reports the missing step
    reverse_diffs: list[TraceDiff] = compare_traces(trace_a=trace_b, trace_b=trace_a)
    assert reverse_diffs[2].status == "only_in_a"

    # A role fallback match survives a container id change
    trace_c: TensorTrace = TensorTrace(kind="activation")
    trace_c.steps.append(TraceStep(
        step=1,
        container_id="fc1_0009",
        description="Linear projection",
        slot_type="activation",
        out_var="hidden",
        op_name="F.linear",
        block_path="fc1",
        tensor=TensorContainer(
            shape=(2, 3),
            dtype="torch.float32",
            value=torch.zeros(2, 3),
            container_id="fc1_0009",
            slot_type="activation"
        )
    ))
    role_diffs: list[TraceDiff] = compare_traces(trace_a=trace_a, trace_b=trace_c)
    assert role_diffs[0].status == "ok"
    assert role_diffs[0].matched_by == "role"
    assert role_diffs[0].container_id_a == "fc1_0001"
    assert role_diffs[0].container_id_b == "fc1_0009"

    print("test_diff.manual: OK")


def main() -> None:
    """
    Run the diff tests.
    """

    # All the diff tests in order
    test_compare_edited_run()
    test_compare_manual_traces()

    # All the assertions passed
    print("tests/test_diff.py: all assertions passed")


if __name__ == "__main__":
    main()
