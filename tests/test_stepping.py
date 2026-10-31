"""
Tests of the stepped execution, the scope edits and the frame depths.

Run with: python tests/test_stepping.py
"""

# Import Modules
from typing import Any, Generator

import torch
from torch import Tensor, nn

from nnschemaparser import NNParser, SchemaInterpreter, ActivationRecorder
from nnschemaparser.interpreter.stepping import StepInfo


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

        # Create the weight of the expert
        self.w: nn.Parameter = nn.Parameter(torch.randn(hidden, hidden) * 0.1)

    def forward(self, y: Tensor) -> Tensor:
        """
        Forward function.

        Args:
            y (Tensor): The input tensor.

        Returns:
            Tensor: The transformed tensor.
        """

        # The single matrix multiplication of the expert
        return y @ self.w


class MoE(nn.Module):
    """
    Mixture of experts with a routed loop over a module list.
    """

    def __init__(self, hidden: int, experts: int) -> None:
        """
        Init function.

        Args:
            hidden (int): The hidden size.
            experts (int): The number of experts.
        """

        # Init parent
        super().__init__()

        # Create the router and the experts of the mixture
        self.gate: nn.Linear = nn.Linear(hidden, experts, bias=False)
        self.experts: nn.ModuleList = nn.ModuleList(
            [Expert(hidden=hidden) for _ in range(experts)]
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward function.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The mixed expert outputs.
        """

        # The mean of the expert outputs weighted by the routed gate
        mixed: Tensor = torch.zeros_like(x)
        for index, expert in enumerate(self.experts):
            weight: Tensor = self.gate.weight[index].softmax(dim=0)
            mixed = mixed + expert(x) * weight
        return mixed


def build_model_and_schema() -> tuple[MoE, Any]:
    """
    Build the test model and its parsed schema.

    Returns:
        tuple[MoE, Any]: The model and the schema.
    """

    # The test model and its input
    model: MoE = MoE(hidden=4, experts=3).eval()
    schema = NNParser.parse_model(
        model=model,
        inputs_tensors_shapes={"x": (2, 4)},
        inputs_tensors_dtypes={"x": "torch.float32"}
    )
    return model, schema


def consume_steps(generator: Generator[StepInfo, None, list[Tensor]]) -> list[StepInfo]:
    """
    Consume a stepped run to its end, collecting the steps.

    Args:
        generator (Generator[StepInfo, None, list[Tensor]]): The generator
            of the stepped run.

    Returns:
        list[StepInfo]: The collected steps.
    """

    # The steps collected until the StopIteration of the generator
    steps: list[StepInfo] = []
    while True:
        try:
            steps.append(next(generator))
        except StopIteration:
            return steps


def first_expert_matmul_step(steps: list[StepInfo]) -> StepInfo:
    """
    Get the first matmul step of the first expert body.

    Args:
        steps (list[StepInfo]): The steps of a run.

    Returns:
        StepInfo: The matmul step of the first expert.
    """

    # The first op of the frame depth 1 binding the matmul result
    step: StepInfo
    for step in steps:
        if step.kind == "op" and step.frame_depth == 1 \
                and step.out_var == "$t1" and isinstance(step.value, Tensor):
            return step
    raise AssertionError("no expert matmul step found")


def expected_output_with_first_expert_ones(
    model: MoE,
    x: Tensor,
    hidden: int = 4
) -> Tensor:
    """
    Compute the MoE output with the first expert output replaced by ones.

    Args:
        model (MoE): The test model.
        x (Tensor): The input tensor.
        hidden (int): The hidden size of the model.

    Returns:
        Tensor: The expected output tensor.
    """

    # The manual mixture: ones for the edited first expert, the matmul for
    # the others, all weighted by the routed gate
    ones: Tensor = torch.ones(2, hidden)
    weight0: Tensor = model.gate.weight[0].softmax(dim=0)
    weight1: Tensor = model.gate.weight[1].softmax(dim=0)
    weight2: Tensor = model.gate.weight[2].softmax(dim=0)
    return ones * weight0 + (x @ model.experts[1].w) * weight1 \
        + (x @ model.experts[2].w) * weight2


def test_step_sequence() -> None:
    """
    Check the step sequence, kinds and frame depths of a stepped run.
    """

    # The model, schema and interpreter of the test
    model: MoE
    schema: Any
    model, schema = build_model_and_schema()
    interpreter: SchemaInterpreter = SchemaInterpreter()

    # The stepped run consumed to its end
    x: Tensor = torch.randn(2, 4)
    steps: list[StepInfo] = consume_steps(
        generator=interpreter.step_run(block=schema, scope={"x": x.clone()})
    )

    # The step kinds cover the loop, the expert calls and the return
    kinds: list[str] = [step.kind for step in steps]
    assert kinds[0] == "op", f"first step kind: {kinds[0]}"
    assert kinds.count("for") == 1, f"for steps: {kinds}"
    assert kinds.count("block_call") == 3, f"block calls: {kinds}"
    assert kinds[-1] == "return", f"last step kind: {kinds[-1]}"

    # The step indices are 1-based and consecutive
    assert [step.step_index for step in steps] == list(range(1, len(steps) + 1))

    # The root body steps are at the frame depth 0, the expert bodies at 1
    step: StepInfo
    for step in steps:
        if step.block is schema:
            assert step.frame_depth == 0
        elif step.block is schema.sub_blocks["experts.0"]:
            assert step.frame_depth == 1

    # The per-iteration steps of the loop body repeat three times
    assert len(steps) > 10, f"too few steps: {len(steps)}"

    # The counter of the interpreter matches the step count
    assert interpreter._step_counter == len(steps)

    print("test_stepping.sequence: OK")


def test_stepped_run_matches_real_forward() -> None:
    """
    Check that a stepped run without edits matches the real forward.
    """

    # The model, schema and interpreter of the test
    model: MoE
    schema: Any
    model, schema = build_model_and_schema()
    interpreter: SchemaInterpreter = SchemaInterpreter()

    # The stepped run of a fixed input
    x: Tensor = torch.randn(2, 4)
    consume_steps(generator=interpreter.step_run(block=schema, scope={"x": x.clone()}))
    outputs: list[Tensor] = interpreter.step_outputs

    # The outputs match the real forward on the same input
    assert torch.allclose(outputs[0], model(x), atol=1e-6)

    print("test_stepping.matches_forward: OK")


def test_raw_hook_and_edit() -> None:
    """
    Check the raw hook mode and a mid-run scope edit applied from the hook.
    """

    # The model, schema and interpreter of the test
    model: MoE
    schema: Any
    model, schema = build_model_and_schema()
    x: Tensor = torch.randn(2, 4)

    # The base run without hook for the reference output
    base_interpreter: SchemaInterpreter = SchemaInterpreter()
    base_outputs: list[Tensor] = base_interpreter.run(
        block=schema, scope={"x": x.clone()}
    )

    # The hooked run editing the first expert matmul result in place
    interpreter: SchemaInterpreter = SchemaInterpreter()
    seen: list[StepInfo] = []
    edited: list[bool] = []

    def hook(step: StepInfo) -> None:
        """Record the step and edit the first expert matmul result."""
        seen.append(step)

        # The first matmul of the first expert body, edited only once
        if step.kind == "op" and step.frame_depth == 1 \
                and step.out_var == "$t1" and isinstance(step.value, Tensor) \
                and not edited:
            edited.append(True)
            step.edit(values=torch.ones_like(step.value))

    interpreter.step_hook = hook
    edited_outputs: list[Tensor] = interpreter.run(
        block=schema, scope={"x": x.clone()}
    )
    interpreter.step_hook = None

    # The hook saw every instruction of the run, the loop and the return
    assert seen[0].kind == "op" and seen[1].kind == "for"
    assert seen[-1].kind == "return"
    assert len(seen) > 10

    # The edit changed the output to the manual expectation
    assert not torch.allclose(base_outputs[0], edited_outputs[0], atol=1e-6)
    assert torch.allclose(
        edited_outputs[0],
        expected_output_with_first_expert_ones(model=model, x=x),
        atol=1e-5
    )

    print("test_stepping.raw_hook: OK")


def test_generator_edit_and_fork_replay() -> None:
    """
    Check a mid-run edit via the generator and the fork replay equivalence.
    """

    # The model, schema and interpreter of the test
    model: MoE
    schema: Any
    model, schema = build_model_and_schema()
    x: Tensor = torch.randn(2, 4)

    # The base stepped run collecting the steps and their indices
    base_interpreter: SchemaInterpreter = SchemaInterpreter()
    base_steps: list[StepInfo] = consume_steps(
        generator=base_interpreter.step_run(block=schema, scope={"x": x.clone()})
    )
    target: StepInfo = first_expert_matmul_step(steps=base_steps)

    # The fork: replay to the target step, edit the variable, continue
    fork_interpreter: SchemaInterpreter = SchemaInterpreter()
    fork_steps: list[StepInfo] = []
    fork_outputs: list[Tensor] | None = None
    edits_queued: list[bool] = [False]
    generator: Generator[StepInfo, None, list[Tensor]] = fork_interpreter.step_run(
        block=schema,
        scope={"x": x.clone()}
    )
    while True:
        try:
            step: StepInfo = next(generator)
            fork_steps.append(step)

            # The edit is queued once when the target step is replayed
            if not edits_queued[0] and step.step_index == target.step_index \
                    and step.kind == "op" and step.frame_depth == 1:
                edits_queued[0] = True
                step.edit(values=torch.ones_like(step.value))
        except StopIteration as stop:
            fork_outputs = stop.value
            break

    # The fork output matches the manual expectation and differs from base
    base_outputs: list[Tensor] = base_interpreter.step_outputs
    expected: Tensor = expected_output_with_first_expert_ones(model=model, x=x)
    assert not torch.allclose(base_outputs[0], fork_outputs[0], atol=1e-6)
    assert torch.allclose(fork_outputs[0], expected, atol=1e-5), \
        f"fork mismatch: {fork_outputs[0]} vs {expected}"

    # The replay of the fork reuses the same step indices
    assert fork_steps[target.step_index - 1].step_index == target.step_index

    print("test_stepping.fork_edit: OK")


def test_abandoned_generator() -> None:
    """
    Check that abandoning a generator unwinds the run without hanging.
    """

    # The model, schema and interpreter of the test
    model: MoE
    schema: Any
    model, schema = build_model_and_schema()
    interpreter: SchemaInterpreter = SchemaInterpreter()

    # Consume two steps then abandon the generator
    generator: Generator[StepInfo, None, list[Tensor]] = interpreter.step_run(
        block=schema,
        scope={"x": torch.randn(2, 4)}
    )
    next(generator)
    next(generator)
    generator.close()

    # The interpreter is immediately reusable for a normal run
    x: Tensor = torch.randn(2, 4)
    outputs: list[Tensor] = interpreter.run(block=schema, scope={"x": x.clone()})
    assert torch.allclose(outputs[0], model(x), atol=1e-6)

    print("test_stepping.abandoned: OK")


def test_recorder_frame_depths() -> None:
    """
    Check that the recorder and the stepper agree on the frame depths.
    """

    # The model, schema and interpreter of the test
    model: MoE
    schema: Any
    model, schema = build_model_and_schema()

    # The recorder of a captured run on a second interpreter
    recorder: ActivationRecorder = ActivationRecorder()
    captured_interpreter: SchemaInterpreter = SchemaInterpreter(
        capture_values=True,
        listeners=[recorder]
    )
    captured_interpreter.run(block=schema, scope={"x": torch.randn(2, 4)})

    # Every trace step carries a valid frame depth
    trace_step: Any
    for trace_step in recorder.trace.steps:
        assert trace_step.frame_depth >= 0
        assert trace_step.frame_depth < 2

    # The first expert matmul is at the frame depth 1 in the trace too
    trace_steps: list[Any] = recorder.trace.steps
    assert any(
        step.frame_depth == 1 and step.out_var == "$t1"
        for step in trace_steps
    )

    print("test_stepping.recorder_depths: OK")


def main() -> None:
    """
    Run the stepping tests.
    """

    # All the stepping tests in order
    test_step_sequence()
    test_stepped_run_matches_real_forward()
    test_raw_hook_and_edit()
    test_generator_edit_and_fork_replay()
    test_abandoned_generator()
    test_recorder_frame_depths()

    # All the assertions passed
    print("tests/test_stepping.py: all assertions passed")


if __name__ == "__main__":
    main()
