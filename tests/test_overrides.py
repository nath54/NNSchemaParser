"""
Tests of the mid-run value overrides: the Override targets (instruction
paths, out_var / op_name descriptors, weights), the occurrence filters, the
from_container_id factory and the trace consistency of the overridden runs.

Run with: python tests/test_overrides.py
"""

# Import Modules
import torch
from torch import Tensor, nn

from nnschemaparser import (
    NNParser,
    Override,
    SchemaInterpreter,
    capture_run,
    compare_traces
)


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
            hidden (int): The hidden size of the model.
            experts (int): The number of experts of the container.
        """

        # Init parent
        super().__init__()

        # Create the experts and the output projection
        self.experts: nn.ModuleList = nn.ModuleList(
            [Expert(hidden) for _ in range(experts)]
        )
        self.out: nn.Linear = nn.Linear(hidden, hidden)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward function.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """

        # The routed sum of the expert outputs
        acc: Tensor = x
        for expert in self.experts:
            acc = acc + expert(acc)

        # The output projection
        return self.out(acc)


class Gated(nn.Module):
    """
    Two branches model whose selection is steerable by a condition.
    """

    def __init__(self) -> None:
        """
        Init function.
        """

        # Init parent
        super().__init__()

        # The gate flag and the two branches
        self.gate: Tensor = nn.Parameter(torch.tensor(0.0))
        self.a: nn.Linear = nn.Linear(4, 4)
        self.b: nn.Linear = nn.Linear(4, 4)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward function.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The output tensor.
        """

        # The branch selected by the gate
        if self.gate > 0.0:
            return self.a(x)
        return self.b(x)


def build_schema(model: nn.Module, shape: tuple[int, ...]):
    """
    Parse a model into a schema with a single tensor input.

    Args:
        model (nn.Module): The model to parse.
        shape (tuple[int, ...]): The shape of the input tensor.

    Returns:
        ModelBlockSchema: The parsed schema of the model.
    """

    # The parse of the model schema
    return NNParser.parse_model(
        model=model,
        inputs_tensors_shapes={"x": shape},
        inputs_tensors_dtypes={"x": "torch.float32"}
    )


def test_value_override_on_instruction() -> None:
    """
    The values replacement of a precise instruction binding, recorded in
    the trace and propagated to the outputs.
    """

    # The test model and its reference run
    model: MoE = MoE(hidden=4, experts=2).eval()
    schema = build_schema(model=model, shape=(2, 4))
    x: Tensor = torch.randn(2, 4)
    reference: Tensor = model(x)
    record = capture_run(block=schema, scope={"x": x.clone()})
    assert torch.allclose(record.outputs[0], reference, atol=1e-5)

    # The zeroing of the expert result bindings, at the top level of the
    # loop body of the root block
    override: Override = Override(
        block_path="",
        instruction="1/body/0",
        values=0.0
    )
    overridden = capture_run(
        block=schema,
        scope={"x": x.clone()},
        overrides=[override]
    )

    # The outputs differ and the trace carries the overridden value
    assert not torch.allclose(record.outputs[0], overridden.outputs[0])
    binding_steps: list = [
        step
        for step in overridden.trace.steps
        if step.block_path == "" and step.out_var == "$t1"
    ]
    assert len(binding_steps) == 2
    assert all(
        float(step.tensor.value.abs().max()) == 0.0
        for step in binding_steps
    )

    # The internal computation of the expert is unchanged
    expert_steps: list = [
        step
        for step in overridden.trace.steps
        if step.block_path == "experts.0"
    ]
    assert all(
        float(step.tensor.value.abs().max()) > 0.0
        for step in expert_steps
    )


def test_occurrence_filter_in_loops() -> None:
    """
    The occurrence filter targets a single iteration of a loop body.
    """

    # The test model and its reference run
    model: MoE = MoE(hidden=4, experts=3).eval()
    schema = build_schema(model=model, shape=(2, 4))
    x: Tensor = torch.randn(2, 4)
    record = capture_run(block=schema, scope={"x": x.clone()})

    # The zeroing of the second iteration only
    override: Override = Override(
        block_path="",
        instruction="1/body/0",
        occurrence=2,
        values=0.0
    )
    overridden = capture_run(
        block=schema,
        scope={"x": x.clone()},
        overrides=[override]
    )

    # The first expert binding is unchanged, the second one is zeroed
    loop_bindings: list = [
        step
        for step in overridden.trace.steps
        if step.block_path == "" and step.out_var == "$t1"
    ]
    assert len(loop_bindings) == 3
    assert float(loop_bindings[0].tensor.value.abs().max()) > 0.0
    assert float(loop_bindings[1].tensor.value.abs().max()) == 0.0
    assert float(loop_bindings[2].tensor.value.abs().max()) > 0.0

    # The diff against the reference run starts at the second binding
    diffs: list = compare_traces(
        trace_a=record.trace,
        trace_b=overridden.trace
    )
    changed: list = [diff for diff in diffs if diff.changed]
    assert len(changed) >= 2


def test_descriptor_override() -> None:
    """
    The out_var / op_name descriptor targets the instructions binding a
    variable, with the occurrence disambiguation.
    """

    # The test model and its reference run
    model: MoE = MoE(hidden=4, experts=2).eval()
    schema = build_schema(model=model, shape=(2, 4))
    x: Tensor = torch.randn(2, 4)
    record = capture_run(block=schema, scope={"x": x.clone()})

    # The zeroing of the second binding of the $t1 variable by the
    # container call of the experts
    override: Override = Override(
        block_path="",
        out_var="$t1",
        op_name="experts",
        occurrence=2,
        values=0.0
    )
    overridden = capture_run(
        block=schema,
        scope={"x": x.clone()},
        overrides=[override]
    )

    # Only the second binding of acc is zeroed
    loop_bindings: list = [
        step
        for step in overridden.trace.steps
        if step.block_path == "" and step.out_var == "$t1"
    ]
    assert float(loop_bindings[0].tensor.value.abs().max()) > 0.0
    assert float(loop_bindings[1].tensor.value.abs().max()) == 0.0
    assert len(record.trace.steps) == len(overridden.trace.steps)


def test_patch_callable_override() -> None:
    """
    The patch callable produces the replacement from the produced value.
    """

    # The test model and its schema
    model: MoE = MoE(hidden=4, experts=2).eval()
    schema = build_schema(model=model, shape=(2, 4))
    x: Tensor = torch.randn(2, 4)

    # The nan injection at the first expert binding, cleaned by the patch
    # of the sum bindings that consume it
    inject: Override = Override(
        block_path="",
        instruction="1/body/0",
        occurrence=1,
        patch=lambda t: t * float("nan")
    )
    clean: Override = Override(
        block_path="",
        out_var="$t2",
        patch=lambda t: torch.nan_to_num(t)
    )
    record = capture_run(
        block=schema,
        scope={"x": x.clone()},
        overrides=[inject, clean]
    )

    # The outputs hold no nan: every sum binding was cleaned
    assert not torch.isnan(record.outputs[0]).any()
    assert not torch.isinf(record.outputs[0]).any()


def test_condition_override_steers_control_flow() -> None:
    """
    The override of a condition instruction steers the branch taken.
    """

    # The test model with a closed gate and its schema
    model: Gated = Gated().eval()
    with torch.no_grad():
        model.gate.fill_(-1.0)
    schema = build_schema(model=model, shape=(2, 4))
    x: Tensor = torch.randn(2, 4)

    # The reference run takes the else branch
    record = capture_run(block=schema, scope={"x": x.clone()})
    assert torch.allclose(record.outputs[0], model.b(x), atol=1e-5)

    # The forcing of the condition value opens the gate
    override: Override = Override(
        block_path="",
        instruction="0",
        values=torch.tensor(True)
    )
    overridden = capture_run(
        block=schema,
        scope={"x": x.clone()},
        overrides=[override]
    )
    assert torch.allclose(overridden.outputs[0], model.a(x), atol=1e-5)


def test_weight_override() -> None:
    """
    The weight overrides replace the scope seeded weights without mutating
    the schema nor the source module.
    """

    # The test model and its schema
    model: MoE = MoE(hidden=4, experts=1).eval()
    schema = build_schema(model=model, shape=(2, 4))
    x: Tensor = torch.randn(2, 4)
    weight_before: Tensor = schema.sub_blocks["out"].weights["weight"].value.clone()
    module_before: Tensor = model.out.weight.detach().clone()

    # The zeroing of the output weight and bias of the run
    overrides: list[Override] = [
        Override(block_path="out", weight="weight", values=torch.zeros(4, 4)),
        Override(block_path="out", weight="bias", values=torch.zeros(4))
    ]
    record = capture_run(
        block=schema,
        scope={"x": x.clone()},
        overrides=overrides
    )

    # The outputs are zeroed and the schema and module are untouched
    assert torch.equal(record.outputs[0], torch.zeros(2, 4))
    assert torch.equal(
        schema.sub_blocks["out"].weights["weight"].value,
        weight_before
    )
    assert torch.equal(model.out.weight.detach(), module_before)

    # A run without the overrides reproduces the real forward
    clean = capture_run(block=schema, scope={"x": x.clone()})
    assert torch.allclose(clean.outputs[0], model(x), atol=1e-5)


def test_from_container_id_round_trip() -> None:
    """
    The factory resolves an override from a container id of a stored run,
    and the re-run with the override diffs against the reference.
    """

    # The test model, its schema and its reference run
    model: MoE = MoE(hidden=4, experts=2).eval()
    schema = build_schema(model=model, shape=(2, 4))
    x: Tensor = torch.randn(2, 4)
    record = capture_run(block=schema, scope={"x": x.clone()})

    # The targeted container: the first binding of acc by the container
    # call of the experts
    target = next(
        step
        for step in record.trace.steps
        if step.block_path == "" and step.out_var == "$t1"
    )

    # The factory resolves the descriptor of the producing instruction
    override: Override = Override.from_container_id(
        trace=record.trace,
        container_id=target.container_id,
        values=torch.full_like(target.tensor.value, 7.0)
    )
    assert override.block_path == ""
    assert override.out_var == "$t1"
    assert override.op_name == "experts"
    assert override.occurrence == 1

    # The re-run with the resolved override fills the first binding
    overridden = capture_run(
        block=schema,
        scope={"x": x.clone()},
        overrides=[override]
    )
    loop_bindings: list = [
        step
        for step in overridden.trace.steps
        if step.block_path == "" and step.out_var == "$t1"
    ]
    assert float(loop_bindings[0].tensor.value.abs().max()) == 7.0

    # The diff of the two runs shows the divergence
    diffs: list = compare_traces(
        trace_a=record.trace,
        trace_b=overridden.trace
    )
    assert any(diff.changed for diff in diffs)


def test_overrides_with_step_run() -> None:
    """
    The overrides apply identically in the stepped execution mode.
    """

    # The test model and its schema
    model: MoE = MoE(hidden=4, experts=2).eval()
    schema = build_schema(model=model, shape=(2, 4))
    x: Tensor = torch.randn(2, 4)

    # The stepped run with the zeroing of the expert result bindings
    override: Override = Override(
        block_path="",
        instruction="1/body/0",
        values=0.0
    )
    interpreter: SchemaInterpreter = SchemaInterpreter(
        capture_values=True,
        overrides=[override]
    )
    steps: list = list(interpreter.step_run(
        block=schema,
        scope={"x": x.clone()}
    ))

    # The root scope carries the overridden values of the loop body
    expert_bindings: list = [
        step
        for step in steps
        if step.block is schema and step.out_var == "$t1"
    ]
    assert len(expert_bindings) == 2
    assert all(
        float(step.value.abs().max()) == 0.0
        for step in expert_bindings
    )


def test_override_errors() -> None:
    """
    The invalid targets and replacements are rejected with clear errors.
    """

    # The test model and its schema
    model: MoE = MoE(hidden=4, experts=1).eval()
    schema = build_schema(model=model, shape=(2, 4))
    x: Tensor = torch.randn(2, 4)

    # The unknown block path
    try:
        capture_run(
            block=schema,
            scope={"x": x.clone()},
            overrides=[
                Override(block_path="nope", instruction="0", values=0.0)
            ]
        )
        raise AssertionError("The unknown block path should raise.")
    except ValueError as error:
        assert "nope" in str(error)

    # The unknown instruction path
    try:
        capture_run(
            block=schema,
            scope={"x": x.clone()},
            overrides=[
                Override(block_path="", instruction="99", values=0.0)
            ]
        )
        raise AssertionError("The unknown instruction path should raise.")
    except ValueError as error:
        assert "99" in str(error)

    # The instruction binding no variable
    try:
        capture_run(
            block=schema,
            scope={"x": x.clone()},
            overrides=[
                Override(block_path="", instruction="1", values=0.0)
            ]
        )
        raise AssertionError("The variable less instruction should raise.")
    except ValueError as error:
        assert "binds no variable" in str(error)

    # The unknown weight name
    try:
        capture_run(
            block=schema,
            scope={"x": x.clone()},
            overrides=[
                Override(block_path="out", weight="nope", values=0.0)
            ]
        )
        raise AssertionError("The unknown weight should raise.")
    except ValueError as error:
        assert "nope" in str(error)

    # The element count mismatch without an explicit shape
    try:
        capture_run(
            block=schema,
            scope={"x": x.clone()},
            overrides=[
                Override(block_path="", instruction="1/body/1", values=[1.0, 2.0])
            ]
        )
        raise AssertionError("The shape mismatch should raise.")
    except ValueError as error:
        assert "Cannot fit" in str(error)

    # The construction errors of the overrides
    for kwargs in (
        {"block_path": "", "values": 0.0, "patch": lambda t: t},
        {"block_path": ""},
        {"block_path": "", "instruction": "0", "weight": "w", "values": 0.0},
        {"block_path": "", "op_name": "", "values": 0.0}
    ):
        try:
            Override(**kwargs)
            raise AssertionError(f"The kwargs {kwargs} should raise.")
        except ValueError:
            pass


def main() -> None:
    """
    Run the override tests.
    """

    # Every test of the file, in order
    test_value_override_on_instruction()
    print("test_value_override_on_instruction OK")

    test_occurrence_filter_in_loops()
    print("test_occurrence_filter_in_loops OK")

    test_descriptor_override()
    print("test_descriptor_override OK")

    test_patch_callable_override()
    print("test_patch_callable_override OK")

    test_condition_override_steers_control_flow()
    print("test_condition_override_steers_control_flow OK")

    test_weight_override()
    print("test_weight_override OK")

    test_from_container_id_round_trip()
    print("test_from_container_id_round_trip OK")

    test_overrides_with_step_run()
    print("test_overrides_with_step_run OK")

    test_override_errors()
    print("test_override_errors OK")

    # The summary of the suite
    print("\nAll override tests passed.")


if __name__ == "__main__":
    main()
