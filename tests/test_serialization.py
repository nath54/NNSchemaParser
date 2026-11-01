"""
Round-trip tests of the JSON (de)serialization of the schemas and traces.
"""

# Import Modules
import json
import os
import tempfile

import torch
from torch import nn

from nnschemaparser import (
    NNParser,
    SchemaInterpreter,
    ActivationRecorder,
    capture_run,
    compare_traces,
    dict_to_schema,
    dict_to_trace,
    diff_to_dict,
    dump_diff_json,
    dump_json,
    load_diff_json,
    load_json,
    load_trace_json,
    schema_to_dict,
    dump_trace_json,
    trace_to_dict
)


class MLP(nn.Module):
    """Small model with two sub-blocks, for the round-trip checks."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(x)
        activated = hidden.relu()
        return self.fc2(activated)


class Control(nn.Module):
    """Model with control flow, containers and a method, for the nesting."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
        self.gate = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        total = x
        for index, block in enumerate(self.blocks):
            if index == 0:
                total = block(total)
            else:
                total = total + block(total)
        return self.predict(features=total)

    def predict(self, features: torch.Tensor) -> torch.Tensor:
        return self.gate(features.relu())


def _parse(model: nn.Module, shape: tuple[int, ...]) -> object:
    """
    Parse a model with one float32 input of the given shape.

    Args:
        model (nn.Module): The model to parse.
        shape (tuple[int, ...]): The shape of the x input.

    Returns:
        ModelBlockSchema: The parsed schema.
    """

    return NNParser.parse_model(
        model=model,
        inputs_tensors_shapes={"x": shape},
        inputs_tensors_dtypes={"x": "torch.float32"}
    )


def test_round_trip_mlp() -> None:
    """The dumped and reloaded MLP schema is structurally identical."""

    # The parsed schema, dumped to a temporary JSON file
    model: MLP = MLP().eval()
    schema = _parse(model=model, shape=(2, 784))
    with tempfile.TemporaryDirectory() as directory:
        path: str = os.path.join(directory, "schema.json")
        dump_json(block=schema, path=path)

        # The reloaded schema, viewer only by default
        reloaded = load_json(path=path)
        assert reloaded.module is None
        assert list(reloaded.inputs) == ["x"]
        assert reloaded.inputs["x"].shape == (2, 784)
        assert set(reloaded.sub_blocks) == {"fc1", "fc2"}

        # The instruction types and variable names are kept
        types: list[str] = [
            type(instruction).__name__ for instruction in reloaded.body
        ]
        assert types == [type(i).__name__ for i in schema.body]
        names: list[str] = [
            instruction.out_var for instruction in reloaded.body
        ]
        assert names == [i.out_var for i in schema.body]

        # The sub-block call sites and the resolved shapes are kept
        assert reloaded.sub_block_call_sites() == schema.sub_block_call_sites()
        reloaded_hidden = reloaded.body[0].resolved
        assert reloaded_hidden is not None
        assert reloaded_hidden.shape == (2, 256)

        # The reloaded schema runs like the model when the module is back
        runnable = load_json(path=path, module=model)
        x: torch.Tensor = torch.randn(2, 784)
        outputs = SchemaInterpreter().run(block=runnable, scope={"x": x.clone()})
        assert torch.equal(outputs[0], model(x))

        # The payload is valid JSON readable by anything
        with open(path, "r", encoding="utf-8") as handle:
            payload: dict = json.load(handle)
        assert payload["format_version"] == 1
        assert payload["sub_block_call_sites"] == {"fc1": [0], "fc2": [3]}


def test_round_trip_control_flow() -> None:
    """The control flow, containers, methods and call sites round-trip."""

    # The parsed schema of the control flow model
    model: Control = Control().eval()
    schema = _parse(model=model, shape=(2, 8))
    payload: dict = schema_to_dict(block=schema)

    # The loop, the branch and the method call are all in the payload
    body_types: list[str] = [
        instruction["type"] for instruction in payload["body"]
    ]
    assert "ControlInstruction" in body_types
    assert payload["methods"], "the predict method must be exported"
    assert payload["method_call_sites"], "the method call site must be exported"

    # The dependencies are keyed by the nested instruction paths
    paths: set[str] = set()
    for instruction in payload["body"]:
        if "dependencies" in instruction:
            paths.add(instruction["path"])
    assert "0" in paths, "the top level instructions carry their dependencies"

    # The decoded schema re-runs with the module re-attached
    reloaded = dict_to_schema(payload=payload)
    runnable = reloaded
    _reattach_for_test(block=runnable, module=model)
    x: torch.Tensor = torch.randn(2, 8)
    outputs = SchemaInterpreter().run(block=runnable, scope={"x": x.clone()})
    assert torch.equal(outputs[0], model(x))


def _reattach_for_test(block: object, module: nn.Module) -> None:
    """
    Re-attach the module tree of a model to a decoded schema.

    Args:
        block (object): The decoded root block schema.
        module (nn.Module): The live model.
    """

    # The public loader does the same, exposed here for the dict entry point
    from nnschemaparser.serialization import _reattach_module
    _reattach_module(block=block, module=module)


def test_export_toggles() -> None:
    """The toggles disable the optional sections of the payload."""

    # The minimal payload, indices and dependencies excluded
    model: MLP = MLP().eval()
    schema = _parse(model=model, shape=(2, 784))
    minimal: dict = schema_to_dict(
        block=schema,
        include_indices=False,
        include_dependencies=False,
        include_call_sites=False,
        include_resolved=False
    )
    first: dict = minimal["body"][0]
    assert "index" not in first and "path" not in first
    assert "dependencies" not in first and "resolved" not in first
    assert "sub_block_call_sites" not in minimal

    # The indices and the dependencies are opt in again
    full: dict = schema_to_dict(block=schema)
    assert "index" in full["body"][0] and "path" in full["body"][0]
    assert "resolved" in full["body"][0]


def test_round_trip_values() -> None:
    """The include_values toggle round-trips the tensor values."""

    # A capture run filling the resolved values, exported with them
    model: MLP = MLP().eval()
    schema = _parse(model=model, shape=(2, 784))
    x: torch.Tensor = torch.randn(2, 784)
    record = capture_run(block=schema, scope={"x": x.clone()})

    # The schema payload with the values, decoded on the other side
    payload: dict = schema_to_dict(block=schema, include_values=True)
    reloaded = dict_to_schema(payload=payload)
    reloaded_hidden = reloaded.body[0].resolved
    original_hidden = schema.body[0].resolved
    assert reloaded_hidden is not None and original_hidden is not None
    assert torch.equal(reloaded_hidden.value, original_hidden.value)

    # Without the values, the resolved representations stay valueless
    minimal_payload: dict = schema_to_dict(block=schema)
    minimal = dict_to_schema(payload=minimal_payload)
    assert minimal.body[0].resolved is not None
    assert not hasattr(minimal.body[0].resolved, "value")
    assert record.outputs is not None


def test_round_trip_trace() -> None:
    """The traces round-trip with their container ids and values."""

    # A capture run with its activation trace
    model: MLP = MLP().eval()
    schema = _parse(model=model, shape=(2, 784))
    recorder = ActivationRecorder()
    x: torch.Tensor = torch.randn(2, 784)
    outputs = SchemaInterpreter(capture_values=True, listeners=[recorder]).run(
        block=schema,
        scope={"x": x.clone()}
    )
    assert outputs is not None

    # The container ids of the trace, exported and reloaded
    with tempfile.TemporaryDirectory() as directory:
        path: str = os.path.join(directory, "trace.json")
        dump_trace_json(trace=recorder.trace, path=path, include_values=True)
        reloaded = load_trace_json(path=path)

        # The steps match one by one, values included
        assert len(reloaded) == len(recorder.trace)
        reloaded_step = reloaded.steps[0]
        assert reloaded_step.container_id == recorder.trace.steps[0].container_id
        assert reloaded_step.slot_type == "input"
        assert torch.equal(
            reloaded_step.tensor.value,
            recorder.trace.steps[0].tensor.value
        )

        # The payload without the values decodes with placeholder tensors
        payload: dict = trace_to_dict(trace=recorder.trace)
        assert "value" not in payload["steps"][0]["tensor"]
        assert payload["steps"][0]["container_id"] == "input_0001"


def test_round_trip_gradients() -> None:
    """The gradient traces round-trip with their source container ids."""

    # A capture run with gradients, backward included
    model: MLP = MLP().eval()
    schema = _parse(model=model, shape=(2, 784))
    x: torch.Tensor = torch.randn(2, 784)
    record = capture_run(block=schema, scope={"x": x.clone()}, capture_grads=True)
    loss = record.outputs[0].pow(2).sum()
    loss.backward()

    # The gradient trace exported and reloaded
    gradients = record.gradients
    payload: dict = trace_to_dict(trace=gradients)
    reloaded = dict_to_trace(payload=payload)
    assert len(reloaded) == len(gradients)
    assert reloaded.steps[0].source_container_id \
        == gradients.steps[0].source_container_id


def test_schema_trace_linkage() -> None:
    """The trace container ids attach to the instructions of the schema."""

    # A capture run whose trace is attached to the schema payload
    model: MLP = MLP().eval()
    schema = _parse(model=model, shape=(2, 784))
    recorder = ActivationRecorder()
    x: torch.Tensor = torch.randn(2, 784)
    outputs = SchemaInterpreter(capture_values=True, listeners=[recorder]).run(
        block=schema,
        scope={"x": x.clone()}
    )
    assert outputs is not None

    # The payload with the attached containers, one self contained file
    payload: dict = schema_to_dict(block=schema, trace=recorder.trace)

    # The input variable carries the input container id, the hidden
    # variable the activation container id of the sub-block call
    first: dict = payload["body"][0]
    assert first["out_var"] == "hidden"
    attached: list[dict] = first["containers"]
    assert [entry["container_id"] for entry in attached] == ["MLP_0003"]
    assert attached[0]["slot_type"] == "activation"

    # The fc2 call instruction carries the output container of the block
    last_call: dict = payload["body"][-2]
    assert last_call["out_var"] == "$t2"
    kinds: list[str] = [entry["slot_type"] for entry in last_call["containers"]]
    assert kinds == ["activation", "output"]

    # The sub-block instructions carry the containers of their blocks
    fc1: dict = payload["sub_blocks"]["fc1"]
    fc1_call: dict = fc1["body"][0]
    assert fc1_call["out_var"] == "$t1"
    assert fc1_call["containers"][0]["container_id"] == "fc1_0002"

    # Without a trace, no containers are attached
    plain: dict = schema_to_dict(block=schema)
    assert "containers" not in plain["body"][0]

    # The activations and the gradients attach together, in order
    record = capture_run(block=schema, scope={"x": x.clone()}, capture_grads=True)
    loss = record.outputs[0].pow(2).sum()
    loss.backward()
    both: dict = schema_to_dict(
        block=schema,
        trace=[record.trace, record.gradients]
    )
    linked: list[dict] = both["body"][0]["containers"]
    slot_kinds: list[str] = [entry["slot_type"] for entry in linked]
    assert slot_kinds == ["activation", "gradient"]
    assert linked[1]["container_id"] == "grad_MLP_0003"


def test_diff_export() -> None:
    """The trace diffs of two runs export and reload as JSON."""

    # A base run and a forked run with an edited activation
    model: MLP = MLP().eval()
    schema = _parse(model=model, shape=(2, 784))
    base = capture_run(block=schema, scope={"x": torch.randn(2, 784)})
    fork_scope: dict = {"x": base.scope["x"].clone()}
    fork = capture_run(block=schema, scope=fork_scope)

    # The comparison of the two runs, exported and reloaded
    diffs = compare_traces(trace_a=base.trace, trace_b=fork.trace, atol=1e-6)
    with tempfile.TemporaryDirectory() as directory:
        path: str = os.path.join(directory, "diff.json")
        dump_diff_json(diffs=diffs, path=path)

        # The reloaded diffs match the originals entry by entry
        reloaded = load_diff_json(path=path)
        assert len(reloaded) == len(diffs)
        assert reloaded[0].status == diffs[0].status
        assert reloaded[0].container_id_a == diffs[0].container_id_a
        assert reloaded[0].container_id_b == diffs[0].container_id_b

        # The payload carries the per status summary of the comparison
        with open(path, "r", encoding="utf-8") as handle:
            payload: dict = json.load(handle)
        assert payload["type"] == "trace_diff"
        assert payload["format_version"] == 1
        assert payload["summary"].get("ok", 0) > 0

        # The dict round-trip without a file, values unavailable without
        # captured values on both sides
        from_dict = diff_to_dict(diffs=diffs)
        assert from_dict["summary"] == payload["summary"]
