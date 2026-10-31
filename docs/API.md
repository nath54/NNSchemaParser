# Public API

The stable surface of the library is everything importable from the package
root. Everything under `nnschemaparser.parser.*`, `nnschemaparser.core.*` and
`nnschemaparser.interpreter.*` is internal and may move between releases.

```python
from nnschemaparser import (
    NNParser,
    SchemaInterpreter,
    TraceListener,
    TraceRecorder,
    ActivationRecorder,
    ExecutionRecord,
    StepInfo,
    capture_run,
    compare_traces,
    dump_schema,
)
```

## Parsing a model

```python
import torch
from nnschemaparser import NNParser

model = MyModel().eval()
schema = NNParser.parse_model(
    model=model,
    inputs_tensors_shapes={"x": (2, 16)},
    inputs_tensors_dtypes={"x": "torch.float32"},
    inputs_tensors_values={"x": torch.randn(2, 16)},
)
```

- `inputs_tensors_shapes` — the shapes of the tensor inputs; when empty, the
  shape resolution pass is skipped entirely.
- `inputs_tensors_values` — optional example values; they type the scalar
  inputs and make the resolution exact.
- The returned `ModelBlockSchema` carries `inputs` (with `input_defaults`),
  `outputs`, `weights`, `sub_blocks`, `body`, `methods`, `method_name`,
  `module` and `skipped` (children that could not be parsed and are
  re-dispatched to the real modules at run time).

## Executing a schema

```python
from nnschemaparser import SchemaInterpreter

outputs = SchemaInterpreter().run(block=schema, scope={"x": x})
```

The `scope` maps the declared input names to their values; the outputs are
the list of the values returned by the forward, bound like python (a single
return value stays unwrapped only inside block calls).

With `SchemaInterpreter(capture_values=True)`, every tensor result is kept
as a `TensorReprWithValue` on the `resolved` field of each instruction, so a
consumer can show the value of every intermediate tensor.

- **`capture_run()` and `ExecutionRecord`** — the one-call capture workflow
  bundling outputs, activation trace, final scope and lazy gradients.
- **`SchemaInterpreter.step_run()` and `StepInfo`** — stepped execution with
  mid-run scope edits (`step.edit(...)`), plus the raw `step_hook` mode.
- **`compare_traces()` / `TraceDiff` / `diff_summary`** — step-by-step
  comparison of two traces (base run vs fork), aligned by container id with
  a role fallback.

## Tracing an execution

Subclass `TraceListener` and override the hooks you need
(`on_block_enter`, `on_block_exit`, `on_op`, `on_block_call`,
`on_control_enter`, `on_control_exit`, `on_try_enter`, `on_try_exit`), then
pass a list of listeners to the interpreter:

```python
from nnschemaparser import SchemaInterpreter, TraceRecorder

tracer = TraceRecorder()
outputs = SchemaInterpreter(listeners=[tracer]).run(block=schema, scope={"x": x})
print(tracer.lines)
```

`TraceRecorder` keeps an indented text trace of the whole execution in its
`lines` attribute.

## Recording activation and gradient traces

The schema is the static structure; the values of one execution live in a
separate `ActivationTrace` of `TensorContainer`s keyed by stable container
ids. The `ActivationRecorder` is a `TraceListener` allocating them:

```python
from nnschemaparser import SchemaInterpreter, ActivationRecorder

recorder = ActivationRecorder(capture_grads=True)
outputs = SchemaInterpreter(capture_values=True, listeners=[recorder]).run(
    block=schema, scope={"x": x.clone().requires_grad_(True)}
)

for step in recorder.trace.steps:
    print(step.step, step.description, step.container_id, step.tensor.shape)

loss = outputs[0].pow(2).sum()
loss.backward()
gradients = recorder.collect_gradients()   # cross-referenced GradientTrace
```

- `ActivationTrace.steps` — the chronological `TraceStep` entries: `step`,
  `container_id` (`"input_0001"`, `"layers_0_0002"`, `"output_0007"`),
  `description`, `slot_type` (`"input"/"activation"/"output"`), `out_var`,
  `op_name`, `block_path` and `tensor` (the `TensorContainer` holding the
  detached value).
- `collect_gradients()` — to call after a backward pass: returns a
  `GradientTrace` whose steps mirror the activation steps in the same order,
  with `container_id = "grad_" + activation id` and `source_container_id`
  pointing back to the activation step.
- Both traces provide `by_container(id)` and `save(path)`.

The schema itself is never modified by a run: `instruction.resolved` keeps
the values of the last capture, while the traces are the per-run history.

## Stepped execution and mid-run edits

```python
from nnschemaparser import SchemaInterpreter

interpreter = SchemaInterpreter(capture_values=True)
for step in interpreter.step_run(block=schema, scope={"x": x}):
    print(step.step_index, step.kind, step.out_var, step.frame_depth)
    if step.out_var == "activated":
        step.edit(values=torch.zeros_like(step.value))   # mid-run edit
outputs = interpreter.step_outputs
```

- `StepInfo` fields: `step_index` (deterministic per schema/input/path),
  `kind` (`"op"`, `"block_call"`, `"if"`, `"for"`, `"while"`, `"try"`,
  `"return"`, `"continue"`, `"break"`), `out_var`, `op_name`, `block`,
  `scope`, `scope_frames` (live stack, root frame at depth 0),
  `frame_depth`, `value`.
- `StepInfo.edit(values, shape=None, dtype=None, frame_depth=None,
  out_var=None)` patches a scope frame before the next instruction; in
  generator mode it must be called between two yields.
- The raw alternative `interpreter.step_hook = callable` receives the same
  `StepInfo` synchronously during a normal `run()`.
- Condition expressions of `if`/`while` re-evaluations are not stepped.
- One stepped run at a time per interpreter; abandoning the generator
  unwinds the run at the next step.

## Comparing two runs

```python
from nnschemaparser import compare_traces, diff_summary

diffs = compare_traces(trace_a=base_record.trace, trace_b=fork_trace, atol=1e-6)
summary = diff_summary(diffs)   # {"ok": 4, "shape_changed": 0, ...}
```

Each `TraceDiff` carries `container_id_a`/`container_id_b`, `status`
(`"ok"`, `"shape_changed"`, `"value_unavailable"`, `"only_in_a"`,
`"only_in_b"`), `matched_by` (`"container_id"` or the `"role"` fallback on
block path, variable name and operation name), and `max_abs_diff`,
`mean_abs_diff`, `changed` when the captured values are comparable.

## Canvas positioning: call sites and dependencies

```python
from nnschemaparser import instruction_dependencies, method_call_sites

# Where each sub-block / parsed method is called in the top level body
schema.sub_block_call_sites()   # {"fc1": [0], "experts": [3], ...}
method_call_sites(block=schema) # {"helper": [2, 4], ...}

# The dataflow edges: variables read by each instruction, resolved to
# their source (input, weight, $self, or the path of the last writer)
deps = instruction_dependencies(block=schema)
# {"0": [Dependency(input:x)], "5": [Dependency(instruction:0), ...], ...}
```

- Call sites anchor at the outermost enclosing instruction for calls nested
  in branches or loops; a container iteration is one static call site of
  the container path. A sub-block called several times is positioned at the
  balanced location of its anchors (mean or median of the indices).
- Instruction paths encode the nesting: top level by index (`"3"`), nested
  bodies as `{parent}/body/{i}`, `{parent}/else/{i}`,
  `{parent}/handlers/{h}/{i}`, `{parent}/finally/{i}` and the condition of
  a control flow as `{parent}/condition`.
- A `Dependency` carries `name` (the source variable), `kind` (`"input"`,
  `"weight"`, `"self"`, `"instruction"`, `"unresolved"`) and `origin` (the
  input/weight name, `$self`, or the last writer path).

## Dumping a schema

```python
from nnschemaparser import dump_schema

dump_schema(schema=schema, path="schema.txt")
```

The dump shows the inputs (with defaults), outputs, weights, skipped
children, sub-blocks and the resolved body of the block.

## Walking a schema programmatically

`ModelBlockSchema.body` is a list of instructions
(`OpInstruction`, `BlockInstruction`, `ControlInstruction`,
`TryInstruction`, `ReturnInstruction`, `ContinueInstruction`,
`BreakInstruction`), each with an optional `resolved`
(`TensorRepr | TensorReprWithValue | None`) field. Sub-blocks live in
`schema.sub_blocks` keyed by attribute name (container entries keyed by
`"container.index"`), and the other parsed methods in `schema.methods`.
