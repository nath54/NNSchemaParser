# NNSchemaParser

[![PyPI](https://img.shields.io/pypi/v/nnschemaparser)](https://pypi.org/project/nnschemaparser/)
[![Python](https://img.shields.io/pypi/pyversions/nnschemaparser)](https://pypi.org/project/nnschemaparser/)

**NNSchemaParser** converts a PyTorch `nn.Module` into a serializable, inspectable
graph representation — the **`ModelBlockSchema`** — that keeps the *original variable
names of the forward source* and *every sub-module* (including MoE experts that a
single trace would never visit). The schema can then be executed by a built-in
**schema interpreter** that re-dispatches every recorded call to the real torch
callables, so the graph runs exactly like the model — and can optionally capture
the value of every intermediate tensor.

```
   nn.Module  ──parse──►  ModelBlockSchema  ──interpret──►  outputs / captured values
  (your code)              (names + graph)                   (identical to model(x))
```

## Why not `torch.fx`?

| | `torch.fx.symbolic_trace` | **NNSchemaParser** |
|---|---|---|
| Variable names | lost (`_0`, `_1`, …) | **kept** (`hidden_states`, `query`, …) |
| Untaken branches (MoE experts) | missing | **all present** |
| Control flow | `graph_module` nodes only | `if` / `for` / `try` as first-class instructions |
| `*args` / `**kwargs` | limited | supported in signatures and call sites |
| Comprehensions | unsupported | compiled to helper functions |
| Re-executable | via the FX machinery | via a small self-contained interpreter |

## Quick start

```python
import torch
from torch import nn

from nnschemaparser import NNParser, SchemaInterpreter, TraceRecorder, dump_schema


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(x)
        activated = hidden.relu()
        return self.fc2(activated)


model = MLP().eval()
x = torch.randn(2, 784)

# 1. Parse the model into a schema (resolves shapes and dtypes)
schema = NNParser.parse_model(
    model=model,
    inputs_tensors_shapes={"x": (2, 784)},
    inputs_tensors_dtypes={"x": "torch.float32"},
    inputs_tensors_values={"x": x},
)

# 2. Dump a readable view of the whole block tree
dump_schema(schema=schema, path="schema.txt")

# 3. Execute it with a tracer and compare with the real forward
tracer = TraceRecorder()
outputs = SchemaInterpreter(capture_values=True, listeners=[tracer]).run(
    block=schema,
    scope={"x": x.clone()},
)
tracer.save("trace.txt")
reference = model(x)
assert torch.equal(outputs[0], reference)
```

The schema dump of `MLP` reads exactly like the source — original variable
names, every sub-module, and the resolved shape of every step:

```
=== BLOCK MLP ===
inputs:
  - x = torch.float32[2, 784]
outputs:
  - torch.float32[2, 10]
weights:
  (none)
sub-blocks:
  - fc1(Linear)
  - fc2(Linear)
body:
  hidden = CALL fc1(x)  -> torch.float32[2, 256]
  $t1 = getattr(hidden, 'relu')  -> builtin_function_or_method
  activated = hidden.relu()  -> torch.float32[2, 256]
  $t2 = CALL fc2(activated)  -> torch.float32[2, 10]
  RETURN $t2
```

The tracer records the same execution chronologically, with the value of
every intermediate tensor (one line per event, indented by the nesting
depth) — ready for any downstream tool to display the model step by step:

```
[00001] ENTER BLOCK MLP
[00002]   in x = torch.float32[2, 784] abs_sum=1241.4496 min=-3.0303 max=3.3046
[00003]   ENTER BLOCK Linear
[00004]     in input = torch.float32[2, 784] abs_sum=1241.4496 min=-3.0303 max=3.3046
[00005]     $t1 = F.linear(input, weight, bias)  [torch.float32[2, 256] abs_sum=215.3293 ...]
[00006]   EXIT BLOCK Linear -> (torch.float32[2, 256] abs_sum=215.3293 ...)
[00007]   hidden = CALL fc1 [Linear]  [torch.float32[2, 256] abs_sum=215.3293 ...]
```

## The schema classes in detail

This section documents every class of the schema data model so you can import
them, walk them and build your own tooling (WebUI, canvas viewer, analysis
scripts) on top of them. Everything is importable from the package root:

```python
from nnschemaparser import (
    ModelBlockSchema, TensorRepr, TensorReprWithValue, TensorContainer,
    VarRef, BlockRef, OpInstruction, BlockInstruction, ControlInstruction,
    TryInstruction, ReturnInstruction, ContinueInstruction, BreakInstruction
)
```

### Tensor representations

| Class | Fields | Meaning |
|---|---|---|
| `TensorRepr` | `shape: tuple[int, ...]`, `dtype: str` | A tensor flowing through the graph, described by its shape and dtype only (e.g. `(2, 16, 768)`, `"torch.float32"`). A non-tensor value gets `shape=()` and `dtype=type(value).__name__` (e.g. `"bool"`). |
| `TensorReprWithValue` | + `value: torch.Tensor` | A `TensorRepr` holding a **detached snapshot** of the real tensor. Produced by `capture_values=True` runs and by the weight collection. |
| `TensorContainer` | + `container_id: str`, `slot_type: str` | A captured tensor identified by a **stable unique id** (e.g. `"input_0001"`, `"layers_0_0002"`) and by its role (`"input"`, `"activation"`, `"output"`, `"gradient"`). This is the payload of the traces (see next section). |

### `ModelBlockSchema` — one block of the tree

A parsed model is a **tree of `ModelBlockSchema`**: one block per `nn.Module`,
each self-contained and independently executable.

| Field | Type | Meaning |
|---|---|---|
| `inputs` | `dict[str, TensorRepr]` | The forward parameters, **by name** (the keys are the names you can pass to `SchemaInterpreter.run(scope=...)`). |
| `outputs` | `list[TensorRepr]` | The values of the `return` statement, in order. |
| `weights` | `dict[str, TensorReprWithValue]` | The direct `nn.Parameter`s and buffers, by attribute name, snapshotted at parse time (live attributes take precedence at run time). |
| `sub_blocks` | `dict[str, ModelBlockSchema]` | The child blocks, keyed by attribute path (`"attention"`, `"layers.0"`, `"experts.3"`, ...). Container entries are keyed by `"container.index"`. |
| `body` | `list[Instruction]` | The lowered forward body — the graph to display. |
| `module` | `nn.Module \| None` | The live source module, used for weight seeding and dynamic attribute resolution. |
| `methods` | `dict[str, ModelBlockSchema]` | The other methods of the module called by the forward, parsed as schemas too. |
| `method_name` | `str \| None` | The parsed method name when the block describes a method instead of the forward. |
| `input_defaults` | `dict[str, Any]` | The default values of the optional forward parameters. |
| `skipped` | `dict[str, str]` | The children that could not be parsed, by name with their reason — they are re-dispatched to their real module at run time. |

### Instruction types — the `body` of a block

Every instruction carries:

- **`out_var: str`** — the **original variable name** of the forward source the
  result is bound to (e.g. `"hidden_states"`), or `""` for instructions that
  bind nothing (`ControlInstruction`, `TryInstruction`, `ReturnInstruction`,
  `ContinueInstruction`, `BreakInstruction`). Temporary results created by the
  lowering get synthetic names like `"$t1"`.
- **`resolved: TensorRepr \| TensorReprWithValue \| None`** — the shape/dtype of
  the result, and its **value** after a capture run. This is what a viewer
  displays on each node.

| Class | Extra fields | Represents |
|---|---|---|
| `OpInstruction` | `target`, `qualified_name`, `args`, `kwargs`, `args_var`, `kwargs_var` | A call to a torch op, function, method or operator. `qualified_name` is the display name (`"torch.nn.functional.softmax"`); `args`/`kwargs` hold `VarRef`s or literals; `args_var`/`kwargs_var` name scope variables unpacked as `*args`/`**kwargs`. Operators are lowered to `operator.*` calls. |
| `BlockInstruction` | `block_ref: BlockRef`, `args`, `kwargs`, `splat_args`, `args_var`, `kwargs_var` | A call to a sub-block. `splat_args=True` means the single argument tuple is unpacked across a container chain (`nn.ModuleList`/`nn.Sequential` semantics). |
| `ControlInstruction` | `kind: "if"\|"for"\|"while"`, `condition`, `body`, `else_body`, `bind_var`, `iter_container`, `index_var` | An `if`/`elif`/`else` or a loop as a first-class node: `condition` is an `OpInstruction` evaluating the condition (or the iterable), `body`/`else_body` are nested instruction lists. Returns inside them propagate to the caller of the block. |
| `TryInstruction` | `handlers`, `handler_names`, `final_body` | A `try`/`except Type as name`/`finally`, with the exception classes resolved at lowering time. |
| `ReturnInstruction` | `out_vars: list[str]` | The names of the returned variables; a single value is returned directly, several as a tuple. |
| `ContinueInstruction` / `BreakInstruction` | — | Loop control signals; `finally` bodies still run. |

The two reference types used inside instructions:

- **`VarRef(name)`** — a reference to a variable of the current block scope.
  When you see `VarRef("hidden_states")` in `args`, it means "the value bound
  to `hidden_states`" — this is how you draw the edges between nodes.
- **`BlockRef(path, index_var)`** — a reference to a sub-block by attribute
  path; `index_var` names the scope variable holding the container index or
  key (dynamic dispatch into `ModuleList`/`ModuleDict`).

### Walking a schema programmatically

```python
from nnschemaparser import SchemaInterpreter, BlockInstruction, ControlInstruction

# 1. Run with capture to fill instruction.resolved with the values
outputs = SchemaInterpreter(capture_values=True).run(block=schema, scope={"x": x})

# 2. Walk the instructions of a block: each one is a node of your canvas
def walk(block, depth=0):
    for instruction in block.body:
        resolved = instruction.resolved
        shape = resolved.shape if resolved else None
        print("  " * depth, instruction.out_var, type(instruction).__name__, shape)
        # A BlockInstruction nests the sub-block it calls: the schema comes
        # from block.sub_blocks[instruction.block_ref.path], or directly from
        # the loop variable for container iteration
        # A ControlInstruction nests its two branches: instruction.body and
        # instruction.else_body are instruction lists, recurse on them too

walk(schema)
```

A `ControlInstruction` is a compound node: render its `body` and `else_body`
as the two branches of a diamond, nested recursively. A `BlockInstruction` is
a compound node too: its `block_ref.path` keys into `block.sub_blocks` (or the
loop variable holds the sub-block schema directly for container iteration).

## Activation & gradient traces

The schema is the **static** structure: parse it once, keep it forever. The
**values** of an execution live in separate, disposable artifacts — the
traces — so one schema can serve any number of runs (one per input sample).

Every captured tensor becomes a **`TensorContainer`** with a stable id and a
slot type, and the **`ActivationRecorder`** (a `TraceListener`) allocates them
chronologically into an **`ActivationTrace`**:

```python
from nnschemaparser import NNParser, SchemaInterpreter, ActivationRecorder

schema = NNParser.parse_model(model=model, inputs_tensors_shapes={"x": (2, 16)})

recorder = ActivationRecorder(capture_grads=True)   # grads: see below
outputs = SchemaInterpreter(capture_values=True, listeners=[recorder]).run(
    block=schema, scope={"x": x.clone().requires_grad_(True)}
)

trace = recorder.trace
for step in trace.steps:
    print(step.step, step.description, step.container_id, step.tensor.shape)
```

The trace of the MLP example reads exactly as you would draw it on a canvas:

```
step 0001: Initial input tensor | container_id: input_0001 | torch.float32[2, 16] | var: x
step 0002: Linear projection | container_id: fc1_0002 | torch.float32[2, 32] | var: $t1 | block: fc1
step 0003: Call to sub-block fc1 (Linear) | container_id: MLP_0003 | torch.float32[2, 32] | var: hidden | block: MLP
step 0004: ReLU activation | container_id: MLP_0004 | torch.float32[2, 32] | var: activated | block: MLP
step 0005: Linear projection | container_id: fc2_0005 | torch.float32[2, 8] | var: $t1 | block: fc2
step 0006: Call to sub-block fc2 (Linear) | container_id: MLP_0006 | torch.float32[2, 8] | var: $t2 | block: MLP
step 0007: Model output tensor | container_id: output_0007 | torch.float32[2, 8] | var: $t2 | block: MLP
```

Each `TraceStep` holds:

| Field | Meaning |
|---|---|
| `step` | The 1-based chronological step number. |
| `container_id` | The stable id of the tensor (`"input_0001"`, `"{block_path}_{step}"` with dots replaced by underscores, `"output_0007"`). |
| `description` | A human readable description (`"Initial input tensor"`, `"Tensor multiplication"`, `"Call to sub-block fc1 (Linear)"`, ...). |
| `slot_type` | `"input"`, `"activation"`, `"output"` or `"gradient"`. |
| `out_var` | The original variable name of the forward source. |
| `op_name` | The qualified name of the producing op, or the called sub-block path. |
| `block_path` | The attribute path of the block that produced the tensor — join it with the schema tree to attach values to nodes. |
| `tensor` | The `TensorContainer` itself (shape, dtype, detached value). |
| `source_container_id` | Gradient steps only: the activation container id the gradient belongs to. |

`ActivationTrace` (and `GradientTrace`) also provide `by_container(id)` and
`save(path)` (one formatted line per step, as shown above).

### Gradients

The interpreter re-dispatches to the real torch callables, so an interpreted
run **builds a real autograd graph**. With `ActivationRecorder(capture_grads=True)`,
the recorder calls `retain_grad()` on every captured tensor; after a backward
pass, `collect_gradients()` returns the cross-referenced `GradientTrace`:

```python
loss = outputs[0].pow(2).sum()      # your loss, from the interpreter outputs
loss.backward()

gradients = recorder.collect_gradients()   # after the backward pass
for step in gradients.steps:
    print(step.container_id, "<-", step.source_container_id,
          step.tensor.value.abs().mean().item())
```

The gradient steps **mirror the activation steps in the same order**: the
container id of a gradient step is its activation id prefixed with `grad_`
(e.g. `grad_MLP_0004`) and `source_container_id` points back to the activation
step — so a viewer can draw the activation and the gradient of a node
side by side, keyed by the same id. The schema itself is never modified by a
run: `instruction.resolved` keeps the values of the last capture, while the
traces remain the per-run history.

## Stepped runs, edits and trace comparison (0.3.0)

The interpreter can also be driven **instruction by instruction**, and the
scope can be **edited mid-run** — the base of interactive viewers, debuggers
and counterfactual analyses:

```python
from nnschemaparser import SchemaInterpreter

interpreter = SchemaInterpreter(capture_values=True)

# Step mode: one StepInfo per executed instruction, in program order
for step in interpreter.step_run(block=schema, scope={"x": x}):
    print(step.step_index, step.kind, step.out_var, step.frame_depth)

    # Edit a scope frame before the next instruction (here: the tensor the
    # step just bound, at its own frame depth)
    if step.out_var == "activated":
        step.edit(values=torch.zeros_like(step.value))
outputs = interpreter.step_outputs   # also the generator return value
```

Each `StepInfo` exposes the live `scope_frames` of the execution and its
`frame_depth` (0 = root block), so an edit can target any visible variable:
`step.edit(values, shape, dtype, frame_depth=0, out_var="x")`. The stream is
deterministic for a schema, input and control path, which makes `step_index`
a stable fork key: re-running with an edit queued at step *N* replays the
prefix and diverges after it. Condition expressions of `if`/`while`
re-evaluations are not stepped; abandoning the generator unwinds the run.

The one-call capture workflow bundles everything:

```python
record = capture_run(block=schema, scope={"x": x}, capture_grads=True)
# record.outputs, record.trace (ActivationTrace), record.scope
loss = record.outputs[0].pow(2).sum(); loss.backward()
gradients = record.gradients            # the cross-referenced GradientTrace
```

And two runs are compared step by step — the diffs are aligned by container
id, with a role fallback surviving control-path changes:

```python
diffs = compare_traces(trace_a=base_record.trace, trace_b=fork_trace, atol=1e-6)
# each TraceDiff: status ("ok", "shape_changed", "only_in_a"/"only_in_b"),
# max_abs_diff, mean_abs_diff, changed; diff_summary(diffs) counts them
```

## What is supported

- All the usual forward constructs: `if`/`elif`/`else`, `for` (containers,
  `enumerate`, iterables, tuple targets), `while`, `try`/`except`/`finally`,
  `continue`/`break`, comprehensions, dict displays, f-strings.
- Starred arguments in any position (`f(*shape, -1)`), starred sequence
  displays, `*args`/`**kwargs` signatures and call sites, chained
  assignments, tuple unpacking, subscript assignments.
- `super()` calls (zero-argument and explicit), indexed self-calls
  (`self[0](...)`), explicit `.forward()` calls on sub-blocks, `nn.ModuleDict`
  runtime dispatch, chained `nn.ModuleList`/`nn.Sequential` calls.
- **Method schemas** — the other methods a forward delegates to (e.g.
  `get_image_features`) are parsed too, and executed as schemas.
- Scalar inputs (`int`/`float`/`bool`/`str`) typed from annotations or
  example values, with their defaults recorded in the schema.
- Children whose forward uses unsupported syntax are **skipped and
  re-dispatched** to the real module — recorded in `schema.skipped`, so the
  graph always runs and the values always match.

## Installation

```bash
pip install nnschemaparser            # from PyPI
pip install .                         # or from a clone
pip install -e .[research]            # with the sweep harness extras
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.0. The model classes must be importable
from a **real file** (the parser reads the forward source with
`inspect.getsource`, which fails for classes defined in a REPL or stdin).

## Validation

A robustness sweep of **58 models** — 48 Hugging Face hub checkpoints (text,
vision, audio, multimodal) plus 10 locally built architectures (MLP, RNN, CNN,
autoencoder, GAN, GNN) — runs each model through *parse → interpret → compare
against the real forward*. **All 58 pass with `maxdiff = 0.0`** against the real
forward. See [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md) for the full model
matrix and the sweep CLI.

## Documentation

- **[docs/API.md](docs/API.md)** — the public API: parsing, executing,
  tracing, dumping and walking schemas.
- **The schema classes in detail** (above) — every class of the schema data
  model, field by field, for building your own tooling on top.
- **Activation & gradient traces** (above) — the `TensorContainer`,
  `ActivationTrace`, `GradientTrace` and `ActivationRecorder` classes for the
  per-run value capture keyed by stable container ids.
- **[docs/DOCUMENTATION.md](docs/DOCUMENTATION.md)** — full documentation:
  how the parser works, the schema and instruction reference, the
  interpreter, usage recipes, supported syntax, supported/unsupported
  models, and limitations.
- **[CHANGELOG.md](CHANGELOG.md)** — the release history.
- `tests/hf_sweep.py` — the validation harness CLI (58 HuggingFace models).
- `tests/test_new_syntax.py`, `tests/test_parser.py`,
  `tests/test_methods_and_scalars.py` — the fixture test suite (pytest,
  no network needed), also runnable with `python -m tests.test_new_syntax`.
- `tests/test_speech_models.py` — the espnet/torchaudio regression tests.

## License

MIT — see [LICENSE](LICENSE).
