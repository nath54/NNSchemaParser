# NNSchemaParser — Documentation

This document describes how NNSchemaParser works, how to use the schema it
produces, and which models are supported.

- [1. Overview](#1-overview)
- [2. How parsing works](#2-how-parsing-works)
  - [2.1 Phase 1 — collection](#21-phase-1--collection)
  - [2.2 Phase 2 — AST lowering](#22-phase-2--ast-lowering)
  - [2.3 Phase 3 — shape and dtype resolution](#23-phase-3--shape-and-dtype-resolution)
- [3. Schema reference](#3-schema-reference)
  - [3.1 `TensorRepr` / `TensorReprWithValue`](#31-tensorrepr--tensorreprwithvalue)
  - [3.2 `ModelBlockSchema`](#32-modelblockschema)
  - [3.3 Instructions](#33-instructions)
- [4. The schema interpreter](#4-the-schema-interpreter)
- [5. Usage recipes](#5-usage-recipes)
- [6. Supported syntax](#6-supported-syntax)
- [7. Supported models](#7-supported-models)
- [8. What does not work (yet)](#8-what-does-not-work-yet)
- [9. Validation harness](#9-validation-harness)
- [10. Project layout](#10-project-layout)

---

## 1. Overview

NNSchemaParser has two deliverables and nothing else:

1. **The parser** (`nnschemaparser/parser/`) — converts any `torch.nn.Module` into a
   `ModelBlockSchema` tree.
2. **The schema** (`nnschemaparser/core/`) — the data model: `ModelBlockSchema`,
   `TensorRepr`, `TensorReprWithValue` and the instruction types.

The **schema interpreter** (`nnschemaparser/interpreter/`) executes a schema in
two modes:

- **parse time** — under `torch._subclasses.fake_tensor.FakeTensorMode` (or with
  real tensors as a fallback) to fill the shape/dtype of every instruction;
- **capture time** — with real tensors and `capture_values=True` to record the
  value of every intermediate tensor in the `resolved` field of each instruction
  (e.g. for a WebUI graph viewer).

The design goal is **faithfulness**: the schema run must produce exactly the
same values as `model(...)`. The interpreter does not reimplement torch — it
re-dispatches every call to the real callable (`F.linear`, `tensor.softmax`,
sub-modules, …), so numerical behaviour is identical.

## 2. How parsing works

`NNParser.parse_model()` performs three phases:

```python
schema = NNParser.parse_model(
    model=model,                                        # any nn.Module
    inputs_tensors_shapes={"x": (2, 784)},              # required: input shapes
    inputs_tensors_dtypes={"x": "torch.float32"},       # optional, float32 default
    inputs_tensors_values={"x": x},                     # optional, for data-dependent shapes
)
```

### 2.1 Phase 1 — collection

For each module (`children.py`, `container_keys.py`):

- **Weights and buffers** — every direct `nn.Parameter` and registered buffer
  becomes a `weights` entry (`TensorReprWithValue` holding a detached snapshot).
  Live module attributes take precedence at execution time (see §4), so runtime
  mutations of the weights are still visible.
- **Sub-blocks** — every child `nn.Module` (direct attribute or
  `nn.ModuleList` / `nn.ModuleDict` / `nn.Sequential` entry) is parsed
  recursively into its own `ModelBlockSchema`. Containers are remembered so the
  lowering can distinguish `self.layers[0](x)` from `self.layer(x)`. ModuleDict
  entries are *not* splat-chained (a call on a ModuleDict is re-dispatched at
  runtime), while ModuleList/Sequential calls chain their outputs.
- **Literals** — plain Python attributes of the module (`self.hidden = 64`,
  `self.activation = F.gelu`, config values, …) are captured as constants of
  the block scope.

A child whose forward cannot be lowered (e.g. it calls `super()`) is **skipped
from the tree** instead of aborting the whole parse: the parser records the
skip, and the interpreter transparently re-dispatches calls to that child to
its **real module** at execution time. The graph stays runnable and the values
stay exact; only the expansion of that child is not shown.

### 2.2 Phase 2 — AST lowering

For each module, the source of `forward` is fetched with `inspect.getsource`
(`forward.py` unwraps the transformers v5 forward decorators so the real body
and its `__globals__` are found), parsed with `ast`, and lowered into a flat
list of `Instruction` objects (the `lowering/` package). The lowering keeps the
**original variable names**: an instruction writing `hidden_states` produces a
scope variable named `hidden_states`, and the WebUI can map any line back to
the source.

The result of the lowering is a pair `(body, output_vars)` — the instruction
list and the names of the variables returned by `return`.

### 2.3 Phase 3 — shape and dtype resolution

`resolution.py` executes the freshly built schema to fill the `resolved` field
of every instruction and the `inputs`/`outputs` of every reached block:

1. **Fake mode** — a first run under `FakeTensorMode` resolves everything with
   zero memory cost. Fake tensors are only created for the *inputs*; the real
   module weights are kept so any re-dispatched real module call works.
2. **Real fallback** — if the model contains data-dependent Python (e.g.
   `tensor.item()` branching, `int(...)` on tensor values), the fake run fails
   and the resolver re-runs the schema with real neutral tensors under
   `torch.no_grad()`, using the optional `inputs_tensors_values` when shapes
   are data-dependent.

If `inputs_tensors_shapes` is empty, the resolution pass is skipped entirely
(a shapeless schema, useful for pure structural inspection).

## 3. Schema reference

All classes live in `src/core/`. The tree of a model is a tree of
`ModelBlockSchema`; each block owns a flat list of instructions.

### 3.1 `TensorRepr` / `TensorReprWithValue`

```python
TensorRepr(shape: tuple[int, ...], dtype: str)          # e.g. (2, 16, 768), "torch.float32"
TensorReprWithValue(shape, dtype, value: torch.Tensor)  # snapshot with the real tensor
```

- `shape` holds **concrete ints** (symbolic dims are not preserved; the
  resolution runs with real or fake values so every dim is materialized).
- A **non-tensor** value produces `TensorRepr(shape=(), dtype=type_name)`,
  e.g. `dtype="bool"` for a python boolean, `"int"` for a python int.

### 3.2 `ModelBlockSchema`

| Field | Type | Meaning |
|---|---|---|
| `inputs` | `dict[str, TensorRepr]` | the forward parameters of the block, by name |
| `outputs` | `list[TensorRepr]` | the values of the `return` statement |
| `weights` | `dict[str, TensorReprWithValue]` | direct parameters/buffers, by attribute name |
| `sub_blocks` | `dict[str, ModelBlockSchema]` | child blocks, keyed by attribute path (`"attention"`, `"layers.0"`, …) |
| `body` | `list[Instruction]` | the lowered forward body |
| `module` | `Any \| None` | the source `nn.Module`, used by the interpreter for dynamic attribute resolution and weight seeding |

### 3.3 Instructions

All instruction classes derive from `Instruction`, which carries:

- `out_var: str` — the scope variable bound to the result (`""` for
  `ControlInstruction` / `TryInstruction` / `ReturnInstruction`);
- `resolved: TensorRepr | TensorReprWithValue | None` — filled by the
  resolution pass or by a capture run.

**`OpInstruction`** — a call to a torch operation, function or method
(`target` is the callable, `qualified_name` a display name such as
`F.softmax`), with `args: list[VarRef | Any]`, `kwargs`, and the optional
`args_var` / `kwargs_var` naming scope variables unpacked as `*args` / `**kwargs`.
Operators (`+`, `@`, `x < y < z`, chained comparisons, bitwise and shifts,
short-circuit `and`/`or`) are lowered to operator calls
(`operator.matmul`, `operator.lt`, …) or nested conditions.

**`BlockInstruction`** — a call to a sub-block, referencing it by
`BlockRef(path, index_var)`. When `splat_args` is True the single argument is a
tuple unpacked across a container chain (ModuleList/Sequential semantics).

**`ControlInstruction`** — `kind` is `"if"` or `"for"`:

- `if` — `condition` is an `OpInstruction` evaluating the condition (the
  lowering of short-circuit `and`/`or` produces *nested* `if` instructions);
  `body` / `else_body` are the branches.
- `for` — either iterates a `condition` expression, or a container of
  sub-blocks via `iter_container` + `bind_var` (+ `index_var` for
  `enumerate`). `bind_var` may hold a sub-block schema (container iteration)
  or any value.

Returns from anywhere inside a `body`/`else_body`/loop are propagated to the
caller of the block.

**`TryInstruction`** — a `try` with typed `handlers` (exception classes are
resolved at lowering time and bound to `handler_names`) and a `final_body`.

**`ReturnInstruction`** — `out_vars` are the names of the returned variables.
The interpreter binds them: a single value is returned directly, several
values are returned as a tuple.

## 4. The schema interpreter

```python
from nnschemaparser import SchemaInterpreter

interpreter = SchemaInterpreter(capture_values=True)
outputs = interpreter.run(block=schema, scope={"x": x.clone()})
```

- `scope` maps the **forward parameter names** to the values (mutated in
  place). The run returns the list produced by the block's `ReturnInstruction`
  (empty if the block returns nothing).
- **Weights** are seeded automatically: the *live attribute* of
  `block.module` is preferred over the parse-time snapshot, so mutations
  performed by the module during execution (dtype casts, cache updates, …)
  are visible.
- The module itself is bound to `$self` in the scope, so attribute accesses
  that could not be resolved statically are re-dispatched with `getattr` on
  the real module at run time (this is how skipped children and dynamic
  attributes keep working).
- Unbound forward parameters with defaults — including `**kwargs` — are seeded
  with their defaults (an empty dict for var-keyword).
- With `capture_values=True`, every instruction that produces a tensor gets a
  `TensorReprWithValue` in `instruction.resolved`; non-tensor results keep a
  shapeless `TensorRepr` describing the python type.

The interpreter is deliberately the *same* executor for both modes: the values
you capture in the WebUI are produced by exactly the code path that validated
the shapes at parse time.

### Tracing an execution

The interpreter accepts `listeners: list[TraceListener]` at construction and
notifies them of every event: block enters and exits, executed operations,
sub-block calls, control flows and try statements. The package ships
`TraceRecorder`, a listener writing the chronological indented text trace
(one line per event, with the value signature of every tensor) used by the
validation harness. Custom listeners subclass `TraceListener` and implement
only the hooks they need.

**Grad-mode note** — a few models (e.g. `beit`) produce slightly different
values with and without grad. Always run the reference and the capture under
the *same* grad mode; `torch.no_grad()` everywhere is the safe choice.

## 5. Usage recipes

### Parse and walk the graph

```python
schema = NNParser.parse_model(model=model, inputs_tensors_shapes={"x": (1, 3, 224, 224)})

def walk(block: ModelBlockSchema, depth: int = 0) -> None:
    print("  " * depth + type(block.module).__name__, list(block.sub_blocks))
    for child in block.sub_blocks.values():
        walk(child, depth + 1)

walk(schema)
```

### Capture every intermediate value

```python
with torch.no_grad():
    SchemaInterpreter(capture_values=True).run(block=schema, scope={"x": x})

for instruction in schema.body:
    if isinstance(instruction.resolved, TensorReprWithValue):
        print(instruction.out_var, instruction.resolved.shape,
              instruction.resolved.value.abs().sum().item())
```

### Compare the schema run with the real forward

```python
outputs = SchemaInterpreter().run(block=schema, scope={"x": x.clone()})
reference = model(x)
assert torch.allclose(outputs[0], reference, atol=1e-4)
```

### Inspect a block in isolation

Because a `ModelBlockSchema` is self-contained, any sub-block can be executed
on its own:

```python
attention = schema.sub_blocks["encoder"].sub_blocks["layers.0"].sub_blocks["attention"]
hidden = torch.randn(2, 16, 768)
out = SchemaInterpreter().run(block=attention, scope={"hidden_states": hidden})
```

### Dump a readable schema and an execution trace

The validation harness includes two text dumps used for debugging and for the
WebUI fixtures:

```python
from nnschemaparser import dump_schema

dump_schema(schema=schema, path="schema.txt")   # full recursive block dump
```

The schema dump shows, for every block, its inputs, outputs, weights and
sub-blocks, then the instruction body with the resolved shape of every step:

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
  activated = F.relu(hidden)  -> torch.float32[2, 256]
  $t1 = CALL fc2(activated)  -> torch.float32[2, 10]
  RETURN $t1
```

## 6. Supported syntax

The lowering covers the Python actually written in `forward` methods:

| Feature | Status |
|---|---|
| Calls to `torch.*` / `torch.nn.functional.*` / methods / operators | ✅ |
| Arithmetic, matmul, bitwise, shift operators | ✅ |
| Chained comparisons (`a < b < c`, each operand evaluated once) | ✅ |
| Short-circuit `and` / `or` (lowered to nested `if`) | ✅ |
| `if` / `elif` / `else` | ✅ |
| `for` over containers, iterables and `enumerate` | ✅ |
| Calls to `nn.ModuleList` / `nn.Sequential` entries (chained) | ✅ |
| Calls to `nn.ModuleDict` entries (runtime dispatch) | ✅ |
| `try` / `except Type as name` / `finally` | ✅ |
| List / set / dict comprehensions and generator expressions | ✅ |
| `*args` / `**kwargs` in signatures and at call sites | ✅ |
| Multiple / tuple return values | ✅ |
| `self.weight`, buffers, plain attributes, config objects | ✅ |
| `super()` inside a forward body | ⚠️ child skipped, re-dispatched to the real module |
| `while` loops | ❌ `UnsupportedSyntaxError` |
| `for` / `while` `else` clauses | ❌ |
| Calls to unknown globals (e.g. building an output dataclass) | ⚠️ child skipped / ❌ at root level |
| `self[index](...)` indexed self-calls | ❌ (not yet) |
| Symbolic / dynamic shapes | ⚠️ concrete shapes are resolved (fake mode → real fallback) |

Unsupported constructs raise `UnsupportedSyntaxError` (with the offending line)
instead of producing a wrong graph. At the root level the parse aborts; inside
a child module the child is skipped and re-dispatched at run time.

## 7. Supported models

The sweep covers **58 models, all passing with `maxdiff = 0.0`** between the
schema interpreter and the real forward. Timings on CPU (Windows, Python 3.12).

### Text (27)

| Model | Instructions | Parse | Execute |
|---|---:|---:|---:|
| bert_tiny | 336 | 0.1s | 0.0s |
| bert_l4_h256 | 284 | 0.1s | 0.0s |
| bert_l2_h128 | 180 | 0.0s | 0.0s |
| minilm_l6 | 388 | 0.1s | 0.0s |
| distilbert | 201 | 0.1s | 0.0s |
| albert_base | 92 | 0.1s | 0.1s |
| roberta_base | 700 | 0.2s | 0.1s |
| deberta_v3_xsmall | 1858 | 0.5s | 0.4s |
| electra_small | 688 | 0.2s | 0.0s |
| squeezebert | 90 | 1.1s | 1.0s |
| mobilebert | 3230 | 0.8s | 0.3s |
| layoutlm_base | 631 | 0.2s | 0.1s |
| tiny_gpt2 | 418 | 0.1s | 0.0s |
| distilgpt2 | 540 | 0.2s | 0.2s |
| gpt_neo_125m | 1002 | 0.3s | 0.3s |
| opt_125m | 27 | 0.4s | 0.2s |
| pythia_70m | 250 | 0.1s | 0.1s |
| qwen2_0_5b | 16 | 1.0s | 0.8s |
| smollm2_135m | 16 | 0.6s | 0.3s |
| bloom_560m | 2315 | 1.2s | 1.1s |
| tiny_starcoder_py | 787 | 0.4s | 0.2s |
| mamba_130m | 2555 | 0.5s | 0.5s |
| t5_small | 1535 | 0.3s | 0.2s |
| flan_t5_small | 2243 | 0.4s | 0.2s |
| bart_base | 299 | 0.3s | 0.2s |
| opus_mt_en_de | 282 | 0.2s | 0.1s |
| switch_base_8 (MoE) | 5415 | 1.2s | 0.6s |

### Vision (10)

| Model | Instructions | Parse | Execute |
|---|---:|---:|---:|
| vit_base | 490 | 0.4s | 0.4s |
| deit_tiny | 490 | 0.2s | 0.1s |
| resnet_18 | 691 | 0.1s | 0.1s |
| mobilenet_v2 | 1963 | 0.2s | 0.4s |
| mit_b0 (SegFormer) | 769 | 0.2s | 0.1s |
| mobilevit_small | 1578 | 0.4s | 0.5s |
| levit_128s | 2190 | 0.2s | 0.2s |
| convnext_tiny | 558 | 0.2s | 0.3s |
| beit_base | 1747 | 0.4s | 0.4s |
| detr_resnet_50 | 2652 | 2.8s | 3.1s |

### Audio (7)

| Model | Instructions | Parse | Execute |
|---|---:|---:|---:|
| wav2vec2_base | 30 | 0.3s | 0.1s |
| wav2vec2_base_960h (CTC head) | 50 | 0.3s | 0.1s |
| hubert_base | 40 | 0.3s | 0.1s |
| ast_speech_commands | 473 | 0.3s | 0.3s |
| whisper_tiny | 223 | 0.5s | 0.6s |
| s2t_small | 517 | 0.7s | 1.0s |
| mms_tts_eng | 2385 | 1.1s | 0.6s |

### Multimodal (4)

| Model | Instructions | Parse | Execute |
|---|---:|---:|---:|
| clip_base | 914 | 0.3s | 0.1s |
| blip_caption | 1900 | 0.8s | 0.8s |
| trocr_small_handwritten | 572 | 0.5s | 0.5s |
| vit_gpt2_caption | 1574 | 0.8s | 0.7s |

### Locally built architectures (10)

| Model | Category | Instructions |
|---|---|---:|
| local_mlp | MLP | 16 |
| local_lstm | RNN | 14 |
| local_rnn | RNN | 13 |
| local_gru_seq2seq | RNN | 11 |
| local_bilstm_attention | RNN | 19 |
| local_resnet_cnn | CNN | 60 |
| local_conv_autoencoder | AutoEncoder | 59 |
| local_gan_generator | GAN | 51 |
| local_gan_discriminator | GAN | 23 |
| local_gcn | GNN | 6 |

Notable covered architectures: mixture-of-experts (`switch_base_8`, with its
router dtype mutation), SSM (`mamba`), MoE-visited experts present in the tree
even though a single trace never visits them, encoder/decoder wrappers
(`VisionEncoderDecoderModel`, `BlipForConditionalGeneration`), speech CTC and
TTS heads.

## 8. What does not work (yet)

Known open items, in rough order of impact:

1. **`while` loops** — raise `UnsupportedSyntaxError`. Rare in forward methods;
   not implemented.
2. **`self[index](...)` indexed self-calls** — e.g.
   `torch.nn.utils.parametrizations` style `ParametrizationList`. Not yet
   lowered (would need an indexed-self BlockRef).
3. **Calls to unknown globals** in a forward body (e.g. constructing an output
   dataclass such as `BaseModelOutput(...)`) — currently skips the child or
   aborts the root parse; the harness input kinds route around it.
4. **Symbolic dims** — shapes are always materialized to ints; a schema parsed
   with one batch size reports that batch size (the interpreter itself is
   shape-agnostic, so re-running with other inputs still works).
5. **Training-mode randomness** — dropout and similar layers run for real
   (the interpreter re-dispatches), so compare in `eval()` and under a fixed
   seed, or the reference and the schema run will legitimately differ.
6. **Decorated forwards** — handled for the transformers v5
   `@check_model_inputs`-style wrappers via `resolve_forward`; exotic wrappers
   that hide the true body from `inspect.getsource` will not parse.

Anything unparseable in a *child* is skipped but still executed through its
real module — so a model with a few exotic leaves still produces exact values;
only those leaves are not expanded into instructions.

## 9. Validation harness

The harness lives in `tests/` and is used to validate every change to the
parser or the interpreter:

```bash
# The whole registry (or a slice / a single model / a category)
python -m tests.hf_sweep --force
python -m tests.hf_sweep --range A:F --force
python -m tests.hf_sweep --only vit_gpt2 --force
python -m tests.hf_sweep --category Vision --force

# Regressions
python -m tests.test_parser          # MoE regression
python -m tests.test_speech_models   # 6 speech models
```

Each model run produces `outputs/hf_sweep/<name>/`:

- `result.json` — status, maxdiff, instruction counts, phase durations;
- `schema.txt` — the recursive readable dump of the schema;
- `trace.txt` — the per-operation execution trace with captured values.

An aggregate `outputs/hf_sweep/summary.md` and an `.errors_to_fix.md` log are
regenerated on every sweep. Weights are deleted from the HF cache after each
model so the full sweep fits on a small disk.

## 10. Project layout

```
nnschemaparser/              The package (pip installable)
  __init__.py                Public API: NNParser, SchemaInterpreter, schema types
  forward.py                 forward unwrapping (transformers v5 decorators)
  dump.py                    readable text dumpers of a schema tree

  core/                      The schema data model
    model_schema.py          ModelBlockSchema
    tensor_schema.py         TensorRepr, TensorReprWithValue
    instructions.py          VarRef, BlockRef, Op/Block/Control/Try/Return instructions
    container_keys.py        container key helpers (dotted paths, dict keys)

  parser/                    The parser
    parser.py                NNParser.parse_model — the public entry point
    children.py              child/container/weight collection
    resolution.py            ShapeResolver — fake mode + real fallback
    lowering/                forward source AST -> instructions
      base.py, body.py, statements.py, control_flow.py
      calls.py, block_calls.py, expressions.py, operators.py
      values.py, naming.py, effects.py, runtime_ops.py, errors.py

  interpreter/               The schema interpreter
    interpreter.py           SchemaInterpreter.run — public entry point
    execution.py             op / block call / return execution
    control_flow.py          if / for / try execution
    scope.py                 scope seeding (weights, defaults, $self)
    tracer.py                TraceListener hooks and TraceRecorder

tests/                       Validation harness and regressions
docs/                        This documentation
```
