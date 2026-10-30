# NNSchemaParser

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
depth) — this is what a WebUI consumes to display every intermediate value:

```
[00001] ENTER BLOCK MLP
[00002]   in x = torch.float32[2, 784] abs_sum=1241.4496 min=-3.0303 max=3.3046
[00003]   ENTER BLOCK Linear
[00004]     in input = torch.float32[2, 784] abs_sum=1241.4496 min=-3.0303 max=3.3046
[00005]     $t1 = F.linear(input, weight, bias)  [torch.float32[2, 256] abs_sum=215.3293 ...]
[00006]   EXIT BLOCK Linear -> (torch.float32[2, 256] abs_sum=215.3293 ...)
[00007]   hidden = CALL fc1 [Linear]  [torch.float32[2, 256] abs_sum=215.3293 ...]
```

## Installation

```bash
pip install .            # or: pip install -e .[research] for the harness extras
```

Requires Python ≥ 3.12 and PyTorch ≥ 2.0. The model classes must be importable
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
