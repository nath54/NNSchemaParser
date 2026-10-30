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
