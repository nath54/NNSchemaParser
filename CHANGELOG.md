# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Instruction dependencies** — `instruction_dependencies(block)` resolves,
  for every instruction of a block body, the variables it reads into
  `Dependency` entries (`input`, `weight`, `self`, `instruction` or
  `unresolved`) keyed by nested instruction paths (`"3"`, `"5/body/0"`,
  `"5/condition"`, ...), pointing to the last earlier instruction binding
  each variable.
- **Method call sites** — `method_call_sites(block)` mirrors
  `sub_block_call_sites()` for the methods parsed as schemas: the top level
  body indices of their calls on self.

## [0.3.1] — 2026-11-02

### Added

- **Canvas positioning helpers** — `ModelBlockSchema.sub_block_call_sites()`
  returns the top level body indices where each sub-block path is called
  (calls nested in branches or loops anchor at their outermost enclosing
  instruction; a container iteration is one static call site), so a viewer
  can place a sub-block box between its call sites or at their balanced
  position when there are several.
- **Instruction dependencies** — `instruction_dependencies(block)` resolves,
  for every instruction of a block body, the variables it reads into
  `Dependency` entries (`input`, `weight`, `self`, `instruction` or
  `unresolved`) keyed by nested instruction paths (`"3"`, `"5/body/0"`,
  `"5/condition"`, ...), pointing to the last earlier instruction binding
  each variable — the dataflow edge list of a canvas.
- **Method call sites** — `method_call_sites(block)` mirrors
  `sub_block_call_sites()` for the methods parsed as schemas: the top level
  body indices of their calls on self.

## [0.3.0] — 2026-10-30

### Added

- **Frame-depth bookkeeping** — the interpreter maintains the stack of the
  executed block scopes, and `TraceStep` gains an optional `frame_depth`
  (0 = root block scope) filled by the `ActivationRecorder`, so consumers can
  locate the scope frame holding any `out_var`.
- **Stepped execution** — `SchemaInterpreter.step_run()`, a synchronous
  generator yielding one `StepInfo` per executed instruction in program order
  (per-iteration steps for loop bodies, nested sub-blocks included), returning
  the outputs and mutating the scope in place. `StepInfo` carries the step
  index, kind, variable name, op name, live scope frames and frame depth.
- **Scope edits** — `StepInfo.edit(values, shape, dtype, frame_depth,
  out_var)` patches a scope frame before the next instruction (mid-run tensor
  editing); in generator mode the edit is applied race-free by the worker
  thread, and the raw `interpreter.step_hook` mode applies it immediately.
  Abandoning the generator unwinds the run; the interpreter stays reusable.
- **`capture_run()` and `ExecutionRecord`** — the one-call capture workflow
  bundling the outputs, the `ActivationTrace`, the final scope and a lazily
  cached `gradients` property.
- **`compare_traces()` and `TraceDiff`** — alignment of two traces by
  container id with a role fallback surviving control-path changes, per-step
  statuses (`ok`, `shape_changed`, `value_unavailable`, `only_in_a`,
  `only_in_b`) and max/mean absolute differences; `diff_summary` counts them.

## [0.2.0] — 2026-10-30

### Added

- **Activation & gradient traces** — `TensorContainer` (a captured tensor
  identified by a stable `container_id` and a `slot_type`), `TraceStep`,
  `ActivationTrace` and `GradientTrace` alongside the `ModelBlockSchema`, and
  the `ActivationRecorder` `TraceListener` allocating them chronologically
  (`step 0001: Initial input tensor — container_id: input_0001`, ...). With
  `capture_grads=True`, `collect_gradients()` returns the backward gradients
  of the recorded activations after a `loss.backward()`, cross-referenced by
  `source_container_id`. The schema instructions keep their
  `capture_values=True` capture unchanged.

## [0.1.0] — 2025

Initial release.

### Added

- **`NNParser`** — parses a `torch.nn.Module` into a `ModelBlockSchema`
  graph that keeps the original variable names of the forward source, all
  sub-blocks (MoE experts included), other methods called by the forward,
  input default values and the skipped children.
- **`SchemaInterpreter`** — executes the schema faithfully by re-dispatching
  to the real torch callables, with optional per-instruction value capture
  (`capture_values=True`) and trace listeners (`TraceListener`,
  `TraceRecorder`).
- **`dump_schema`** — human readable dumps of the parsed schemas and of the
  resolved instruction values.
- Supported forward syntax: if/elif/else, for loops (containers, enumerate,
  iterables, tuple targets), while loops, try/except/finally, continue and
  break, comprehensions (list/set/dict/generator), dict displays, starred
  call arguments in any order, starred sequence displays, chained
  assignments, tuple unpacking, subscript assignments, `*args`/`**kwargs`
  signatures, f-strings, short-circuit boolean operators, chained
  comparisons, `not in`, super() calls (zero-argument and explicit),
  indexed self-calls (`self[0](...)`), explicit `.forward()` calls on
  sub-block targets, scalar inputs (`int`/`float`/`bool`/`str`) with
  defaults, and method schemas.
- 58-model HuggingFace robustness sweep (parse → interpret → compare with
  a maximum difference of 1e-4), runnable with
  `python -m tests.hf_sweep --force`.
- CI: fixture test suite (no network) on Python 3.10–3.13 with CPU torch,
  plus a style check.

[0.1.0]: https://github.com/nath54/NNViewer/releases/tag/v0.1.0
