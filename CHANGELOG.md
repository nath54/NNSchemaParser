# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
