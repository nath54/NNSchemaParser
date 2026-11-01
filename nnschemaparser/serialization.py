"""
JSON (de)serialization of the schemas and the traces.

schema_to_dict turns a ModelBlockSchema tree into JSON safe dicts: the block
tree, the lowered instructions with their original variable names, and —
behind toggle parameters — the instruction indices and nested paths, the
resolved variable dependencies, the sub-block and method call sites and the
tensor values. dict_to_schema rebuilds a runnable viewer schema from the
payload, re-resolving the callables by their dotted names.

The traces are serialized by trace_to_dict: one entry per step, keyed by the
stable container ids, with the optional base64 values of the captured
tensors. The module of the blocks is not serialized: the decoded schema is
viewer only (module None) until the caller re-attaches a live module with
load_json(module=...).
"""

# Import Modules
from typing import Any, Callable

import base64
import importlib
import json

import torch
from torch import nn, Tensor

from nnschemaparser.core.dependencies import (
    Dependency,
    instruction_dependencies,
    method_call_sites
)
from nnschemaparser.core.diff import TraceDiff, diff_summary
from nnschemaparser.core.instructions import (
    BlockInstruction,
    BlockRef,
    BreakInstruction,
    ContinueInstruction,
    ControlInstruction,
    Instruction,
    OpInstruction,
    ReturnInstruction,
    TryInstruction,
    VarRef
)
from nnschemaparser.core.model_schema import ModelBlockSchema
from nnschemaparser.core.tensor_schema import (
    PLACEHOLDER_DTYPE,
    TensorContainer,
    TensorRepr,
    TensorReprWithValue
)
from nnschemaparser.core.traces import (
    ActivationTrace,
    GradientTrace,
    TensorTrace,
    TraceStep
)
from nnschemaparser.parser.lowering.errors import UnsupportedSyntaxError
from nnschemaparser.parser.lowering.naming import (
    RESOLVE_NAMESPACE,
    GlobalCallable,
    resolve_dotted_name
)

# The version of the JSON format, bumped on breaking payload changes
FORMAT_VERSION: int = 1


def schema_to_dict(
    block: ModelBlockSchema,
    include_indices: bool = True,
    include_dependencies: bool = True,
    include_call_sites: bool = True,
    include_resolved: bool = True,
    include_values: bool = False,
    trace: TensorTrace | list[TensorTrace] | None = None
) -> dict[str, Any]:
    """
    Encode a block schema tree into JSON safe dicts.

    Args:
        block (ModelBlockSchema): The block schema to encode.
        include_indices (bool): When True, every instruction carries its
            "index" (its position in its own body) and its "path" (the
            nested path the dependencies are keyed by, e.g. "5/body/0").
        include_dependencies (bool): When True, every instruction carries
            its resolved variable dependencies (the dataflow edges). Takes
            effect only with include_indices, as the dependencies are keyed
            by the instruction paths.
        include_call_sites (bool): When True, every block carries its
            "sub_block_call_sites" and "method_call_sites" index maps.
        include_resolved (bool): When True, every instruction carries the
            resolved shape and dtype of its output.
        include_values (bool): When True, the tensor values of the resolved
            representations and of the weights are exported as base64
            encoded raw bytes (heavy, off by default).
        trace (TensorTrace | list[TensorTrace] | None): The optional trace
            or traces of a run (activations, gradients or both): the
            container ids are attached to the instructions binding their
            variables, as the "containers" lists, so one payload links the
            graph structure to the captured tensors of the run.

    Returns:
        dict[str, Any]: The JSON safe payload of the block tree.
    """

    # The options gathered once, passed down the recursion
    options: dict[str, Any] = {
        "include_indices": include_indices,
        "include_dependencies": include_dependencies,
        "include_resolved": include_resolved,
        "include_values": include_values,
        "containers": _container_index(
            traces=trace,
            root_name=type(block.module).__name__ if block.module is not None else ""
        )
    }

    # Encode the block tree from the root, each block resolving its own
    # dependencies, keyed by the nested paths of its own instructions
    payload: dict[str, Any] = _encode_block(
        block=block,
        path="",
        options=options,
        include_call_sites=include_call_sites
    )

    # The format version header, for the forward compatibility
    payload["format_version"] = FORMAT_VERSION
    return payload


def dict_to_schema(payload: dict[str, Any]) -> ModelBlockSchema:
    """
    Decode a schema payload into a viewer block schema tree.

    The decoded blocks hold no live module: the schema can be walked and
    dumped, and executed for the calls re-resolvable by dotted name, but
    the skipped children and the self method calls need a module. Pass the
    live module to load_json to re-attach it.

    Args:
        payload (dict[str, Any]): The payload built by schema_to_dict.

    Returns:
        ModelBlockSchema: The decoded block schema tree.

    Raises:
        ValueError: When an argument or an exception type cannot be decoded.
    """

    # The decoded tree, without any live module
    return _decode_block(payload=payload)


def dump_json(
    block: ModelBlockSchema,
    path: str,
    include_indices: bool = True,
    include_dependencies: bool = True,
    include_call_sites: bool = True,
    include_resolved: bool = True,
    include_values: bool = False,
    trace: TensorTrace | list[TensorTrace] | None = None
) -> None:
    """
    Write a block schema tree to a JSON file.

    Args:
        block (ModelBlockSchema): The block schema to write.
        path (str): The path of the output JSON file.
        include_indices (bool): Whether the instruction indices and paths
            are exported.
        include_dependencies (bool): Whether the variable dependencies are
            exported (needs include_indices).
        include_call_sites (bool): Whether the sub-block and method call
            site index maps are exported.
        include_resolved (bool): Whether the resolved shapes and dtypes are
            exported.
        include_values (bool): Whether the tensor values are exported as
            base64 encoded raw bytes.
        trace (TensorTrace | list[TensorTrace] | None): The optional trace
            or traces of a run, their container ids attached to the
            instructions binding their variables.
    """

    # The encoded payload, written with the format version header
    payload: dict[str, Any] = schema_to_dict(
        block=block,
        include_indices=include_indices,
        include_dependencies=include_dependencies,
        include_call_sites=include_call_sites,
        include_resolved=include_resolved,
        include_values=include_values,
        trace=trace
    )
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path: str, module: Any | None = None) -> ModelBlockSchema:
    """
    Read a block schema tree from a JSON file.

    Args:
        path (str): The path of the JSON file to read.
        module (Any | None): The optional live module to re-attach to the
            decoded tree, walked by attribute path so the skipped children
            and the self method calls resolve again.

    Returns:
        ModelBlockSchema: The decoded block schema tree.
    """

    # The payload read from the file, decoded into a tree
    with open(path, "r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)

    # The format version guard, for the forward compatibility
    version: int = payload.get("format_version", 0)
    if version != FORMAT_VERSION:
        raise ValueError(f"Unsupported schema format version: {version}")

    # The decoded tree, with the optional module re-attached by path
    block: ModelBlockSchema = dict_to_schema(payload=payload)
    if module is not None:
        _reattach_module(block=block, module=module)
    return block


def trace_to_dict(trace: TensorTrace, include_values: bool = False) -> dict[str, Any]:
    """
    Encode a tensor trace into JSON safe dicts, container ids included.

    Args:
        trace (TensorTrace): The activation or gradient trace to encode.
        include_values (bool): When True, the value of every captured
            tensor is exported as base64 encoded raw bytes (heavy).

    Returns:
        dict[str, Any]: The JSON safe payload of the trace steps.
    """

    # One entry per step, the container identity first
    steps: list[dict[str, Any]] = []
    step: TraceStep
    for step in trace.steps:
        entry: dict[str, Any] = {
            "step": step.step,
            "container_id": step.container_id,
            "description": step.description,
            "slot_type": step.slot_type,
            "out_var": step.out_var,
            "op_name": step.op_name,
            "block_path": step.block_path,
            "frame_depth": step.frame_depth,
            "source_container_id": step.source_container_id,
            "tensor": {
                "container_id": step.tensor.container_id,
                "slot_type": step.tensor.slot_type,
                "shape": list(step.tensor.shape),
                "dtype": step.tensor.dtype
            }
        }

        # The optional base64 value of the captured tensor
        if include_values:
            entry["tensor"]["value"] = _encode_value(tensor=step.tensor.value)
        steps.append(entry)
    return {
        "format_version": FORMAT_VERSION,
        "type": "trace",
        "kind": trace.kind,
        "steps": steps
    }


def dict_to_trace(payload: dict[str, Any]) -> TensorTrace:
    """
    Decode a trace payload into an activation or gradient trace.

    The steps exported without values decode with zero filled placeholder
    tensors: only the shapes, dtypes and container ids are faithful.

    Args:
        payload (dict[str, Any]): The payload built by trace_to_dict.

    Returns:
        TensorTrace: The decoded ActivationTrace or GradientTrace.
    """

    # The format version guard, for the forward compatibility
    version: int = payload.get("format_version", 0)
    if version != FORMAT_VERSION:
        raise ValueError(f"Unsupported trace format version: {version}")

    # The trace of the recorded kind, filled step by step
    trace: TensorTrace = (
        ActivationTrace() if payload["kind"] == "activation" else GradientTrace()
    )
    entry: dict[str, Any]
    for entry in payload["steps"]:

        # The tensor description of the step
        tensor_payload: dict[str, Any] = entry["tensor"]
        shape: tuple[int, ...] = tuple(tensor_payload["shape"])
        dtype: torch.dtype = _dtype_from_name(name=tensor_payload["dtype"])

        # The decoded value, or the zero filled placeholder of the shape
        if "value" in tensor_payload:
            value: Tensor = _decode_value(
                payload=tensor_payload["value"],
                shape=shape,
                dtype=dtype
            )
        else:
            value = torch.zeros(shape, dtype=dtype)

        # The rebuilt step, with its container identity
        trace.steps.append(TraceStep(
            step=entry["step"],
            container_id=entry["container_id"],
            description=entry["description"],
            slot_type=entry["slot_type"],
            out_var=entry["out_var"],
            op_name=entry["op_name"],
            block_path=entry["block_path"],
            tensor=TensorContainer(
                shape=shape,
                dtype=tensor_payload["dtype"],
                value=value,
                container_id=tensor_payload["container_id"],
                slot_type=tensor_payload["slot_type"]
            ),
            source_container_id=entry["source_container_id"],
            frame_depth=entry["frame_depth"]
        ))
    return trace


def dump_trace_json(
    trace: TensorTrace,
    path: str,
    include_values: bool = False
) -> None:
    """
    Write a tensor trace to a JSON file.

    Args:
        trace (TensorTrace): The activation or gradient trace to write.
        path (str): The path of the output JSON file.
        include_values (bool): Whether the tensor values are exported as
            base64 encoded raw bytes.
    """

    # The encoded payload, written with the format version header
    payload: dict[str, Any] = trace_to_dict(trace=trace, include_values=include_values)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_trace_json(path: str) -> TensorTrace:
    """
    Read a tensor trace from a JSON file.

    Args:
        path (str): The path of the JSON file to read.

    Returns:
        TensorTrace: The decoded ActivationTrace or GradientTrace.
    """

    # The payload read from the file, decoded into a trace
    with open(path, "r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    return dict_to_trace(payload=payload)
def _encode_block(
    block: ModelBlockSchema,
    path: str,
    options: dict[str, bool],
    include_call_sites: bool
) -> dict[str, Any]:
    """
    Encode one block of the tree, recursing into its sub-blocks.

    Args:
        block (ModelBlockSchema): The block schema to encode.
        path (str): The attribute path of the block, "" for the root.
        options (dict[str, bool]): The shared export toggles.
        include_call_sites (bool): Whether the call site index maps are
            exported.

    Returns:
        dict[str, Any]: The JSON safe payload of the block.
    """

    # The identity of the block, with its attribute path for the viewers
    payload: dict[str, Any] = {
        "type": "block",
        "path": path,
        "method_name": block.method_name
    }

    # The declared inputs, their defaults and the returned outputs
    payload["inputs"] = {
        name: _encode_tensor_repr(repr_=tensor_repr, options=options)
        for name, tensor_repr in block.inputs.items()
    }
    payload["input_defaults"] = {
        name: _encode_arg(arg=value) for name, value in block.input_defaults.items()
    }
    payload["outputs"] = [
        _encode_tensor_repr(repr_=tensor_repr, options=options)
        for tensor_repr in block.outputs
    ]

    # The direct parameters and buffers of the block
    payload["weights"] = {
        name: _encode_tensor_repr(repr_=tensor_repr, options=options)
        for name, tensor_repr in block.weights.items()
    }

    # The children, the parsed methods and the skipped children
    payload["sub_blocks"] = {
        name: _encode_block(
            block=child,
            path=f"{path}.{name}" if path else name,
            options=options,
            include_call_sites=include_call_sites
        )
        for name, child in block.sub_blocks.items()
    }
    payload["methods"] = {
        name: _encode_block(
            block=method,
            path=f"{path}.{name}" if path else name,
            options=options,
            include_call_sites=include_call_sites
        )
        for name, method in block.methods.items()
    }
    payload["skipped"] = dict(block.skipped)

    # The optional call site index maps of the block
    if include_call_sites:
        payload["sub_block_call_sites"] = {
            name: list(indices)
            for name, indices in block.sub_block_call_sites().items()
        }
        payload["method_call_sites"] = {
            name: list(indices)
            for name, indices in method_call_sites(block=block).items()
        }

    # The dependencies of this block instructions, keyed by nested path
    dependencies: dict[str, list[Dependency]] = (
        instruction_dependencies(block=block)
        if options["include_indices"] and options["include_dependencies"]
        else {}
    )

    # The lowered body, instruction by instruction, the containers of the
    # traces attached by block path and variable name
    payload["body"] = _encode_instructions(
        instructions=block.body,
        prefix="",
        options=options,
        dependencies=dependencies,
        block_path=path
    )
    return payload


def _container_index(
    traces: TensorTrace | list[TensorTrace] | None,
    root_name: str = ""
) -> dict[tuple[str, str], list[dict[str, str]]]:
    """
    Index the container ids of a run by block path and variable name.

    The root block is traced under its module class name (e.g. "MLP")
    while the schema encoder keys it by the empty path, so the steps of
    the root name are also indexed under the empty path.

    Args:
        traces (TensorTrace | list[TensorTrace] | None): The traces to
            index, in attachment order (the activations before the
            gradients typically).
        root_name (str): The module class name of the root block, "" when
            unknown (the steps of the empty path are then already keyed
            under the empty path).

    Returns:
        dict[tuple[str, str], list[dict[str, str]]]: The container id and
        slot type of each captured step, by (block path, variable name),
        in chronological order.
    """

    # The empty index of the no trace case
    if traces is None:
        return {}

    # The single trace accepted, normalized to a list
    trace_list: list[TensorTrace] = (
        [traces] if isinstance(traces, TensorTrace) else list(traces)
    )

    # The indexed steps, one entry per (block path, variable name) pair
    index: dict[tuple[str, str], list[dict[str, str]]] = {}
    trace: TensorTrace
    for trace in trace_list:
        step: TraceStep
        for step in trace.steps:

            # The unnamed steps have no instruction to attach to
            if not step.out_var:
                continue
            key: tuple[str, str] = (step.block_path, step.out_var)
            entry: dict[str, str] = {
                "container_id": step.container_id,
                "slot_type": step.slot_type
            }
            index.setdefault(key, []).append(entry)

            # The steps of the root block are also keyed by its empty path
            if root_name and step.block_path == root_name:
                index.setdefault(("", step.out_var), []).append(entry)
    return index


def _encode_instructions(
    instructions: list[Instruction],
    prefix: str,
    options: dict[str, Any],
    dependencies: dict[str, list[Dependency]],
    block_path: str
) -> list[dict[str, Any]]:
    """
    Encode a list of instructions, recursing into the nested bodies.

    Args:
        instructions (list[Instruction]): The instructions to encode.
        prefix (str): The path prefix of the body, "" at the top level.
        options (dict[str, Any]): The shared export toggles and the
            container index of the attached traces.
        dependencies (dict[str, list[Dependency]]): The dependencies of the
            block instructions, keyed by nested instruction path.
        block_path (str): The attribute path of the block owning the
            instructions, "" for the root.

    Returns:
        list[dict[str, Any]]: The JSON safe payloads of the instructions.
    """

    # One payload per instruction, in body order
    encoded: list[dict[str, Any]] = []
    index: int
    instruction: Instruction
    for index, instruction in enumerate(instructions):

        # The nested path of the instruction, aligned with the dependencies
        path: str = f"{prefix}/{index}" if prefix else str(index)
        payload: dict[str, Any] = _encode_instruction(
            instruction=instruction,
            path=path,
            options=options,
            dependencies=dependencies,
            block_path=block_path
        )

        # The nested bodies of the control flow instructions
        if isinstance(instruction, ControlInstruction):
            payload["body"] = _encode_instructions(
                instructions=instruction.body,
                prefix=f"{path}/body",
                options=options,
                dependencies=dependencies,
                block_path=block_path
            )
            payload["else_body"] = _encode_instructions(
                instructions=instruction.else_body,
                prefix=f"{path}/else",
                options=options,
                dependencies=dependencies,
                block_path=block_path
            )

        # The nested bodies of the try instructions
        elif isinstance(instruction, TryInstruction):
            payload["body"] = _encode_instructions(
                instructions=instruction.body,
                prefix=f"{path}/body",
                options=options,
                dependencies=dependencies,
                block_path=block_path
            )
            handler_index: int
            for handler_index, (_, handler_body) in enumerate(instruction.handlers):
                payload["handlers"][handler_index]["body"] = _encode_instructions(
                    instructions=handler_body,
                    prefix=f"{path}/handlers/{handler_index}",
                    options=options,
                    dependencies=dependencies,
                    block_path=block_path
                )
            payload["final_body"] = _encode_instructions(
                instructions=instruction.final_body,
                prefix=f"{path}/finally",
                options=options,
                dependencies=dependencies,
                block_path=block_path
            )
        encoded.append(payload)
    return encoded


def _encode_instruction(
    instruction: Instruction,
    path: str,
    options: dict[str, Any],
    dependencies: dict[str, list[Dependency]],
    block_path: str
) -> dict[str, Any]:
    """
    Encode one instruction, its nested bodies excluded.

    Args:
        instruction (Instruction): The instruction to encode.
        path (str): The nested path of the instruction.
        options (dict[str, Any]): The shared export toggles and the
            container index of the attached traces.
        dependencies (dict[str, list[Dependency]]): The dependencies of the
            block instructions, keyed by nested instruction path.
        block_path (str): The attribute path of the block owning the
            instruction, "" for the root.

    Returns:
        dict[str, Any]: The JSON safe payload of the instruction.
    """

    # The common fields of every instruction, the type first
    payload: dict[str, Any] = {
        "type": type(instruction).__name__,
        "out_var": instruction.out_var
    }

    # The optional index and nested path of the instruction, the index
    # omitted for the condition expressions (not body positions)
    if options["include_indices"]:
        last_segment: str = path.rsplit("/", 1)[-1]
        if last_segment.isdigit():
            payload["index"] = int(last_segment)
        payload["path"] = path

    # The optional resolved dependencies of the instruction, dataflow edges
    if options["include_dependencies"] and path in dependencies:
        payload["dependencies"] = [
            {"name": dep.name, "kind": dep.kind, "origin": dep.origin}
            for dep in dependencies[path]
        ]

    # The optional container ids of the run attached to the instruction, by
    # its block path and the variable name it binds
    containers: dict[tuple[str, str], list[dict[str, str]]] = options["containers"]
    if instruction.out_var and containers:
        matches: list[dict[str, str]] = containers.get(
            (block_path, instruction.out_var), []
        )
        if matches:
            payload["containers"] = [dict(match) for match in matches]

    # The optional resolved shape and dtype of the instruction output
    if options["include_resolved"] and instruction.resolved is not None:
        payload["resolved"] = _encode_tensor_repr(
            repr_=instruction.resolved,
            options=options
        )

    # The type specific fields, one branch per instruction class
    if isinstance(instruction, OpInstruction):
        payload["target"] = _encode_arg(arg=instruction.target)
        payload["qualified_name"] = instruction.qualified_name
        payload["args"] = [_encode_arg(arg=arg) for arg in instruction.args]
        payload["kwargs"] = {
            key: _encode_arg(arg=arg) for key, arg in instruction.kwargs.items()
        }
        payload["args_var"] = instruction.args_var
        payload["kwargs_var"] = instruction.kwargs_var

    # The call to a sub-block, with its container semantics
    elif isinstance(instruction, BlockInstruction):
        payload["block_ref"] = {
            "path": instruction.block_ref.path,
            "index_var": instruction.block_ref.index_var
        }
        payload["args"] = [_encode_arg(arg=arg) for arg in instruction.args]
        payload["kwargs"] = {
            key: _encode_arg(arg=arg) for key, arg in instruction.kwargs.items()
        }
        payload["splat_args"] = instruction.splat_args
        payload["args_var"] = instruction.args_var
        payload["kwargs_var"] = instruction.kwargs_var

    # The control flow, its condition encoded like an operation
    elif isinstance(instruction, ControlInstruction):
        payload["kind"] = instruction.kind
        if instruction.condition is not None:
            payload["condition"] = _encode_instruction(
                instruction=instruction.condition,
                path=f"{path}/condition",
                options=options,
                dependencies=dependencies,
                block_path=block_path
            )
        else:
            payload["condition"] = None
        payload["bind_var"] = instruction.bind_var
        payload["iter_container"] = (
            {
                "path": instruction.iter_container.path,
                "index_var": instruction.iter_container.index_var
            }
            if instruction.iter_container is not None else None
        )
        payload["index_var"] = instruction.index_var

    # The try instruction, its exception types by qualified name
    elif isinstance(instruction, TryInstruction):
        payload["handlers"] = [
            {
                "exceptions": [
                    f"{exception.__module__}.{exception.__qualname__}"
                    for exception in handler_types
                ],
                "as_name": instruction.handler_names[handler_index]
            }
            for handler_index, (handler_types, _) in enumerate(
                instruction.handlers
            )
        ]

    # The return instruction, the names of the returned variables
    elif isinstance(instruction, ReturnInstruction):
        payload["out_vars"] = list(instruction.out_vars)
    return payload


def _encode_arg(arg: VarRef | Any) -> dict[str, Any]:
    """
    Encode an instruction argument or a target into JSON safe dicts.

    Args:
        arg (VarRef | Any): The argument to encode, a variable reference, a
            sub-block reference, a literal or a callable.

    Returns:
        dict[str, Any]: The tagged JSON safe payload of the argument.
    """

    # The variable references, by name
    if isinstance(arg, VarRef):
        return {"kind": "var", "name": arg.name}

    # The sub-block references, by attribute path
    if isinstance(arg, BlockRef):
        return {"kind": "block_ref", "path": arg.path, "index_var": arg.index_var}

    # The json safe scalars, kept as literals
    if arg is None or isinstance(arg, (bool, int, float, str)):
        return {"kind": "literal", "value": arg}

    # The containers of literals, with their python type to rebuild them
    if isinstance(arg, (list, tuple)):
        return {
            "kind": "literal",
            "value": [_encode_arg(arg=item) for item in arg],
            "python_type": type(arg).__name__
        }
    if isinstance(arg, dict):
        return {
            "kind": "literal",
            "value": {key: _encode_arg(arg=item) for key, item in arg.items()},
            "python_type": "dict"
        }

    # The callables and any other python value, by their dotted name or type
    if callable(arg):
        return {"kind": "callable", "name": _dotted_name_of(callable_=arg)}
    return {"kind": "opaque", "python_type": type(arg).__name__}


def _dotted_name_of(callable_: Callable[..., Any]) -> str:
    """
    Get the dotted name of a callable, for its JSON encoding.

    Args:
        callable_ (Callable[..., Any]): The callable to name.

    Returns:
        str: The module and qualname of the callable (e.g.
        "torch.nn.functional.linear"), or its plain name when it has no
        module (a lambda or a local function).
    """

    # The module qualified name, when the callable belongs to a module
    module: str | None = getattr(callable_, "__module__", None)
    qualname: str = getattr(
        callable_, "__qualname__", getattr(callable_, "__name__", repr(callable_))
    )
    if module and module != "builtins":
        return f"{module}.{qualname}"
    return qualname


def _encode_tensor_repr(
    repr_: TensorRepr | TensorReprWithValue,
    options: dict[str, bool]
) -> dict[str, Any]:
    """
    Encode a tensor representation into JSON safe dicts.

    Args:
        repr_ (TensorRepr | TensorReprWithValue): The representation to
            encode, its value exported only with include_values.
        options (dict[str, bool]): The shared export toggles.

    Returns:
        dict[str, Any]: The shape and dtype payload, with the optional
        base64 encoded value.
    """

    # The placeholder inputs have no shape to export
    payload: dict[str, Any] = {"dtype": repr_.dtype}
    if repr_.dtype != PLACEHOLDER_DTYPE:
        payload["shape"] = list(repr_.shape)

    # The optional base64 value of the held tensor
    if isinstance(repr_, TensorReprWithValue) and options["include_values"]:
        payload["value"] = _encode_value(tensor=repr_.value)
    return payload


def _decode_tensor_repr(payload: dict[str, Any]) -> TensorRepr:
    """
    Decode a tensor representation payload.

    Args:
        payload (dict[str, Any]): The payload built by _encode_tensor_repr.

    Returns:
        TensorRepr: A TensorReprWithValue when the payload holds a value,
        else a plain TensorRepr.
    """

    # The optional decoded value of the representation
    if "value" in payload:
        shape: tuple[int, ...] = tuple(payload.get("shape", ()))
        dtype: torch.dtype = _dtype_from_name(name=payload["dtype"])
        return TensorReprWithValue(
            shape=shape,
            dtype=payload["dtype"],
            value=_decode_value(payload=payload["value"], shape=shape, dtype=dtype)
        )

    # The plain shape and dtype of the representation
    return TensorRepr(shape=tuple(payload.get("shape", ())), dtype=payload["dtype"])


def _decode_weight(payload: dict[str, Any]) -> TensorReprWithValue:
    """
    Decode a weight representation payload, the interpreter requirement.

    The weights exported without values decode with zero filled placeholder
    tensors: a re-attached live module provides the real values at run
    time, the live attributes taking precedence over the snapshots.

    Args:
        payload (dict[str, Any]): The payload built by _encode_tensor_repr.

    Returns:
        TensorReprWithValue: The decoded weight representation.
    """

    # The decoded value, or the zero filled placeholder of the shape
    shape: tuple[int, ...] = tuple(payload.get("shape", ()))
    dtype: torch.dtype = _dtype_from_name(name=payload["dtype"])
    if "value" in payload:
        value: Tensor = _decode_value(
            payload=payload["value"],
            shape=shape,
            dtype=dtype
        )
    else:
        value = torch.zeros(shape, dtype=dtype)
    return TensorReprWithValue(
        shape=shape,
        dtype=payload["dtype"],
        value=value
    )


def _encode_value(tensor: Tensor) -> str:
    """
    Encode a tensor value as base64 encoded raw bytes.

    Args:
        tensor (Tensor): The tensor to encode, detached on the fly.

    Returns:
        str: The base64 of the raw bytes of the tensor.
    """

    # The raw bytes of the contiguous cpu tensor, reinterpreted as uint8
    detached: Tensor = tensor.detach().cpu().contiguous()
    raw: bytes = detached.reshape(-1).view(torch.uint8).numpy().tobytes()
    return base64.b64encode(raw).decode("ascii")


def _decode_value(payload: str, shape: tuple[int, ...], dtype: torch.dtype) -> Tensor:
    """
    Decode a base64 encoded tensor value.

    Args:
        payload (str): The base64 raw bytes built by _encode_value.
        shape (tuple[int, ...]): The shape of the decoded tensor.
        dtype (torch.dtype): The torch dtype of the decoded tensor.

    Returns:
        Tensor: The decoded tensor, on the cpu.
    """

    # The uint8 buffer reinterpreted to the dtype, reshaped to the shape
    buffer: bytearray = bytearray(base64.b64decode(payload))
    return torch.frombuffer(buffer, dtype=torch.uint8).view(dtype).reshape(shape)


def _dtype_from_name(name: str) -> torch.dtype:
    """
    Resolve a torch dtype from its serialized name.

    Args:
        name (str): The serialized dtype name (e.g. "torch.float32").

    Returns:
        torch.dtype: The resolved torch dtype.

    Raises:
        ValueError: When the name does not designate a torch dtype.
    """

    # The dtype is looked up on the torch module by its plain name
    plain_name: str = name.split(".")[-1]
    dtype: torch.dtype | None = getattr(torch, plain_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unknown torch dtype: {name}")
    return dtype


def _decode_block(payload: dict[str, Any]) -> ModelBlockSchema:
    """
    Decode a block payload into a viewer block schema tree.

    Args:
        payload (dict[str, Any]): The payload built by _encode_block.

    Returns:
        ModelBlockSchema: The decoded block schema tree, without module.
    """

    # The decoded body, instruction by instruction
    body: list[Instruction] = _decode_instructions(payloads=payload["body"])

    # The decoded tree, its children and methods recursed
    return ModelBlockSchema(
        inputs={
            name: _decode_tensor_repr(payload=repr_)
            for name, repr_ in payload["inputs"].items()
        },
        outputs=[_decode_tensor_repr(payload=repr_) for repr_ in payload["outputs"]],
        weights={
            name: _decode_weight(payload=repr_)
            for name, repr_ in payload["weights"].items()
        },
        sub_blocks={
            name: _decode_block(payload=child)
            for name, child in payload["sub_blocks"].items()
        },
        body=body,
        module=None,
        methods={
            name: _decode_block(payload=method)
            for name, method in payload["methods"].items()
        },
        method_name=payload["method_name"],
        input_defaults={
            name: _decode_arg(payload=arg)
            for name, arg in payload["input_defaults"].items()
        },
        skipped=dict(payload["skipped"])
    )


def _decode_instructions(payloads: list[dict[str, Any]]) -> list[Instruction]:
    """
    Decode a list of instruction payloads, nested bodies included.

    Args:
        payloads (list[dict[str, Any]]): The instruction payloads to decode.

    Returns:
        list[Instruction]: The decoded instructions, in body order.

    Raises:
        ValueError: When an instruction type or an argument is unknown.
    """

    # One instruction per payload, dispatched by the type discriminator
    instructions: list[Instruction] = []
    payload: dict[str, Any]
    for payload in payloads:
        instruction_type: str = payload["type"]

        # The call to a torch callable or to a variable held callable
        if instruction_type == "OpInstruction":
            instructions.append(OpInstruction(
                target=_decode_arg(payload=payload["target"]),
                qualified_name=payload["qualified_name"],
                args=[_decode_arg(payload=arg) for arg in payload["args"]],
                kwargs={
                    key: _decode_arg(payload=arg)
                    for key, arg in payload["kwargs"].items()
                },
                out_var=payload["out_var"],
                kwargs_var=payload["kwargs_var"],
                args_var=payload["args_var"]
            ))

        # The call to a sub-block
        elif instruction_type == "BlockInstruction":
            instructions.append(BlockInstruction(
                block_ref=BlockRef(
                    path=payload["block_ref"]["path"],
                    index_var=payload["block_ref"]["index_var"]
                ),
                args=[_decode_arg(payload=arg) for arg in payload["args"]],
                out_var=payload["out_var"],
                kwargs={
                    key: _decode_arg(payload=arg)
                    for key, arg in payload["kwargs"].items()
                },
                splat_args=payload["splat_args"],
                kwargs_var=payload["kwargs_var"],
                args_var=payload["args_var"]
            ))

        # The control flow, with its nested bodies recursed
        elif instruction_type == "ControlInstruction":
            instructions.append(ControlInstruction(
                kind=payload["kind"],
                condition=(
                    _decode_instructions(payloads=[payload["condition"]])[0]
                    if payload["condition"] is not None else None
                ),
                body=_decode_instructions(payloads=payload["body"]),
                else_body=_decode_instructions(payloads=payload["else_body"]),
                bind_var=payload["bind_var"],
                iter_container=(
                    BlockRef(
                        path=payload["iter_container"]["path"],
                        index_var=payload["iter_container"]["index_var"]
                    )
                    if payload["iter_container"] is not None else None
                ),
                index_var=payload["index_var"],
                out_var=payload["out_var"]
            ))

        # The try instruction, its exception types resolved by name
        elif instruction_type == "TryInstruction":
            instructions.append(TryInstruction(
                body=_decode_instructions(payloads=payload["body"]),
                handlers=[
                    (
                        tuple(
                            _decode_exception(name=name)
                            for name in handler["exceptions"]
                        ),
                        _decode_instructions(payloads=handler["body"])
                    )
                    for handler in payload["handlers"]
                ],
                handler_names=[handler["as_name"] for handler in payload["handlers"]],
                final_body=_decode_instructions(payloads=payload["final_body"]),
                out_var=payload["out_var"]
            ))

        # The return instruction
        elif instruction_type == "ReturnInstruction":
            instructions.append(ReturnInstruction(out_vars=payload["out_vars"]))

        # The loop control instructions
        elif instruction_type == "ContinueInstruction":
            instructions.append(ContinueInstruction())
        elif instruction_type == "BreakInstruction":
            instructions.append(BreakInstruction())

        # Any other type is not supported by this format version
        else:
            raise ValueError(f"Unknown instruction type: {instruction_type}")

        # The optional resolved representation of the decoded instruction
        if "resolved" in payload:
            instructions[-1].resolved = _decode_tensor_repr(
                payload=payload["resolved"]
            )
    return instructions


def _decode_arg(payload: dict[str, Any]) -> Any:
    """
    Decode an argument or target payload into its python value.

    Args:
        payload (dict[str, Any]): The tagged payload built by _encode_arg.

    Returns:
        Any: The variable reference, the sub-block reference, the literal or
        the resolved callable of the payload.

    Raises:
        ValueError: When the payload kind or an opaque value is undecodable.
    """

    # The tagged payload kinds, one branch per kind
    kind: str = payload["kind"]
    if kind == "var":
        return VarRef(name=payload["name"])
    if kind == "block_ref":
        return BlockRef(path=payload["path"], index_var=payload["index_var"])
    if kind == "literal":
        return _decode_literal(payload=payload)
    if kind == "callable":
        return _resolve_callable(name=payload["name"])
    raise ValueError(
        f"Cannot decode an argument of kind {kind} "
        f"(python type {payload.get('python_type', 'unknown')})"
    )


def _decode_literal(payload: dict[str, Any]) -> Any:
    """
    Decode a literal payload, the containers of literals included.

    Args:
        payload (dict[str, Any]): The literal payload to decode.

    Returns:
        Any: The decoded literal value.
    """

    # The containers are rebuilt with their python type
    if "python_type" in payload:
        items: list[Any] = [_decode_literal(payload=item) for item in payload["value"]]
        python_type: str = payload["python_type"]
        if python_type == "tuple":
            return tuple(items)
        if python_type == "dict":
            return dict(items)
        return items

    # The json safe scalars are returned as they are
    return payload["value"]


def _resolve_callable(name: str) -> Any:
    """
    Resolve a callable from its serialized dotted name.

    Args:
        name (str): The dotted name of the callable (e.g.
        "torch.nn.functional.linear").

    Returns:
        Any: The resolved callable, or a lazily resolving proxy when the
        name is not importable right now.
    """

    # The known namespaces and the builtins are resolved eagerly
    try:
        return resolve_dotted_name(dotted=name, node=None, forward_globals={})
    except UnsupportedSyntaxError:
        pass

    # The absolute module paths are imported directly (e.g. the runtime
    # helpers of the lowering, or any importable python callable)
    if "." in name:
        module_name: str
        qualname: str
        module_name, _, qualname = name.rpartition(".")
        try:
            resolved: Any = importlib.import_module(name=module_name)
            part: str
            for part in qualname.split("."):
                resolved = getattr(resolved, part)
            return resolved
        except (ImportError, AttributeError, ValueError):
            pass

    # Else the name is resolved lazily at call time
    return GlobalCallable(dotted=name, forward_globals={})


def _decode_exception(name: str) -> type[BaseException]:
    """
    Resolve an exception class from its serialized qualified name.

    Args:
        name (str): The qualified name of the exception (e.g.
        "builtins.ValueError").

    Returns:
        type[BaseException]: The resolved exception class.

    Raises:
        ValueError: When the name does not designate an exception class.
    """

    # The module and the qualname of the exception, imported on the fly
    module_name: str
    qualname: str
    module_name, qualname = name.rsplit(".", 1)
    module: Any = importlib.import_module(name=module_name)
    exception: Any = getattr(module, qualname)
    if not (isinstance(exception, type) and issubclass(exception, BaseException)):
        raise ValueError(f"The name does not designate an exception: {name}")
    return exception


def _reattach_module(block: ModelBlockSchema, module: Any) -> None:
    """
    Re-attach a live module tree to a decoded block schema tree.

    The sub-blocks are walked by attribute path: the plain attributes with
    getattr, the container entries ("layers.0") by index or key access. The
    children with no matching live module keep a None module.

    Args:
        block (ModelBlockSchema): The decoded root block to fill.
        module (Any): The live module of the root block.
    """

    # The root module itself, then the children by attribute path. The
    # method blocks run on the same self module, so they get it directly
    block.module = module
    name: str
    child: ModelBlockSchema
    for name, child in block.sub_blocks.items():
        child_module: Any = _walk_attribute(module=module, name=name)
        if child_module is not None:
            _reattach_module(block=child, module=child_module)
    for method in block.methods.values():
        _reattach_module(block=method, module=module)


def _walk_attribute(module: Any, name: str) -> Any:
    """
    Resolve one attribute path on a module, container entries included.

    Args:
        module (Any): The live module to walk from.
        name (str): The attribute path (e.g. "attention" or "layers.0").

    Returns:
        Any: The resolved attribute, or None when unresolvable.
    """

    # The path parts are walked one by one, getattr first
    current: Any = module
    part: str
    for part in name.split("."):

        # A plain attribute of the current module
        if hasattr(current, part):
            current = getattr(current, part)
            continue

        # An index of a container (e.g. "layers.0")
        if part.isdigit() and isinstance(current, (list, tuple, nn.ModuleList)):
            index: int = int(part)
            if index < len(current):
                current = current[index]
                continue

        # A key of a dict like container (e.g. a ModuleDict)
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue

        # The path does not resolve on the live module tree
        return None
    return current


def diff_to_dict(diffs: list[TraceDiff]) -> dict[str, Any]:
    """
    Encode the trace diffs of two runs into JSON safe dicts.

    Args:
        diffs (list[TraceDiff]): The diffs built by compare_traces, in
            their alignment order.

    Returns:
        dict[str, Any]: The JSON safe payload of the comparison, the
            per status summary first, then one entry per diff.
    """

    # One entry per diff, the alignment identity first
    return {
        "format_version": FORMAT_VERSION,
        "type": "trace_diff",
        "summary": diff_summary(diffs=diffs),
        "diffs": [
            {
                "container_id_a": diff.container_id_a,
                "container_id_b": diff.container_id_b,
                "status": diff.status,
                "matched_by": diff.matched_by,
                "description": diff.description,
                "out_var": diff.out_var,
                "block_path": diff.block_path,
                "max_abs_diff": diff.max_abs_diff,
                "mean_abs_diff": diff.mean_abs_diff,
                "changed": diff.changed
            }
            for diff in diffs
        ]
    }


def dict_to_diffs(payload: dict[str, Any]) -> list[TraceDiff]:
    """
    Decode a diff payload into the TraceDiff list of a comparison.

    Args:
        payload (dict[str, Any]): The payload built by diff_to_dict.

    Returns:
        list[TraceDiff]: The decoded diffs, in their alignment order.

    Raises:
        ValueError: When the payload format version is unsupported.
    """

    # The format version guard, for the forward compatibility
    version: int = payload.get("format_version", 0)
    if version != FORMAT_VERSION:
        raise ValueError(f"Unsupported diff format version: {version}")

    # The rebuilt diffs, the optional magnitudes kept optional
    return [
        TraceDiff(
            container_id_a=entry["container_id_a"],
            container_id_b=entry["container_id_b"],
            status=entry["status"],
            matched_by=entry["matched_by"],
            description=entry["description"],
            out_var=entry["out_var"],
            block_path=entry["block_path"],
            max_abs_diff=entry["max_abs_diff"],
            mean_abs_diff=entry["mean_abs_diff"],
            changed=entry["changed"]
        )
        for entry in payload["diffs"]
    ]


def dump_diff_json(diffs: list[TraceDiff], path: str) -> None:
    """
    Write the trace diffs of two runs to a JSON file.

    Args:
        diffs (list[TraceDiff]): The diffs built by compare_traces.
        path (str): The path of the output JSON file.
    """

    # The encoded payload, written with the format version header
    payload: dict[str, Any] = diff_to_dict(diffs=diffs)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_diff_json(path: str) -> list[TraceDiff]:
    """
    Read the trace diffs of a comparison from a JSON file.

    Args:
        path (str): The path of the JSON file to read.

    Returns:
        list[TraceDiff]: The decoded diffs, in their alignment order.
    """

    # The payload read from the file, decoded into the diff list
    with open(path, "r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    return dict_to_diffs(payload=payload)
