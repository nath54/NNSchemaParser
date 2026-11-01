"""
NNSchemaParser.

Converts a torch nn.Module into an inspectable ModelBlockSchema graph that
keeps the original variable names of the forward source and every sub-module,
and executes it with a schema interpreter re-dispatching to the real torch
callables.
"""

# Import Modules
from nnschemaparser.core import (
    Dependency,
    TraceDiff,
    compare_traces,
    diff_summary,
    instruction_dependencies,
    method_call_sites,
    ActivationTrace,
    GradientTrace,
    ModelBlockSchema,
    TensorContainer,
    TensorRepr,
    TensorReprWithValue,
    TensorTrace,
    TraceStep,
    format_step
)
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
from nnschemaparser.dump import dump_schema
from nnschemaparser.interpreter import (
    ActivationRecorder,
    ExecutionRecord,
    SchemaInterpreter,
    StepInfo,
    TraceListener,
    TraceRecorder,
    capture_run
)
from nnschemaparser.parser import NNParser
from nnschemaparser.serialization import (
    dict_to_diffs,
    dict_to_schema,
    dict_to_trace,
    diff_to_dict,
    dump_diff_json,
    dump_json,
    dump_trace_json,
    load_diff_json,
    load_json,
    load_trace_json,
    schema_to_dict,
    trace_to_dict
)

__all__ = [
    "ActivationRecorder",
    "ActivationTrace",
    "BlockInstruction",
    "BlockRef",
    "BreakInstruction",
    "ContinueInstruction",
    "ControlInstruction",
    "Dependency",
    "ExecutionRecord",
    "GradientTrace",
    "Instruction",
    "ModelBlockSchema",
    "NNParser",
    "OpInstruction",
    "ReturnInstruction",
    "SchemaInterpreter",
    "StepInfo",
    "TensorContainer",
    "TensorRepr",
    "TensorReprWithValue",
    "TensorTrace",
    "TraceListener",
    "TraceRecorder",
    "TraceDiff",
    "TraceStep",
    "TryInstruction",
    "VarRef",
    "capture_run",
    "compare_traces",
    "instruction_dependencies",
    "load_diff_json",
    "load_json",
    "load_trace_json",
    "method_call_sites",
    "schema_to_dict",
    "trace_to_dict",
    "diff_summary",
    "dict_to_diffs",
    "dict_to_schema",
    "dict_to_trace",
    "diff_to_dict",
    "dump_diff_json",
    "dump_json",
    "dump_schema",
    "dump_trace_json",
    "format_step"
]

__version__: str = "0.3.2"
