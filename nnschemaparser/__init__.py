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
    "method_call_sites",
    "diff_summary",
    "dump_schema",
    "format_step"
]

__version__: str = "0.3.1"
