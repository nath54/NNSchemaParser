"""
NNSchemaParser.

Converts a torch nn.Module into an inspectable ModelBlockSchema graph that
keeps the original variable names of the forward source and every sub-module,
and executes it with a schema interpreter re-dispatching to the real torch
callables.
"""

# Import Modules
from nnschemaparser.core import (
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
    SchemaInterpreter,
    TraceListener,
    TraceRecorder
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
    "GradientTrace",
    "Instruction",
    "ModelBlockSchema",
    "NNParser",
    "OpInstruction",
    "ReturnInstruction",
    "SchemaInterpreter",
    "TensorContainer",
    "TensorRepr",
    "TensorReprWithValue",
    "TensorTrace",
    "TraceListener",
    "TraceRecorder",
    "TraceStep",
    "TryInstruction",
    "VarRef",
    "dump_schema",
    "format_step"
]

__version__: str = "0.2.0"
