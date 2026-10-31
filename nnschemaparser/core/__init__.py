"""
Core data model of NNSchemaParser.

The schema types describing a parsed model: the block schema tree, the tensor
representations, the instruction types of the block bodies and the activation
and gradient traces of the executions.
"""

# Import Modules
from nnschemaparser.core.dependencies import (
    Dependency,
    instruction_dependencies,
    method_call_sites
)
from nnschemaparser.core.diff import TraceDiff, compare_traces, diff_summary
from nnschemaparser.core.traces import (
    ActivationTrace,
    GradientTrace,
    TensorTrace,
    TraceStep,
    format_step
)
from nnschemaparser.core.model_schema import ModelBlockSchema
from nnschemaparser.core.tensor_schema import (
    TensorContainer,
    TensorRepr,
    TensorReprWithValue
)

__all__ = [
    "ActivationTrace",
    "Dependency",
    "GradientTrace",
    "ModelBlockSchema",
    "TensorContainer",
    "TensorRepr",
    "TensorReprWithValue",
    "TensorTrace",
    "TraceDiff",
    "TraceStep",
    "compare_traces",
    "instruction_dependencies",
    "method_call_sites",
    "diff_summary",
    "format_step"
]
