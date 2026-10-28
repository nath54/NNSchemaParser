"""
NNSchemaParser.

Converts a torch nn.Module into an inspectable ModelBlockSchema graph that
keeps the original variable names of the forward source and every sub-module,
and executes it with a schema interpreter re-dispatching to the real torch
callables.
"""

# Import Modules
from nnschemaparser.core import ModelBlockSchema, TensorRepr, TensorReprWithValue
from nnschemaparser.interpreter import SchemaInterpreter
from nnschemaparser.parser import NNParser

__all__ = [
    "ModelBlockSchema",
    "NNParser",
    "SchemaInterpreter",
    "TensorRepr",
    "TensorReprWithValue"
]

__version__: str = "0.1.0"
