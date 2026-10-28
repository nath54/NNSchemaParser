"""
Core data model of NNSchemaParser.

The schema types describing a parsed model: the block schema tree, the tensor
representations and the instruction types of the block bodies.
"""

# Import Modules
from nnschemaparser.core.model_schema import ModelBlockSchema
from nnschemaparser.core.tensor_schema import TensorRepr, TensorReprWithValue

__all__ = [
    "ModelBlockSchema",
    "TensorRepr",
    "TensorReprWithValue"
]
