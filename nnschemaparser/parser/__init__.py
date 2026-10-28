"""
Parser package of NNSchemaParser.

Converts torch nn.Module models into ModelBlockSchema block trees by walking
the modules recursively, lowering the AST of each forward method into
instructions and resolving the shapes and dtypes of every step.
"""

# Import Modules
from nnschemaparser.parser.parser import NNParser

__all__ = [
    "NNParser"
]
