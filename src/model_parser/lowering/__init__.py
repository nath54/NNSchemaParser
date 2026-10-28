"""
Lowering package of the NNViewer model parser.

Turns the abstract syntax tree of the forward method of a module into the list
of instructions of a block body, keeping the original variable names of the
source code. The lowering is split into composed parts:

- the base context holds the symbol tables and the shared utilities;
- the statement, control flow, call, expression, operator and f-string parts
  lower their own syntax category into the instructions.
"""

# Import Modules
from src.model_parser.lowering.body import BodyLowering
from src.model_parser.lowering.errors import UnsupportedSyntaxError
from src.model_parser.lowering.naming import GlobalCallable
