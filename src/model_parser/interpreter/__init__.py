"""
Interpreter package of the NNViewer model parser.

Executes the body of a ModelBlockSchema instruction by instruction. The
interpreter is composed of an instruction executor (operations, block calls,
returns, placeholder resolutions) and a control flow executor (if, for, try),
and is used both at parse time for the shape resolution and at WebUI time for
the intermediate values capture.
"""

# Import Modules
from src.model_parser.interpreter.interpreter import SchemaInterpreter
