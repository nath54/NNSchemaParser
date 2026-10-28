"""
Interpreter package of the NNSchemaParser.

Executes the body of a ModelBlockSchema instruction by instruction. The
interpreter is composed of an instruction executor (operations, block calls,
returns, placeholder resolutions) and a control flow executor (if, for, try),
and is used both at parse time for the shape resolution and at WebUI time for
the intermediate values capture.
"""

# Import Modules
from nnschemaparser.interpreter.interpreter import SchemaInterpreter
