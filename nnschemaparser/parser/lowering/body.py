"""
Facade of the lowering of a forward method into NNSchemaParser instructions.

Composes the statement, side effect, control flow, block call, call, expression,
operator and f-string lowering mixin classes into the single BodyLowering entry
point used by the parser.
"""

# Import Modules
from nnschemaparser.parser.lowering.base import LoweringContext
from nnschemaparser.parser.lowering.calls import CallLowering
from nnschemaparser.parser.lowering.values import FStringLowering
from nnschemaparser.parser.lowering.effects import SideEffectLowering
from nnschemaparser.parser.lowering.operators import OperatorLowering
from nnschemaparser.parser.lowering.statements import StatementLowering
from nnschemaparser.parser.lowering.block_calls import BlockCallLowering
from nnschemaparser.parser.lowering.expressions import ExpressionLowering
from nnschemaparser.parser.lowering.control_flow import ControlFlowLowering


class BodyLowering(
    StatementLowering,
    SideEffectLowering,
    ControlFlowLowering,
    BlockCallLowering,
    CallLowering,
    ExpressionLowering,
    OperatorLowering,
    FStringLowering,
    LoweringContext
):
    """
    Lowers the AST of the forward method of a module into a list of instructions.

    The lowering keeps the original variable names of the parsed python code so
    the WebUI can display a graph that maps back to the source.
    """
