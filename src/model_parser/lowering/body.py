"""
Facade of the lowering of a forward method into NNViewer instructions.

Composes the statement, side effect, control flow, block call, call, expression,
operator and f-string lowering mixin classes into the single BodyLowering entry
point used by the parser.
"""

# Import Modules
from src.model_parser.lowering.base import LoweringContext
from src.model_parser.lowering.calls import CallLowering
from src.model_parser.lowering.values import FStringLowering
from src.model_parser.lowering.effects import SideEffectLowering
from src.model_parser.lowering.operators import OperatorLowering
from src.model_parser.lowering.statements import StatementLowering
from src.model_parser.lowering.block_calls import BlockCallLowering
from src.model_parser.lowering.expressions import ExpressionLowering
from src.model_parser.lowering.control_flow import ControlFlowLowering


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
