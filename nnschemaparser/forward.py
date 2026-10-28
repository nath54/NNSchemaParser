"""
Forward resolution helper of NNSchemaParser.

The forward methods of the transformers models are wrapped by decorators
(e.g. check_model_inputs) whose globals live in another module than the real
code. The parser needs the globals and the signature of the REAL forward, so
the wrapped method is unwrapped before the inspection.
"""

# Import Modules
import inspect
from typing import Any, Callable

import torch.nn as nn


def resolve_forward(module: nn.Module) -> Callable[..., Any]:
    """
    Get the unwrapped forward method of a module.

    Args:
        module (nn.Module): The module whose forward is resolved.

    Returns:
        Callable[..., Any]: The forward method without its decorators.
    """

    # The forward method unwrapped from its decorator chain
    return inspect.unwrap(module.forward)
