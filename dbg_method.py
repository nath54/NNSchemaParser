"""Debug the espnet_transformer method call losing a value."""

# Import Modules
import sys

sys.path.insert(0, r"C:/Users/Cerisara Nathan/Documents/GitHub/NNViewer")

import torch  # noqa: I001

import tests.test_speech_models as T
from nnschemaparser.interpreter.execution import InstructionExecutor

_ORIG = InstructionExecutor._execute_method_call


def dbg(self, instruction, block, method_name, resolved_args, resolved_kwargs, scope):
    result_args = [
        (tuple(a.shape), str(a.dtype)) if isinstance(a, torch.Tensor) else repr(a)[:30]
        for a in resolved_args
    ]
    print(
        f"METHOD {type(block.module).__name__}.{method_name} "
        f"args={result_args} kwargs={list(resolved_kwargs.keys())}",
        flush=True
    )
    _ORIG(
        self,
        instruction=instruction,
        block=block,
        method_name=method_name,
        resolved_args=resolved_args,
        resolved_kwargs=resolved_kwargs,
        scope=scope
    )


InstructionExecutor._execute_method_call = dbg
T.main()
print("ALL OK")
