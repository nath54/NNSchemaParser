"""Test while loops, chained assignments, tuple unpacking and super()."""

# Import Modules
import sys

sys.path.insert(0, r"C:/Users/Cerisara Nathan/Documents/GitHub/NNViewer")

import torch  # noqa: I001
from torch import nn  # noqa: I001

from nnschemaparser import SchemaInterpreter  # noqa: E402
from nnschemaparser.parser import NNParser  # noqa: E402


class WhileModel(nn.Module):
    """Unrolls n iterations of a residual addition with a while loop."""

    def __init__(self) -> None:
        super().__init__()
        self.n = 3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        i = 0
        while i < self.n:
            x = x + 0.1
            i = i + 1
        return x


class ChainModel(nn.Module):
    """Uses chained assignment and tuple slice unpacking."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = b = x * 2
        c, d = (y + 1 for y in (a, b))
        e, f, g = (a, b, c)[:3]
        return e + f + g + d


class ParentLayer(nn.Module):
    """Parent layer with a forward used through super()."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * 3


class SuperModel(ParentLayer):
    """Delegates to the parent forward through an explicit super."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x) + 1


def check(name: str, model: nn.Module, x: torch.Tensor) -> None:
    schema = NNParser.parse_model(
        model=model,
        inputs_tensors_shapes={"x": tuple(x.shape)},
        inputs_tensors_dtypes={"x": "torch.float32"},
        inputs_tensors_values={"x": x}
    )
    interpreter = SchemaInterpreter()
    ours = interpreter.run(block=schema, scope={"x": x})
    expected = model(x)
    ok = torch.allclose(ours[0], expected)
    print(f"{name}: match={ok}")
    assert ok, name


check("while", WhileModel(), torch.randn(4, 4))
check("chain", ChainModel(), torch.randn(3, 3))
check("super", SuperModel(), torch.randn(2, 2))
print("ALL OK")


class StarredModel(nn.Module):
    """Uses starred displays and mixed starred call arguments."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:2]
        hidden = (*shape, -1, 8)
        y = x.reshape(*shape, -1)
        z = x.reshape(*shape, -1, 8).sum(dim=(2, 3))
        w = [*shape, 3]
        return y + z.sum() * 0 + x.reshape(hidden).sum() * 0 + sum(w) * 0


check("starred", StarredModel(), torch.randn(2, 3, 16))


class SetitemModel(nn.Module):
    """Assigns to the items of an object (setitem lowering)."""

    def __init__(self) -> None:
        super().__init__()
        self.cache: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.cache["x"] = x
        y = self.cache["x"] * 2
        self.cache["y"] = y
        return self.cache["y"] + 1


check("setitem", SetitemModel(), torch.randn(3, 4))


class IndexedSelfModel(nn.ModuleList):
    """Calls its own entries through self[index](...)."""

    def __init__(self) -> None:
        super().__init__()
        self.append(nn.Linear(4, 4))
        self.append(nn.ReLU())
        self.append(nn.Linear(4, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self[0](x)
        x = self[1](x)
        x = self[-1](x)
        return x


check("indexed_self", IndexedSelfModel(), torch.randn(2, 4))


class ForwardCallModel(nn.ModuleList):
    """Calls its entries through an explicit .forward() method call."""

    def __init__(self) -> None:
        super().__init__()
        self.append(nn.Linear(4, 4))
        self.append(nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for entry in self:
            x = entry.forward(x)
        return x


check("forward_call", ForwardCallModel(), torch.randn(2, 4))
