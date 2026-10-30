"""
Tests of the syntax lowering features of the parser.

Run with: python -m tests.test_new_syntax or pytest tests/test_new_syntax.py
"""

# Import Modules
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


class StarredModel(nn.Module):
    """Uses starred displays and mixed starred call arguments."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape[:2]
        hidden = (*shape, -1, 8)
        y = x.reshape(*shape, -1)
        z = x.reshape(*shape, -1, 8).sum(dim=(2, 3))
        w = [*shape, 3]
        return y + z.sum() * 0 + x.reshape(hidden).sum() * 0 + sum(w) * 0


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


class ZipLoopModel(nn.ModuleList):
    """Runs pairs of layers zipped together (HiFiCodec ResBlock pattern)."""

    def __init__(self) -> None:
        super().__init__()
        self.convs1 = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.convs2 = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for c1, c2 in zip(self.convs1, self.convs2):
            x = c1(x) + 1
            x = c2(x) + 1
        return x


class NestedLoopModel(nn.Module):
    """Unpacks a nested tuple loop target from an enumerate."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        self.scales = nn.Parameter(torch.ones(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for i, (layer, scale) in enumerate(
            zip(self.layers, (self.scales[0], self.scales[1]))
        ):
            outputs.append(layer(x * scale))
        return sum(outputs)


class NotInModel(nn.Module):
    """Checks a membership negation on a dimension (torch RNN pattern)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() not in (2, 3):
            raise ValueError("bad rank")
        return x * 2


# The test cases of the module, checked against the real forwards
TEST_CASES: list[tuple[str, type[nn.Module], tuple[int, ...]]] = [
    ("while", WhileModel, (4, 4)),
    ("chain", ChainModel, (3, 3)),
    ("super", SuperModel, (2, 2)),
    ("starred", StarredModel, (2, 3, 16)),
    ("setitem", SetitemModel, (3, 4)),
    ("indexed_self", IndexedSelfModel, (2, 4)),
    ("forward_call", ForwardCallModel, (2, 4)),
    ("zip_loop", ZipLoopModel, (2, 4)),
    ("nested_loop", NestedLoopModel, (2, 4)),
    ("not_in", NotInModel, (2, 4)),
]


def check(name: str, model: nn.Module, x: torch.Tensor) -> None:
    """
    Check the schema execution of a model against its real forward.

    Args:
        name (str): The display name of the test case.
        model (nn.Module): The model to parse and execute.
        x (torch.Tensor): The example input of the model.
    """

    schema = NNParser.parse_model(
        model=model,
        inputs_tensors_shapes={"x": tuple(x.shape)},
        inputs_tensors_dtypes={"x": "torch.float32"},
        inputs_tensors_values={"x": x}
    )
    interpreter = SchemaInterpreter()
    ours = interpreter.run(block=schema, scope={"x": x})
    expected = model(x)
    assert torch.allclose(ours[0], expected), name


def run_all() -> None:
    """
    Run all the test cases of the module, printing their names.
    """

    name: str
    model_class: type[nn.Module]
    shape: tuple[int, ...]
    for name, model_class, shape in TEST_CASES:
        check(name=name, model=model_class(), x=torch.randn(*shape))
        print(f"{name}: OK")
    print("ALL OK")


def test_case() -> None:
    """
    Parametrize the schema execution checks over the test cases.
    """

    torch.manual_seed(0)
    name: str
    model_class: type[nn.Module]
    shape: tuple[int, ...]
    for name, model_class, shape in TEST_CASES:
        check(name=name, model=model_class(), x=torch.randn(*shape))


if __name__ == "__main__":
    run_all()
