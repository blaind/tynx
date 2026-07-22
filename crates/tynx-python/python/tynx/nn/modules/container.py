"""Sequential layer composition."""

from collections.abc import Iterator

from ..._tynx import Tensor
from .module import Module


class Sequential(Module):
    """Call a fixed list of modules in order."""

    def __init__(self, *modules: Module) -> None:
        super().__init__()
        for index, module in enumerate(modules):
            if not isinstance(module, Module):
                raise TypeError(
                    f"Sequential item {index} must be a Module, got {type(module).__qualname__}"
                )
        self.layers = list(modules)

    def forward(self, input: Tensor) -> Tensor:
        output = input
        for module in self.layers:
            output = module(output)
        return output

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, index: int) -> Module:
        return self.layers[index]

    def __iter__(self) -> Iterator[Module]:
        return iter(self.layers)

    def extra_repr(self) -> str:
        return f"layers={len(self.layers)}"


__all__ = ["Sequential"]
