"""Dropout regularization layer."""

from ..._tynx import Tensor, _dropout
from .module import Module


class Dropout(Module):
    """Randomly zero elements during training and rescale surviving values."""

    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super().__init__()
        if isinstance(p, bool) or not isinstance(p, (int, float)):
            raise TypeError("p must be a real number")
        if not 0.0 <= float(p) <= 1.0:
            raise ValueError(f"dropout probability must be between 0 and 1, got {p}")
        if not isinstance(inplace, bool):
            raise TypeError("inplace must be bool")
        if inplace:
            raise NotImplementedError("in-place Dropout is not supported")
        self.p = float(p)
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return input
        return _dropout(input, self.p)

    def extra_repr(self) -> str:
        return f"p={self.p}, inplace={self.inplace}"


__all__ = ["Dropout"]
