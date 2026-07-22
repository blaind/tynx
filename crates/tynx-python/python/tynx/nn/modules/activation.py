"""Callable activation layers."""

from ..._tynx import Tensor
from .module import Module


class ReLU(Module):
    """Apply rectified linear activation elementwise."""

    def forward(self, input: Tensor) -> Tensor:
        return input.relu()


__all__ = ["ReLU"]
