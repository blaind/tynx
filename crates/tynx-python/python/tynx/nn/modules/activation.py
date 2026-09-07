"""Callable activation layers."""

from ..._tynx import Tensor
from .module import Module


class ReLU(Module):
    """Apply rectified linear activation elementwise."""

    def forward(self, input: Tensor) -> Tensor:
        return input.relu()


class SiLU(Module):
    """Apply the sigmoid linear unit activation elementwise."""

    def forward(self, input: Tensor) -> Tensor:
        return input * input.sigmoid()


__all__ = ["ReLU", "SiLU"]
