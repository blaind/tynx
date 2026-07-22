"""Authored convolution layers."""

import math
from typing import TYPE_CHECKING, Union, cast

from ..._tynx import Parameter, Tensor
from .._random import uniform
from ..functional import conv2d
from .module import Module

if TYPE_CHECKING:
    from ..._tynx import TensorData

IntOrPair = Union[int, tuple[int, int]]


class Conv2d(Module):
    """Apply a two-dimensional convolution to an NCHW input."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrPair,
        stride: IntOrPair = 1,
        padding: IntOrPair = 0,
        dilation: IntOrPair = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ) -> None:
        super().__init__()
        self.in_channels = _positive_int(in_channels, "in_channels")
        self.out_channels = _positive_int(out_channels, "out_channels")
        self.kernel_size = _pair(kernel_size, "kernel_size", positive=True)
        self.stride = _pair(stride, "stride", positive=True)
        self.padding = _pair(padding, "padding", positive=False)
        self.dilation = _pair(dilation, "dilation", positive=True)
        self.groups = _positive_int(groups, "groups")
        if self.in_channels % self.groups != 0 or self.out_channels % self.groups != 0:
            raise ValueError("in_channels and out_channels must be divisible by groups")
        if type(bias) is not bool:
            raise TypeError(f"bias must be a bool, got {type(bias).__qualname__}")
        if padding_mode != "zeros":
            raise ValueError("only padding_mode='zeros' is supported")
        self.padding_mode = padding_mode

        channels_per_group = self.in_channels // self.groups
        bound = 1.0 / math.sqrt(channels_per_group * math.prod(self.kernel_size))
        weights = [
            [
                [
                    [uniform(-bound, bound) for _ in range(self.kernel_size[1])]
                    for _ in range(self.kernel_size[0])
                ]
                for _ in range(channels_per_group)
            ]
            for _ in range(self.out_channels)
        ]
        self.weight = Parameter(cast("TensorData", weights), name="weight")
        self.bias = (
            Parameter([uniform(-bound, bound) for _ in range(self.out_channels)], name="bias")
            if bias
            else None
        )

    def forward(self, input: Tensor) -> Tensor:
        if input.dtype != "float32" or input.ndim != 4:
            raise ValueError(
                f"Conv2d expects a rank-4 float32 NCHW Tensor, got {input.dtype} {input.shape}"
            )
        if input.shape[1] != self.in_channels:
            raise ValueError(
                f"Conv2d expected {self.in_channels} input channels, got {input.shape[1]}"
            )
        return conv2d(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
            f"dilation={self.dilation}, groups={self.groups}, bias={self.bias is not None}, "
            f"padding_mode={self.padding_mode!r}"
        )


def _positive_int(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


def _pair(value: IntOrPair, name: str, *, positive: bool) -> tuple[int, int]:
    pair = (value, value) if type(value) is int else value
    minimum = 1 if positive else 0
    if (
        type(pair) is not tuple
        or len(pair) != 2
        or any(type(item) is not int or item < minimum for item in pair)
    ):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be an int or pair of {qualifier} integers, got {value!r}")
    return pair


__all__ = ["Conv2d"]
