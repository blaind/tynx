"""Authored convolution layers."""

import math

from ..._tynx import Parameter, Tensor, empty
from .._utils import _IntOrPair, _pair
from ..functional import conv2d
from ..init import kaiming_uniform_, uniform_
from .module import Module


class Conv2d(Module):
    """Apply a two-dimensional convolution to an NCHW input."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _IntOrPair,
        stride: _IntOrPair = 1,
        padding: _IntOrPair = 0,
        dilation: _IntOrPair = 1,
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
        self.weight = Parameter(
            empty((self.out_channels, channels_per_group, *self.kernel_size)), name="weight"
        )
        kaiming_uniform_(self.weight, a=math.sqrt(5.0))
        self.bias = Parameter(empty((self.out_channels,)), name="bias") if bias else None
        if self.bias is not None:
            fan_in = channels_per_group * math.prod(self.kernel_size)
            bound = 1.0 / math.sqrt(fan_in)
            uniform_(self.bias, -bound, bound)

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


__all__ = ["Conv2d"]
