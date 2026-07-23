"""Authored pooling layers."""

from typing import Optional as _Optional

from ..._tynx import Tensor
from .._utils import _bool, _IntOrPair, _pair
from ..functional import adaptive_avg_pool2d, avg_pool2d, max_pool2d
from .module import Module


class MaxPool2d(Module):
    """Apply two-dimensional max pooling to an NCHW input."""

    def __init__(
        self,
        kernel_size: _IntOrPair,
        stride: _Optional[_IntOrPair] = None,
        padding: _IntOrPair = 0,
        dilation: _IntOrPair = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        _pair(kernel_size, "kernel_size", positive=True)
        if stride is not None:
            _pair(stride, "stride", positive=True)
        _pair(padding, "padding", positive=False)
        _pair(dilation, "dilation", positive=True)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = _bool(return_indices, "return_indices")
        self.ceil_mode = _bool(ceil_mode, "ceil_mode")
        if self.return_indices:
            raise NotImplementedError("MaxPool2d return_indices=True is not supported")

    def forward(self, input: Tensor) -> Tensor:
        return max_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
        )

    def extra_repr(self) -> str:
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
            f"dilation={self.dilation}, ceil_mode={self.ceil_mode}"
        )


class AvgPool2d(Module):
    """Apply two-dimensional average pooling to an NCHW input."""

    def __init__(
        self,
        kernel_size: _IntOrPair,
        stride: _Optional[_IntOrPair] = None,
        padding: _IntOrPair = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: _Optional[int] = None,
    ) -> None:
        super().__init__()
        _pair(kernel_size, "kernel_size", positive=True)
        if stride is not None:
            _pair(stride, "stride", positive=True)
        _pair(padding, "padding", positive=False)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = _bool(ceil_mode, "ceil_mode")
        self.count_include_pad = _bool(count_include_pad, "count_include_pad")
        self.divisor_override = divisor_override
        if self.divisor_override is not None:
            raise NotImplementedError("AvgPool2d divisor_override is not supported")

    def forward(self, input: Tensor) -> Tensor:
        return avg_pool2d(
            input,
            self.kernel_size,
            self.stride,
            self.padding,
            self.ceil_mode,
            self.count_include_pad,
            self.divisor_override,
        )

    def extra_repr(self) -> str:
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
            f"ceil_mode={self.ceil_mode}, count_include_pad={self.count_include_pad}"
        )


class AdaptiveAvgPool2d(Module):
    """Pool an NCHW input to an explicit spatial size."""

    def __init__(self, output_size: _IntOrPair) -> None:
        super().__init__()
        _pair(output_size, "output_size", positive=True)
        self.output_size = output_size

    def forward(self, input: Tensor) -> Tensor:
        return adaptive_avg_pool2d(input, self.output_size)

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


__all__ = ["AdaptiveAvgPool2d", "AvgPool2d", "MaxPool2d"]
