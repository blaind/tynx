"""Authored pooling layers."""

from typing import Optional

from ..._tynx import Tensor
from ..functional import IntOrPair, adaptive_avg_pool2d, avg_pool2d, max_pool2d
from .module import Module


class MaxPool2d(Module):
    """Apply two-dimensional max pooling to an NCHW input."""

    def __init__(
        self,
        kernel_size: IntOrPair,
        stride: Optional[IntOrPair] = None,
        padding: IntOrPair = 0,
        dilation: IntOrPair = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
    ) -> None:
        super().__init__()
        if return_indices:
            raise NotImplementedError("MaxPool2d return_indices=True is not supported")
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

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
        kernel_size: IntOrPair,
        stride: Optional[IntOrPair] = None,
        padding: IntOrPair = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad
        self.divisor_override = divisor_override

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

    def __init__(self, output_size: IntOrPair) -> None:
        super().__init__()
        self.output_size = output_size

    def forward(self, input: Tensor) -> Tensor:
        return adaptive_avg_pool2d(input, self.output_size)

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


__all__ = ["AdaptiveAvgPool2d", "AvgPool2d", "MaxPool2d"]
