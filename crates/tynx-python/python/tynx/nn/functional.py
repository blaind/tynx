"""Composable eager neural-network functions."""

from typing import Literal, Optional, Union

from .._tynx import (
    Tensor,
    _adaptive_avg_pool2d,
    _avg_pool2d,
    _conv2d,
    _max_pool2d,
    maximum,
)

Reduction = Literal["none", "mean", "sum"]
IntOrPair = Union[int, tuple[int, int]]


def conv2d(
    input: Tensor,
    weight: Tensor,
    bias: Optional[Tensor] = None,
    stride: IntOrPair = 1,
    padding: IntOrPair = 0,
    dilation: IntOrPair = 1,
    groups: int = 1,
) -> Tensor:
    """Apply a two-dimensional convolution to an NCHW input."""
    if type(groups) is not int or groups <= 0:
        raise ValueError(f"groups must be a positive integer, got {groups!r}")
    return _conv2d(
        input,
        weight,
        bias,
        _pair(stride, "stride", positive=True),
        _pair(padding, "padding", positive=False),
        _pair(dilation, "dilation", positive=True),
        groups,
    )


def max_pool2d(
    input: Tensor,
    kernel_size: IntOrPair,
    stride: Optional[IntOrPair] = None,
    padding: IntOrPair = 0,
    dilation: IntOrPair = 1,
    ceil_mode: bool = False,
) -> Tensor:
    """Apply two-dimensional max pooling to an NCHW input."""
    kernel = _pair(kernel_size, "kernel_size", positive=True)
    return _max_pool2d(
        input,
        kernel,
        kernel if stride is None else _pair(stride, "stride", positive=True),
        _pair(padding, "padding", positive=False),
        _pair(dilation, "dilation", positive=True),
        _bool(ceil_mode, "ceil_mode"),
    )


def avg_pool2d(
    input: Tensor,
    kernel_size: IntOrPair,
    stride: Optional[IntOrPair] = None,
    padding: IntOrPair = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
    divisor_override: Optional[int] = None,
) -> Tensor:
    """Apply two-dimensional average pooling to an NCHW input."""
    if divisor_override is not None:
        raise NotImplementedError("avg_pool2d divisor_override is not supported")
    kernel = _pair(kernel_size, "kernel_size", positive=True)
    return _avg_pool2d(
        input,
        kernel,
        kernel if stride is None else _pair(stride, "stride", positive=True),
        _pair(padding, "padding", positive=False),
        _bool(ceil_mode, "ceil_mode"),
        _bool(count_include_pad, "count_include_pad"),
    )


def adaptive_avg_pool2d(input: Tensor, output_size: IntOrPair) -> Tensor:
    """Pool an NCHW input to an explicit spatial size."""
    return _adaptive_avg_pool2d(input, _pair(output_size, "output_size", positive=True))


def mse_loss(input: Tensor, target: Tensor, reduction: Reduction = "mean") -> Tensor:
    """Return elementwise, mean, or summed squared error for exactly matching shapes."""
    _require_same_shape(input, target, "mse_loss")
    error = input - target
    return _reduce(error * error, reduction)


def cross_entropy(input: Tensor, target: Tensor, reduction: Reduction = "mean") -> Tensor:
    """Return cross entropy for rank-2 logits and rank-1 int64 class targets."""
    if input.dtype != "float32" or input.ndim != 2:
        raise ValueError(
            f"cross_entropy input must be rank-2 float32 logits, got {input.dtype} {input.shape}"
        )
    if target.dtype != "int64" or target.shape != (input.shape[0],):
        raise ValueError(
            "cross_entropy target must be a rank-1 int64 Tensor with one class index per row, "
            f"got {target.dtype} {target.shape}"
        )
    selected = input.log_softmax(1).gather(1, target.unsqueeze(1)).squeeze(1)
    return _reduce(-selected, reduction)


def binary_cross_entropy_with_logits(
    input: Tensor,
    target: Tensor,
    reduction: Reduction = "mean",
) -> Tensor:
    """Return stable binary cross entropy from logits for exactly matching float32 tensors."""
    _require_same_shape(input, target, "binary_cross_entropy_with_logits")
    if input.dtype != "float32" or target.dtype != "float32":
        raise TypeError("binary_cross_entropy_with_logits requires float32 input and target")
    absolute = maximum(input, -input)
    loss = maximum(input, 0.0) - input * target + ((-absolute).exp() + 1.0).log()
    return _reduce(loss, reduction)


def _require_same_shape(input: Tensor, target: Tensor, operation: str) -> None:
    if input.shape != target.shape:
        raise ValueError(
            f"{operation} requires exactly matching shapes, got {input.shape} and {target.shape}"
        )


def _reduce(input: Tensor, reduction: Reduction) -> Tensor:
    if reduction == "none":
        return input
    if reduction == "mean":
        return input.mean()
    if reduction == "sum":
        return input.sum()
    raise ValueError(f"reduction must be 'none', 'mean', or 'sum', got {reduction!r}")


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


def _bool(value: bool, name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a bool, got {type(value).__qualname__}")
    return value


__all__ = [
    "adaptive_avg_pool2d",
    "avg_pool2d",
    "binary_cross_entropy_with_logits",
    "conv2d",
    "cross_entropy",
    "max_pool2d",
    "mse_loss",
]
