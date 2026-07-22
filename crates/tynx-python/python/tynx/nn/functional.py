"""Composable eager neural-network functions."""

from typing import Literal

from .._tynx import Tensor, maximum

Reduction = Literal["none", "mean", "sum"]


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


__all__ = ["binary_cross_entropy_with_logits", "cross_entropy", "mse_loss"]
