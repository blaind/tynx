"""Authored eager normalization layers."""

import math
from typing import TYPE_CHECKING, Union, cast

from ..._tynx import Buffer, Parameter, Tensor
from .module import Module

_NormalizedShape = Union[int, list[int], tuple[int, ...]]

if TYPE_CHECKING:
    from ..._tynx import TensorData


class LayerNorm(Module):
    """Normalize the final dimensions of each sample with optional affine parameters."""

    def __init__(
        self,
        normalized_shape: _NormalizedShape,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.normalized_shape = _normalized_shape(normalized_shape)
        if not isinstance(eps, (int, float)) or isinstance(eps, bool) or eps < 0:
            raise ValueError(f"eps must be a non-negative real number, got {eps!r}")
        if type(elementwise_affine) is not bool or type(bias) is not bool:
            raise TypeError("elementwise_affine and bias must be bool values")
        self.eps = float(eps)
        self.elementwise_affine = elementwise_affine
        self.weight = (
            Parameter(cast("TensorData", _filled(self.normalized_shape, 1.0)), name="weight")
            if elementwise_affine
            else None
        )
        self.bias = (
            Parameter(cast("TensorData", _filled(self.normalized_shape, 0.0)), name="bias")
            if elementwise_affine and bias
            else None
        )

    def forward(self, input: Tensor) -> Tensor:
        if input.dtype != "float32":
            raise TypeError(f"LayerNorm requires a float32 Tensor, got {input.dtype}")
        dimensions = len(self.normalized_shape)
        if input.ndim < dimensions or input.shape[-dimensions:] != self.normalized_shape:
            raise ValueError(
                f"LayerNorm expected trailing shape {self.normalized_shape}, got {input.shape}"
            )
        axes = tuple(range(input.ndim - dimensions, input.ndim))
        mean = input.mean(axes, keepdim=True)
        centered = input - mean
        variance = (centered * centered).mean(axes, keepdim=True)
        output = centered / (variance + self.eps).sqrt()
        if self.weight is not None:
            output = output * self.weight
        if self.bias is not None:
            output = output + self.bias
        return output

    def extra_repr(self) -> str:
        return (
            f"normalized_shape={self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}, bias={self.bias is not None}"
        )


class BatchNorm(Module):
    """Normalize channels using batch or running statistics."""

    _valid_ranks: tuple[int, ...] = (2, 3, 4, 5, 6)

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        if type(num_features) is not int or num_features <= 0:
            raise ValueError(f"num_features must be a positive integer, got {num_features!r}")
        if not isinstance(eps, (int, float)) or isinstance(eps, bool) or eps < 0:
            raise ValueError(f"eps must be a non-negative real number, got {eps!r}")
        if (
            not isinstance(momentum, (int, float))
            or isinstance(momentum, bool)
            or not 0.0 <= momentum <= 1.0
        ):
            raise ValueError(f"momentum must be a real number in [0, 1], got {momentum!r}")
        if type(affine) is not bool or type(track_running_stats) is not bool:
            raise TypeError("affine and track_running_stats must be bool values")
        self.num_features = num_features
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.weight = Parameter([1.0] * num_features, name="weight") if affine else None
        self.bias = Parameter([0.0] * num_features, name="bias") if affine else None
        self.running_mean = (
            Buffer([0.0] * num_features, name="running_mean") if track_running_stats else None
        )
        self.running_var = (
            Buffer([1.0] * num_features, name="running_var") if track_running_stats else None
        )

    def forward(self, input: Tensor) -> Tensor:
        if input.dtype != "float32":
            raise TypeError(f"{type(self).__name__} requires a float32 Tensor, got {input.dtype}")
        if input.ndim not in self._valid_ranks:
            raise ValueError(
                f"{type(self).__name__} expects input rank in {self._valid_ranks}, "
                f"got shape {input.shape}"
            )
        if input.shape[1] != self.num_features:
            raise ValueError(
                f"{type(self).__name__} expected {self.num_features} channels, "
                f"got shape {input.shape}"
            )

        axes = (0, *range(2, input.ndim))
        use_batch_stats = self.training or not self.track_running_stats
        if use_batch_stats:
            sample_count = math.prod(input.shape[axis] for axis in axes)
            if sample_count <= 1:
                raise ValueError(
                    f"{type(self).__name__} requires more than one value per channel "
                    "when using batch statistics"
                )
            mean = input.mean(axes, keepdim=True)
            centered = input - mean
            variance = (centered * centered).mean(axes, keepdim=True)
            if self.training and self.track_running_stats:
                self._update_running_stats(mean, variance, sample_count)
        else:
            if self.running_mean is None or self.running_var is None:
                raise RuntimeError("BatchNorm running statistics are unavailable")
            mean = self.running_mean.reshape(_channel_view(self.num_features, input.ndim))
            variance = self.running_var.reshape(_channel_view(self.num_features, input.ndim))
            centered = input - mean

        output = centered / (variance + self.eps).sqrt()
        view = _channel_view(self.num_features, input.ndim)
        if self.weight is not None:
            output = output * self.weight.reshape(view)
        if self.bias is not None:
            output = output + self.bias.reshape(view)
        return output

    def _update_running_stats(
        self, mean: Tensor, biased_variance: Tensor, sample_count: int
    ) -> None:
        if self.running_mean is None or self.running_var is None:
            return
        mean = mean.reshape(self.num_features)
        unbiased_variance = biased_variance.reshape(self.num_features) * (
            sample_count / (sample_count - 1)
        )
        self.running_mean.copy_(self.running_mean * (1.0 - self.momentum) + mean * self.momentum)
        self.running_var.copy_(
            self.running_var * (1.0 - self.momentum) + unbiased_variance * self.momentum
        )

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}, "
            f"affine={self.affine}, track_running_stats={self.track_running_stats}"
        )


class BatchNorm1d(BatchNorm):
    """Batch normalization for `(N, C)` or `(N, C, L)` inputs."""

    _valid_ranks = (2, 3)


class BatchNorm2d(BatchNorm):
    """Batch normalization for `(N, C, H, W)` inputs."""

    _valid_ranks = (4,)


def _normalized_shape(value: _NormalizedShape) -> tuple[int, ...]:
    shape: tuple[int, ...]
    if type(value) is int:
        shape = (value,)
    elif isinstance(value, (list, tuple)):
        shape = tuple(value)
    else:
        raise TypeError("normalized_shape must be an int or a list/tuple of integers")
    if not shape or len(shape) > 6:
        raise ValueError(
            "normalized_shape must be a non-empty int list/tuple with at most six dimensions"
        )
    if any(type(dimension) is not int or dimension <= 0 for dimension in shape):
        raise ValueError(f"normalized_shape dimensions must be positive integers, got {shape!r}")
    return shape


def _filled(shape: tuple[int, ...], value: float) -> object:
    if len(shape) == 1:
        return [value] * shape[0]
    return [_filled(shape[1:], value) for _ in range(shape[0])]


def _channel_view(num_features: int, rank: int) -> list[int]:
    return [1, num_features, *([1] * (rank - 2))]


__all__ = ["BatchNorm", "BatchNorm1d", "BatchNorm2d", "LayerNorm"]
