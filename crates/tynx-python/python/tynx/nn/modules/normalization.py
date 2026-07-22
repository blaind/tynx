"""Authored eager normalization layers."""

from typing import TYPE_CHECKING, Union, cast

from ..._tynx import Parameter, Tensor
from .module import Module

if TYPE_CHECKING:
    from ..._tynx import TensorData


class LayerNorm(Module):
    """Normalize the final dimensions of each sample with optional affine parameters."""

    def __init__(
        self,
        normalized_shape: Union[int, tuple[int, ...]],
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


def _normalized_shape(value: Union[int, tuple[int, ...]]) -> tuple[int, ...]:
    shape = (value,) if type(value) is int else value
    if type(shape) is not tuple or not shape or len(shape) > 6:
        raise ValueError(
            "normalized_shape must be a non-empty int tuple with at most six dimensions"
        )
    if any(type(dimension) is not int or dimension <= 0 for dimension in shape):
        raise ValueError(f"normalized_shape dimensions must be positive integers, got {shape!r}")
    return shape


def _filled(shape: tuple[int, ...], value: float) -> object:
    if len(shape) == 1:
        return [value] * shape[0]
    return [_filled(shape[1:], value) for _ in range(shape[0])]


__all__ = ["LayerNorm"]
