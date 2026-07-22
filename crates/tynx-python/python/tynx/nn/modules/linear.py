"""Fully connected eager layer."""

import math
import random
from typing import TYPE_CHECKING, cast

from ..._tynx import Parameter, Tensor
from .module import Module

if TYPE_CHECKING:
    from ..._tynx import TensorData


class Linear(Module):
    """Apply an affine transform to the final input dimension."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = _positive_features(in_features, "in_features")
        self.out_features = _positive_features(out_features, "out_features")
        if type(bias) is not bool:
            raise TypeError(f"bias must be a bool, got {type(bias).__qualname__}")

        bound = 1.0 / math.sqrt(self.in_features)
        weights = [
            [random.uniform(-bound, bound) for _ in range(self.in_features)]
            for _ in range(self.out_features)
        ]
        self.weight = Parameter(cast("TensorData", weights), name="weight")
        self.bias = (
            Parameter(
                [random.uniform(-bound, bound) for _ in range(self.out_features)], name="bias"
            )
            if bias
            else None
        )

    def forward(self, input: Tensor) -> Tensor:
        if input.dtype != "float32":
            raise TypeError(f"Linear requires a float32 Tensor, got {input.dtype}")
        if input.shape[-1] != self.in_features:
            raise ValueError(
                f"Linear expected final dimension {self.in_features}, got shape {input.shape}"
            )
        leading_shape = input.shape[:-1]
        flattened = input.reshape(-1, self.in_features)
        output = flattened @ self.weight.transpose(0, 1)
        if self.bias is not None:
            output = output + self.bias
        return output.reshape([*leading_shape, self.out_features])

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


def _positive_features(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


__all__ = ["Linear"]
