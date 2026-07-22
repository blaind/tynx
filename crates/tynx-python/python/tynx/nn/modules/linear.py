"""Fully connected eager layer."""

import math

from ..._tynx import Parameter, Tensor, empty
from ..init import kaiming_uniform_, uniform_
from .module import Module


class Linear(Module):
    """Apply an affine transform to the final input dimension."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = _positive_features(in_features, "in_features")
        self.out_features = _positive_features(out_features, "out_features")
        if type(bias) is not bool:
            raise TypeError(f"bias must be a bool, got {type(bias).__qualname__}")

        self.weight = Parameter(empty((self.out_features, self.in_features)), name="weight")
        kaiming_uniform_(self.weight, a=math.sqrt(5.0))
        self.bias = Parameter(empty((self.out_features,)), name="bias") if bias else None
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            uniform_(self.bias, -bound, bound)

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
