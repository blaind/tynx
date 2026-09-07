"""Deterministic feature encoders."""

import math

from ..._tynx import Tensor, cat
from ...nn.modules.module import Module


class FourierFeatures(Module):
    """Append fixed power-of-two sine/cosine features on the final dimension."""

    def __init__(
        self,
        in_features: int,
        *,
        frequencies: int,
        include_input: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = _positive_integer(in_features, "in_features")
        self.frequencies = _positive_integer(frequencies, "frequencies")
        if type(include_input) is not bool:
            raise TypeError(f"include_input must be a bool, got {type(include_input).__qualname__}")
        self.include_input = include_input
        self.out_features = self.in_features * (2 * self.frequencies + int(self.include_input))

    def forward(self, input: Tensor) -> Tensor:
        if input.dtype != "float32":
            raise TypeError(f"FourierFeatures requires a float32 Tensor, got {input.dtype}")
        if not input.shape or input.shape[-1] != self.in_features:
            raise ValueError(
                "FourierFeatures expected final dimension "
                f"{self.in_features}, got shape {input.shape}"
            )
        encoded = [input] if self.include_input else []
        for exponent in range(self.frequencies):
            scaled = input * (math.pi * float(2**exponent))
            encoded.extend((scaled.sin(), scaled.cos()))
        return cat(encoded, dim=-1)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, frequencies={self.frequencies}, "
            f"include_input={self.include_input}, out_features={self.out_features}"
        )


def _positive_integer(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return value


__all__ = ["FourierFeatures"]
