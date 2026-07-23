"""Private neural-network argument normalization helpers."""

from typing import Union as _Union

_IntOrPair = _Union[int, tuple[int, int]]


def _pair(value: _IntOrPair, name: str, *, positive: bool) -> tuple[int, int]:
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


__all__: list[str] = []
