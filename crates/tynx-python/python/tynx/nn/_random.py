"""Private RNG for authored module parameter initialization."""

import random

_generator = random.Random()


def seed(value: int) -> None:
    _generator.seed(value)


def uniform(lower: float, upper: float) -> float:
    return _generator.uniform(lower, upper)


__all__: list[str] = []
