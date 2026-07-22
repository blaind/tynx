from __future__ import annotations

import math
from collections.abc import Callable
from typing import cast

import pytest
import tynx


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (tynx.zeros, [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        (tynx.ones, [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
    ],
)
def test_float_factories_create_typed_grad_leaves(
    factory: Callable[..., tynx.Tensor], expected: object
) -> None:
    value = factory((2, 3), requires_grad=True)

    assert value.shape == (2, 3)
    assert value.dtype == "float32"
    assert value.requires_grad
    assert value.tolist() == expected


def test_full_and_like_factories_preserve_or_override_dtype() -> None:
    integers = tynx.full((2, 2), 7, dtype="int64")
    booleans = tynx.ones_like(integers, dtype="bool")

    assert integers.tolist() == [[7, 7], [7, 7]]
    assert booleans.tolist() == [[True, True], [True, True]]
    assert tynx.zeros_like(integers).dtype == "int64"
    assert tynx.full_like(integers, -3).tolist() == [[-3, -3], [-3, -3]]
    assert tynx.empty_like(integers).shape == integers.shape


def test_random_factories_share_advancing_seeded_native_rng() -> None:
    tynx.manual_seed(19)
    first = cast(list[list[float]], tynx.rand((2, 4)).tolist())
    second = cast(list[list[float]], tynx.randn((2, 4)).tolist())
    tynx.manual_seed(19)

    assert tynx.rand((2, 4)).tolist() == first
    assert tynx.randn((2, 4)).tolist() == second
    assert first != tynx.rand((2, 4)).tolist()
    assert all(0.0 <= item < 1.0 for row in first for item in row)
    assert all(math.isfinite(item) for row in second for item in row)


def test_randint_bounds_and_arange_signed_steps() -> None:
    tynx.manual_seed(7)
    values = tynx.randint(-2, 3, (4, 8))
    nested = cast(list[list[int]], values.tolist())

    assert values.dtype == "int64"
    assert all(-2 <= item < 3 for row in nested for item in row)
    assert tynx.arange(5).tolist() == [0, 1, 2, 3, 4]
    assert tynx.arange(5, -2, -2).tolist() == [5, 3, 1, -1]
    assert tynx.arange(1, 5, dtype="float32", requires_grad=True).tolist() == [
        1.0,
        2.0,
        3.0,
        4.0,
    ]


def test_factory_validation_is_explicit() -> None:
    with pytest.raises(ValueError, match="rank-zero"):
        tynx.zeros(())
    with pytest.raises(TypeError, match="requires_grad"):
        tynx.ones((2,), dtype="int64", requires_grad=True)
    with pytest.raises(ValueError, match="low < high"):
        tynx.randint(3, 3, (2,))
    with pytest.raises(ValueError, match="nonzero"):
        tynx.arange(0, 3, 0)
