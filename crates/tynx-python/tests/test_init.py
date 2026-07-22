from __future__ import annotations

import math
from typing import cast

import pytest
import tynx


def test_uniform_and_normal_initialization_use_seeded_native_rng() -> None:
    first = tynx.Parameter(tynx.empty((3, 4)))
    second = tynx.Parameter(tynx.empty((3, 4)))

    tynx.manual_seed(42)
    returned = tynx.nn.init.uniform_(first, -0.25, 0.5)
    tynx.manual_seed(42)
    tynx.nn.init.uniform_(second, -0.25, 0.5)

    assert returned is first
    assert first.tolist() == second.tolist()
    values = cast(list[list[float]], first.tolist())
    assert all(-0.25 <= value <= 0.5 for row in values for value in row)

    tynx.manual_seed(91)
    tynx.nn.init.normal_(first, mean=2.0, std=0.5)
    tynx.manual_seed(91)
    tynx.nn.init.normal_(second, mean=2.0, std=0.5)
    assert first.tolist() == second.tolist()


def test_constant_zero_and_one_initializers_preserve_parameter_identity() -> None:
    parameter = tynx.Parameter([[4.0, 5.0], [6.0, 7.0]])

    assert tynx.nn.init.constant_(parameter, 2.5) is parameter
    assert parameter.tolist() == [[2.5, 2.5], [2.5, 2.5]]
    assert tynx.nn.init.zeros_(parameter) is parameter
    assert parameter.tolist() == [[0.0, 0.0], [0.0, 0.0]]
    assert tynx.nn.init.ones_(parameter) is parameter
    assert parameter.tolist() == [[1.0, 1.0], [1.0, 1.0]]


def test_truncated_normal_is_bounded_and_seeded() -> None:
    first = tynx.Parameter(tynx.empty((512,)))
    second = tynx.Parameter(tynx.empty((512,)))

    tynx.manual_seed(101)
    returned = tynx.nn.init.trunc_normal_(first, mean=0.5, std=0.75, a=-0.5, b=1.25)
    tynx.manual_seed(101)
    tynx.nn.init.trunc_normal_(second, mean=0.5, std=0.75, a=-0.5, b=1.25)

    values = cast(list[float], first.tolist())
    assert returned is first
    assert first.tolist() == second.tolist()
    assert all(-0.5 <= value <= 1.25 for value in values)
    assert len(set(values)) > 100


def test_truncated_normal_rejects_invalid_or_impractical_intervals() -> None:
    parameter = tynx.Parameter([0.0])
    with pytest.raises(ValueError, match="std > 0"):
        tynx.nn.init.trunc_normal_(parameter, std=0.0)
    with pytest.raises(ValueError, match="a < b"):
        tynx.nn.init.trunc_normal_(parameter, a=1.0, b=1.0)
    with pytest.raises(ValueError, match="too narrow"):
        tynx.nn.init.trunc_normal_(parameter, a=-0.001, b=0.001)


def test_xavier_initializers_use_expected_scale() -> None:
    uniform = tynx.Parameter(tynx.empty((6, 4)))
    normal = tynx.Parameter(tynx.empty((6, 4)))

    tynx.nn.init.xavier_uniform_(uniform)
    tynx.nn.init.xavier_normal_(normal)

    bound = math.sqrt(6.0 / 10.0)
    values = cast(list[list[float]], uniform.tolist())
    assert all(-bound <= value <= bound for row in values for value in row)
    assert normal.shape == (6, 4)


def test_kaiming_initializers_support_fan_modes_and_gain() -> None:
    uniform = tynx.Parameter(tynx.empty((8, 4, 3, 3)))
    normal = tynx.Parameter(tynx.empty((8, 4, 3, 3)))

    tynx.nn.init.kaiming_uniform_(uniform, mode="fan_in", nonlinearity="relu")
    tynx.nn.init.kaiming_normal_(normal, mode="fan_out", nonlinearity="relu")

    bound = math.sqrt(3.0) * math.sqrt(2.0) / math.sqrt(4 * 3 * 3)
    flattened = cast(list[float], uniform.flatten().tolist())
    assert all(-bound <= value <= bound for value in flattened)
    assert normal.shape == (8, 4, 3, 3)
    assert tynx.nn.init.calculate_gain("tanh") == pytest.approx(5.0 / 3.0)


def test_initializers_validate_shape_dtype_and_options() -> None:
    with pytest.raises(ValueError, match="at least two dimensions"):
        tynx.nn.init.xavier_uniform_(tynx.Parameter([0.0, 0.0]))
    with pytest.raises(ValueError, match="fan_in"):
        tynx.nn.init.kaiming_uniform_(tynx.Parameter([[0.0]]), mode="sideways")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="float32"):
        tynx.nn.init.uniform_(tynx.Tensor([1], dtype="int64"))
    with pytest.raises(ValueError, match="std"):
        tynx.nn.init.normal_(tynx.Parameter([0.0]), std=-1.0)


def test_linear_and_conv2d_initialization_share_native_seed() -> None:
    tynx.manual_seed(7)
    first_linear = tynx.nn.Linear(4, 3)
    first_conv = tynx.nn.Conv2d(2, 3, 3)
    tynx.manual_seed(7)
    second_linear = tynx.nn.Linear(4, 3)
    second_conv = tynx.nn.Conv2d(2, 3, 3)

    assert first_linear.weight.tolist() == second_linear.weight.tolist()
    assert first_linear.bias is not None and second_linear.bias is not None
    assert first_linear.bias.tolist() == second_linear.bias.tolist()
    assert first_conv.weight.tolist() == second_conv.weight.tolist()
    assert first_conv.bias is not None and second_conv.bias is not None
    assert first_conv.bias.tolist() == second_conv.bias.tolist()
