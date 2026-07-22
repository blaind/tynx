from __future__ import annotations

import pytest
import tynx
from tynx.nn import functional as F


def _input(*, requires_grad: bool = False) -> tynx.Tensor:
    return tynx.Tensor(
        [[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]],
        requires_grad=requires_grad,
    )


def test_max_pool2d_forward_and_backward() -> None:
    value = _input(requires_grad=True)

    output = F.max_pool2d(value, 2, stride=1)
    output.sum().backward()

    assert output.tolist() == [[[[5.0, 6.0], [8.0, 9.0]]]]
    assert value.grad is not None
    assert value.grad.tolist() == [[[[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]]]


def test_avg_pool2d_forward_and_backward() -> None:
    value = _input(requires_grad=True)

    output = F.avg_pool2d(value, 2, stride=1)
    output.sum().backward()

    assert output.tolist() == [[[[3.0, 4.0], [6.0, 7.0]]]]
    assert value.grad is not None
    assert value.grad.tolist() == [[[[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]]]]


def test_pooling_nondefault_padding_ceil_and_adaptive_modes() -> None:
    small = tynx.Tensor([[[[1.0, 2.0], [3.0, 4.0]]]])

    assert F.avg_pool2d(small, 2, stride=2, padding=1).tolist() == [[[[0.25, 0.5], [0.75, 1.0]]]]
    assert F.max_pool2d(_input(), 2, stride=2, ceil_mode=True).tolist() == [
        [[[5.0, 6.0], [8.0, 9.0]]]
    ]
    assert F.adaptive_avg_pool2d(_input(), (1, 1)).item() == pytest.approx(5.0)


def test_authored_pooling_modules_project_functionals() -> None:
    value = _input()

    assert tynx.nn.MaxPool2d(2, stride=1)(value).shape == (1, 1, 2, 2)
    assert tynx.nn.AvgPool2d(2, stride=1)(value).shape == (1, 1, 2, 2)
    assert tynx.nn.AdaptiveAvgPool2d(1)(value).shape == (1, 1, 1, 1)


def test_pooling_rejects_unsupported_options_and_shapes() -> None:
    with pytest.raises(NotImplementedError, match="return_indices"):
        tynx.nn.MaxPool2d(2, return_indices=True)
    with pytest.raises(NotImplementedError, match="divisor_override"):
        F.avg_pool2d(_input(), 2, divisor_override=2)
    with pytest.raises(ValueError, match="rank-4"):
        F.max_pool2d(tynx.Tensor([1.0, 2.0]), 2)
