from __future__ import annotations

from collections.abc import Callable

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


def test_compile_replays_pooling_and_gradients() -> None:
    calls = 0

    @tynx.compile(fullgraph=True)
    def pooled(input: tynx.Tensor) -> tuple[tynx.Tensor, tynx.Tensor, tynx.Tensor]:
        nonlocal calls
        calls += 1
        return (
            F.max_pool2d(input, 2, stride=1),
            F.avg_pool2d(input, 2, stride=1),
            F.adaptive_avg_pool2d(input, (1, 1)),
        )

    first = pooled(_input())
    second_input = _input(requires_grad=True)
    second = pooled(second_input)
    (second[0].sum() + second[1].sum() + second[2].sum()).backward()

    assert [output.tolist() for output in second] == [output.tolist() for output in first]
    assert second_input.grad is not None
    assert second_input.grad.shape == second_input.shape
    assert calls == 1
    assert pooled.compile_count == 1
    assert pooled.fallback_count == 0
    assert pooled.replay_count == 1


def test_pooling_nondefault_padding_ceil_and_adaptive_modes() -> None:
    small = tynx.Tensor([[[[1.0, 2.0], [3.0, 4.0]]]])

    assert F.avg_pool2d(small, 2, stride=2, padding=1).tolist() == [[[[0.25, 0.5], [0.75, 1.0]]]]
    assert F.max_pool2d(_input(), 2, stride=2, ceil_mode=True).tolist() == [
        [[[5.0, 6.0], [8.0, 9.0]]]
    ]
    assert F.adaptive_avg_pool2d(_input(), (1, 1)).item() == pytest.approx(5.0)


def test_authored_pooling_modules_project_functionals() -> None:
    value = _input()

    maximum = tynx.nn.MaxPool2d(2, stride=1)
    average = tynx.nn.AvgPool2d(2, stride=1)
    adaptive = tynx.nn.AdaptiveAvgPool2d(1)

    assert maximum.kernel_size == (2, 2)
    assert maximum.stride == (1, 1)
    assert average.kernel_size == (2, 2)
    assert average.stride == (1, 1)
    assert adaptive.output_size == (1, 1)
    assert maximum(value).shape == (1, 1, 2, 2)
    assert average(value).shape == (1, 1, 2, 2)
    assert adaptive(value).shape == (1, 1, 1, 1)


def test_pooling_rejects_unsupported_options_and_shapes() -> None:
    with pytest.raises(NotImplementedError, match="return_indices"):
        tynx.nn.MaxPool2d(2, return_indices=True)
    with pytest.raises(NotImplementedError, match="divisor_override"):
        F.avg_pool2d(_input(), 2, divisor_override=2)
    with pytest.raises(ValueError, match="rank-4"):
        F.max_pool2d(tynx.Tensor([1.0, 2.0]), 2)


@pytest.mark.parametrize(
    "constructor, message",
    [
        (lambda: tynx.nn.MaxPool2d(0), "kernel_size"),
        (lambda: tynx.nn.MaxPool2d(2, stride=0), "stride"),
        (lambda: tynx.nn.AvgPool2d(2, padding=-1), "padding"),
        (
            lambda: tynx.nn.AdaptiveAvgPool2d(1.5),  # type: ignore[arg-type]
            "output_size",
        ),
    ],
)
def test_pooling_modules_validate_sizes_during_construction(
    constructor: Callable[[], object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        constructor()


def test_pooling_modules_validate_flags_during_construction() -> None:
    with pytest.raises(TypeError, match="ceil_mode"):
        tynx.nn.MaxPool2d(2, ceil_mode=1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="count_include_pad"):
        tynx.nn.AvgPool2d(2, count_include_pad=0)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="divisor_override"):
        tynx.nn.AvgPool2d(2, divisor_override=2)
