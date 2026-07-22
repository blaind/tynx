from __future__ import annotations

import tynx


class WeightedSum(tynx.nn.Module):
    def forward(
        self,
        left: tynx.Tensor,
        right: tynx.Tensor,
        *,
        left_scale: float = 1.0,
        right_scale: float = 1.0,
    ) -> tynx.Tensor:
        return left * left_scale + right * right_scale


def test_module_call_forwards_multiple_positional_arguments() -> None:
    module = WeightedSum()

    result = module(tynx.Tensor([1.0, 2.0]), tynx.Tensor([3.0, 4.0]))

    assert result.tolist() == [4.0, 6.0]


def test_module_call_forwards_keyword_arguments() -> None:
    module = WeightedSum()

    result = module(
        right=tynx.Tensor([3.0, 4.0]),
        left=tynx.Tensor([1.0, 2.0]),
        left_scale=2.0,
        right_scale=-1.0,
    )

    assert result.tolist() == [-1.0, 0.0]


def test_module_call_preserves_autograd_across_arguments() -> None:
    left = tynx.Parameter([1.0, 2.0])
    right = tynx.Parameter([3.0, 4.0])

    WeightedSum()(left, right, left_scale=2.0, right_scale=3.0).sum().backward()

    assert left.grad is not None
    assert right.grad is not None
    assert left.grad.tolist() == [2.0, 2.0]
    assert right.grad.tolist() == [3.0, 3.0]
