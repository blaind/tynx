from __future__ import annotations

import pytest
import tynx


def test_cat_and_stack_values_shapes_and_negative_dims() -> None:
    left = tynx.Tensor([[1.0, 2.0], [3.0, 4.0]])
    right = tynx.Tensor([[5.0, 6.0]])

    assert tynx.cat([left, right], dim=0).tolist() == [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ]
    assert tynx.cat((left, left), dim=-1).shape == (2, 4)
    assert tynx.stack([left, left], dim=-1).shape == (2, 2, 2)


@pytest.mark.parametrize("dtype", ["int64", "bool"])
def test_cat_and_stack_preserve_typed_storage(dtype: str) -> None:
    data = [1, 0] if dtype == "int64" else [True, False]
    value = tynx.Tensor(data, dtype=dtype)  # type: ignore[arg-type]

    assert tynx.cat([value, value]).dtype == dtype
    assert tynx.stack([value, value]).dtype == dtype


def test_cat_and_stack_route_gradients_to_every_source() -> None:
    left = tynx.Tensor([1.0, 2.0], requires_grad=True)
    right = tynx.Tensor([3.0, 4.0], requires_grad=True)

    loss = tynx.cat([left, right]).sum() + tynx.stack([left, right]).sum()
    loss.backward()

    assert left.grad is not None
    assert right.grad is not None
    assert left.grad.tolist() == [2.0, 2.0]
    assert right.grad.tolist() == [2.0, 2.0]


def test_compile_replays_cat_stack_and_gradients() -> None:
    calls = 0

    @tynx.compile(fullgraph=True)
    def combine(left: tynx.Tensor, right: tynx.Tensor) -> tuple[tynx.Tensor, tynx.Tensor]:
        nonlocal calls
        calls += 1
        return tynx.cat([left, right], dim=0), tynx.stack([left, right], dim=1)

    combine(tynx.Tensor([[1.0, 2.0]]), tynx.Tensor([[3.0, 4.0]]))
    left = tynx.Tensor([[5.0, 6.0]], requires_grad=True)
    right = tynx.Tensor([[7.0, 8.0]], requires_grad=True)
    joined, stacked = combine(left, right)
    (joined.sum() + stacked.sum()).backward()

    assert joined.tolist() == [[5.0, 6.0], [7.0, 8.0]]
    assert stacked.shape == (1, 2, 2)
    assert left.grad is not None
    assert left.grad.tolist() == [[2.0, 2.0]]
    assert right.grad is not None
    assert right.grad.tolist() == [[2.0, 2.0]]
    assert calls == 1
    assert combine.compile_count == 1
    assert combine.fallback_count == 0
    assert combine.replay_count == 1


@pytest.mark.parametrize(
    "dtype",
    ["int64", "bool"],
)
def test_compile_replays_typed_stack(dtype: str) -> None:
    @tynx.compile(fullgraph=True)
    def duplicate(input: tynx.Tensor) -> tynx.Tensor:
        return tynx.stack([input, input])

    if dtype == "int64":
        duplicate(tynx.Tensor([1, 2], dtype="int64"))
        expected = [2, 1]
        result = duplicate(tynx.Tensor(expected, dtype="int64"))
    else:
        duplicate(tynx.Tensor([True, False], dtype="bool"))
        expected = [False, True]
        result = duplicate(tynx.Tensor(expected, dtype="bool"))

    assert result.tolist() == [expected, expected]
    assert duplicate.replay_count == 1


def test_cat_and_stack_validate_sequences_shapes_and_dtypes() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        tynx.cat([])
    with pytest.raises(TypeError, match="entries"):
        tynx.stack([tynx.Tensor([1.0]), 2])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="outside axis"):
        tynx.cat([tynx.Tensor([[1.0, 2.0]]), tynx.Tensor([[3.0], [4.0]])])
    with pytest.raises((TypeError, ValueError), match="dtype"):
        tynx.cat([tynx.Tensor([1.0]), tynx.Tensor([1], dtype="int64")])
