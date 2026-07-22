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


def test_cat_and_stack_validate_sequences_shapes_and_dtypes() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        tynx.cat([])
    with pytest.raises(TypeError, match="entries"):
        tynx.stack([tynx.Tensor([1.0]), 2])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="outside axis"):
        tynx.cat([tynx.Tensor([[1.0, 2.0]]), tynx.Tensor([[3.0], [4.0]])])
    with pytest.raises((TypeError, ValueError), match="dtype"):
        tynx.cat([tynx.Tensor([1.0]), tynx.Tensor([1], dtype="int64")])
