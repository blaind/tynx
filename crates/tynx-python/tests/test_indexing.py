from __future__ import annotations

import pytest
import tynx


def test_basic_indexing_matches_python_shape_and_value_rules() -> None:
    value = tynx.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    assert value[1, -2].shape == (1,)
    assert value[1, -2].item() == pytest.approx(5.0)
    assert value[..., 1:].tolist() == [[2.0, 3.0], [5.0, 6.0]]
    assert value[::-1, ::-2].tolist() == [[6.0, 4.0], [3.0, 1.0]]
    assert value[:, 3:3].shape == (2, 0)
    assert value[:, 3:3].tolist() == [[], []]


def test_basic_indexing_preserves_typed_storage() -> None:
    integers = tynx.Tensor([[1, 2], [3, 4]], dtype="int64")
    booleans = tynx.Tensor([[True, False], [False, True]], dtype="bool")

    assert integers[0].dtype == "int64"
    assert integers[0].tolist() == [1, 2]
    assert booleans[:, 1].dtype == "bool"
    assert booleans[:, 1].tolist() == [False, True]


def test_repeated_selected_value_accumulates_gradient() -> None:
    value = tynx.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    loss = value[0, 1] * 3.0 + value[0, 1]
    loss.backward()

    assert value.grad is not None
    assert value.grad.tolist() == [[0.0, 4.0], [0.0, 0.0]]


@pytest.mark.parametrize("index", [True, [0], (0, 0, 0)])
def test_unsupported_or_excess_indices_raise(index: object) -> None:
    value = tynx.Tensor([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises((IndexError, TypeError)):
        value[index]


def test_out_of_bounds_index_raises() -> None:
    value = tynx.Tensor([1.0, 2.0])

    with pytest.raises(IndexError, match="out of bounds"):
        value[-3]
