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


def test_new_axis_indexing_matches_pytorch_shape_rules() -> None:
    value = tynx.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    assert value[None].shape == (1, 2, 3)
    assert value[:, None, 1:].shape == (2, 1, 2)
    assert value[..., None].shape == (2, 3, 1)
    assert value[1, None, 2].shape == (1,)
    assert value[1, None, 2].item() == pytest.approx(6.0)


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


def test_integer_list_indexing_supports_negative_and_repeated_rows() -> None:
    value = tynx.Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], requires_grad=True)

    selected = value[[2, 0, -1]]
    assert selected.tolist() == [[5.0, 6.0], [1.0, 2.0], [5.0, 6.0]]
    selected.sum().backward()

    assert value.grad is not None
    assert value.grad.tolist() == [[1.0, 1.0], [0.0, 0.0], [2.0, 2.0]]


def test_tensor_advanced_indices_support_integer_and_boolean_rows() -> None:
    value = tynx.Tensor([[1, 2], [3, 4], [5, 6]], dtype="int64")
    integer_index = tynx.Tensor([2, -1, 0], dtype="int64")
    boolean_index = tynx.Tensor([True, False, True], dtype="bool")

    assert value[integer_index].tolist() == [[5, 6], [5, 6], [1, 2]]
    assert value[boolean_index].tolist() == [[1, 2], [5, 6]]
    assert value[tynx.Tensor([False, False, False], dtype="bool")].shape == (0, 2)
    assert value[[]].shape == (0, 2)


def test_index_select_supports_any_dimension_and_gradient() -> None:
    value = tynx.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    index = tynx.Tensor([1, 0, 1], dtype="int64")

    selected = value.index_select(1, index)
    assert selected.tolist() == [[2.0, 1.0, 2.0], [4.0, 3.0, 4.0]]
    assert tynx.index_select(value, 0, index).shape == (3, 2)
    selected.sum().backward()

    assert value.grad is not None
    assert value.grad.tolist() == [[1.0, 2.0], [1.0, 2.0]]


@pytest.mark.parametrize("index", [True, (0, 0, 0)])
def test_unsupported_or_excess_indices_raise(index: object) -> None:
    value = tynx.Tensor([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises((IndexError, TypeError)):
        value[index]


def test_out_of_bounds_index_raises() -> None:
    value = tynx.Tensor([1.0, 2.0])

    with pytest.raises(IndexError, match="out of bounds"):
        value[-3]

    with pytest.raises(IndexError, match="out of bounds"):
        value[[2]]


def test_advanced_index_validation_is_explicit() -> None:
    value = tynx.Tensor([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(IndexError, match="one-dimensional"):
        value[tynx.Tensor([[0]], dtype="int64")]
    with pytest.raises(IndexError, match="does not match"):
        value[tynx.Tensor([True], dtype="bool")]
    with pytest.raises(IndexError, match="int64 or bool"):
        value[tynx.Tensor([0.0])]
    with pytest.raises(TypeError, match="only integers"):
        value[[True]]
    with pytest.raises(ValueError, match="one-dimensional"):
        value.index_select(0, tynx.Tensor([[0]], dtype="int64"))
    with pytest.raises(TypeError, match="int64"):
        value.index_select(0, tynx.Tensor([0.0]))


def test_new_axes_cannot_exceed_maximum_rank() -> None:
    value = tynx.Tensor([1.0])

    with pytest.raises(IndexError, match="maximum supported rank"):
        value[None, None, None, None, None, None, :]
