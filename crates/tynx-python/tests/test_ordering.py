from __future__ import annotations

import pytest
import tynx


def test_sort_and_argsort_match_pytorch_shaped_results() -> None:
    input = tynx.Tensor([[3.0, 1.0, 2.0], [0.0, 5.0, 4.0]])

    values, indices = input.sort()

    assert values.tolist() == [[1.0, 2.0, 3.0], [0.0, 4.0, 5.0]]
    assert indices.tolist() == [[1, 2, 0], [0, 2, 1]]
    assert input.argsort(descending=True).tolist() == [[0, 2, 1], [1, 2, 0]]
    top_values, top_indices = tynx.sort(input, dim=0, descending=True)
    assert top_values.tolist() == [[3.0, 5.0, 4.0], [0.0, 1.0, 2.0]]
    assert top_indices.tolist() == [[0, 1, 1], [1, 0, 0]]
    assert tynx.argsort(input).tolist() == indices.tolist()


def test_sort_values_preserve_autograd_permutation() -> None:
    input = tynx.Parameter([3.0, 1.0, 2.0])
    weights = tynx.Tensor([10.0, 20.0, 30.0])

    values, _ = input.sort()
    (values * weights).sum().backward()

    assert input.grad is not None
    assert input.grad.tolist() == [30.0, 10.0, 20.0]


def test_topk_supports_largest_smallest_and_full_dimension() -> None:
    input = tynx.Tensor([[3, 1, 2], [0, 5, 4]], dtype="int64")

    largest, largest_indices = input.topk(2)
    smallest, smallest_indices = tynx.topk(input, 2, largest=False, sorted=False)
    all_values, all_indices = input.topk(3)
    empty_values, empty_indices = input.topk(0)

    assert largest.tolist() == [[3, 2], [5, 4]]
    assert largest_indices.tolist() == [[0, 2], [1, 2]]
    assert smallest.tolist() == [[1, 2], [0, 4]]
    assert smallest_indices.tolist() == [[1, 2], [0, 2]]
    assert all_values.shape == input.shape
    assert all_indices.shape == input.shape
    assert empty_values.shape == (2, 0)
    assert empty_indices.shape == (2, 0)


def test_topk_values_preserve_sparse_gradient_routing() -> None:
    input = tynx.Parameter([3.0, 1.0, 2.0])

    values, _ = input.topk(2)
    values.sum().backward()

    assert input.grad is not None
    assert input.grad.tolist() == [1.0, 0.0, 1.0]


def test_nonzero_supports_matrix_and_tuple_forms_for_all_dtypes() -> None:
    for input in (
        tynx.Tensor([[0.0, 2.0], [3.0, 0.0]]),
        tynx.Tensor([[0, 2], [3, 0]], dtype="int64"),
        tynx.Tensor([[False, True], [True, False]], dtype="bool"),
    ):
        coordinates = input.nonzero()
        rows, columns = tynx.nonzero(input, as_tuple=True)
        assert coordinates.tolist() == [[0, 1], [1, 0]]
        assert rows.tolist() == [0, 1]
        assert columns.tolist() == [1, 0]

    empty = tynx.zeros((2, 2)).nonzero()
    assert empty.shape == (0, 2)
    assert empty.tolist() == []


def test_ordering_rejects_unsupported_or_invalid_options() -> None:
    input = tynx.Tensor([3.0, 1.0, 2.0])
    with pytest.raises(NotImplementedError, match="stable=True"):
        input.sort(stable=True)
    with pytest.raises(ValueError, match="exceeds"):
        input.topk(4)
    with pytest.raises(ValueError, match="out of range"):
        input.argsort(dim=2)
    with pytest.raises(TypeError, match="bool"):
        tynx.Tensor([True, False], dtype="bool").sort()
