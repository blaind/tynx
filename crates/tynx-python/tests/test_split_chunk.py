from __future__ import annotations

import pytest
import tynx


def test_split_accepts_size_and_explicit_sections() -> None:
    value = tynx.arange(7)

    assert [part.tolist() for part in value.split(3)] == [[0, 1, 2], [3, 4, 5], [6]]
    assert [part.tolist() for part in tynx.split(value, [2, 1, 4])] == [
        [0, 1],
        [2],
        [3, 4, 5, 6],
    ]


def test_chunk_returns_at_most_requested_parts() -> None:
    value = tynx.arange(5)

    assert [part.tolist() for part in value.chunk(3)] == [[0, 1], [2, 3], [4]]
    assert [part.tolist() for part in tynx.chunk(tynx.arange(2), 5)] == [[0], [1]]


def test_split_and_chunk_support_negative_dims_and_typed_storage() -> None:
    value = tynx.Tensor([[True, False, True], [False, True, False]], dtype="bool")

    parts = value.split([1, 2], dim=-1)
    assert [part.shape for part in parts] == [(2, 1), (2, 2)]
    assert all(part.dtype == "bool" for part in parts)


def test_split_branches_remain_on_one_autodiff_tape() -> None:
    value = tynx.Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    left, right = value.split(2)

    (left.sum() * 2.0 + right.sum() * 3.0).backward()

    assert value.grad is not None
    assert value.grad.tolist() == [2.0, 2.0, 3.0, 3.0]


def test_split_and_chunk_validate_arguments() -> None:
    value = tynx.arange(4)

    with pytest.raises(ValueError, match="positive"):
        value.split(0)
    with pytest.raises(ValueError, match="sum"):
        value.split([1, 1])
    with pytest.raises(ValueError, match="positive"):
        value.chunk(0)
