from __future__ import annotations

import pytest
import tynx


def test_expand_broadcasts_without_materializing_repetition() -> None:
    value = tynx.Tensor([[1.0, 2.0, 3.0]], requires_grad=True)

    output = value.expand(2, -1)
    output.sum().backward()

    assert output.shape == (2, 3)
    assert output.tolist() == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    assert value.grad is not None
    assert value.grad.tolist() == [[2.0, 2.0, 2.0]]


def test_expand_can_prepend_dimensions() -> None:
    value = tynx.Tensor([1, 2, 3], dtype="int64")

    assert value.expand(2, -1).tolist() == [[1, 2, 3], [1, 2, 3]]


def test_repeat_materializes_tiles_and_accumulates_gradients() -> None:
    value = tynx.Tensor([[1.0, 2.0]], requires_grad=True)

    output = value.repeat(2, 3)
    output.sum().backward()

    assert output.tolist() == [
        [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
    ]
    assert value.grad is not None
    assert value.grad.tolist() == [[6.0, 6.0]]


def test_repeat_can_prepend_dimensions_and_preserves_bool() -> None:
    value = tynx.Tensor([True, False], dtype="bool")

    output = value.repeat(2, 3, 1)

    assert output.shape == (2, 3, 2)
    assert output.dtype == "bool"


def test_expand_and_repeat_validate_shape_contracts() -> None:
    value = tynx.Tensor([[1.0, 2.0]])

    with pytest.raises(ValueError, match="cannot expand"):
        value.expand(2, 3)
    with pytest.raises(ValueError, match="new leading"):
        value.expand(-1, 1, 2)
    with pytest.raises(ValueError, match="counts"):
        value.repeat(2)


def test_leading_expand_replays_under_capture_and_repeat_rejects_strict_capture() -> None:
    expanded = tynx.compile(lambda value: value.expand(2, -1), fullgraph=True)
    value = tynx.Tensor([1.0, 2.0, 3.0])

    assert expanded(value).tolist() == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]
    assert expanded(value).tolist() == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]

    repeated = tynx.compile(lambda input: input.repeat(2), fullgraph=True)
    with pytest.raises(RuntimeError, match="cannot capture"):
        repeated(value)
