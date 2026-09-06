"""PyTorch-shaped tensor rolling behavior."""

import pytest
import tynx


def test_roll_flattens_when_dims_is_omitted() -> None:
    value = tynx.Tensor([[1, 2, 3], [4, 5, 6]], dtype="int64")

    assert tynx.roll(value, 2).tolist() == [[5, 6, 1], [2, 3, 4]]
    assert tynx.roll(value, -1).tolist() == [[2, 3, 4], [5, 6, 1]]


def test_roll_supports_multiple_negative_and_repeated_dimensions() -> None:
    value = tynx.Tensor([[1, 2, 3], [4, 5, 6]], dtype="int64")

    assert tynx.roll(value, (1, -1), (0, 1)).tolist() == [[5, 6, 4], [2, 3, 1]]
    assert tynx.roll(value, (1, 2), (0, 0)).tolist() == [[4, 5, 6], [1, 2, 3]]
    assert tynx.roll(value, 7, -1).tolist() == [[3, 1, 2], [6, 4, 5]]


def test_roll_preserves_bool_storage_and_empty_dimensions() -> None:
    value = tynx.Tensor([[True, False], [False, True]], dtype="bool")
    empty = tynx.empty((2, 0))

    assert tynx.roll(value, 1, 1).tolist() == [[False, True], [True, False]]
    assert tynx.roll(empty, 3, 1).shape == (2, 0)


def test_roll_routes_gradients_to_source_positions() -> None:
    value = tynx.Tensor([1.0, 2.0, 3.0], requires_grad=True)

    (tynx.roll(value, 1) * tynx.Tensor([1.0, 2.0, 3.0])).sum().backward()

    assert value.grad is not None
    assert value.grad.tolist() == [2.0, 3.0, 1.0]


def test_compile_replays_roll() -> None:
    calls = 0

    @tynx.compile(fullgraph=True)
    def rolled(input: tynx.Tensor) -> tynx.Tensor:
        nonlocal calls
        calls += 1
        return tynx.roll(input, (1, -1), (0, 1))

    rolled(tynx.Tensor([[1.0, 2.0], [3.0, 4.0]]))
    output = rolled(tynx.Tensor([[5.0, 6.0], [7.0, 8.0]]))

    assert output.tolist() == [[8.0, 7.0], [6.0, 5.0]]
    assert calls == 1
    assert rolled.compile_count == 1
    assert rolled.fallback_count == 0
    assert rolled.replay_count == 1


@pytest.mark.parametrize(
    ("shifts", "dims", "error", "message"),
    [
        ((1, 2), None, ValueError, "align"),
        ((1, 2), (0,), ValueError, "align"),
        (1, 2, IndexError, "out of range"),
        (True, 0, TypeError, "shifts"),
        (1, False, TypeError, "dims"),
    ],
)
def test_roll_validates_arguments(
    shifts: object,
    dims: object,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        tynx.roll(
            tynx.Tensor([[1.0, 2.0]]),
            shifts,  # type: ignore[arg-type]
            dims,  # type: ignore[arg-type]
        )
