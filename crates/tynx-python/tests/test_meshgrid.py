"""PyTorch-shaped coordinate-grid behavior."""

import pytest
import tynx


def test_meshgrid_ij_values_and_sequence_form() -> None:
    rows = tynx.Tensor([1, 2], dtype="int64")
    columns = tynx.Tensor([10, 20, 30], dtype="int64")

    row_grid, column_grid = tynx.meshgrid([rows, columns], indexing="ij")

    assert row_grid.tolist() == [[1, 1, 1], [2, 2, 2]]
    assert column_grid.tolist() == [[10, 20, 30], [10, 20, 30]]
    assert row_grid.dtype == "int64"


def test_meshgrid_xy_values() -> None:
    x = tynx.Tensor([1.0, 2.0])
    y = tynx.Tensor([10.0, 20.0, 30.0])

    x_grid, y_grid = tynx.meshgrid(x, y, indexing="xy")

    assert x_grid.tolist() == [[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]
    assert y_grid.tolist() == [[10.0, 10.0], [20.0, 20.0], [30.0, 30.0]]


def test_meshgrid_defaults_to_ij_and_supports_three_dimensions() -> None:
    first = tynx.Tensor([1.0, 2.0])
    second = tynx.Tensor([3.0])
    third = tynx.Tensor([4.0, 5.0])

    grids = tynx.meshgrid(first, second, third)

    assert [grid.shape for grid in grids] == [(2, 1, 2)] * 3
    assert grids[0].tolist() == [[[1.0, 1.0]], [[2.0, 2.0]]]
    assert grids[1].tolist() == [[[3.0, 3.0]], [[3.0, 3.0]]]
    assert grids[2].tolist() == [[[4.0, 5.0]], [[4.0, 5.0]]]


def test_meshgrid_routes_gradients_through_expansion() -> None:
    x = tynx.Tensor([1.0, 2.0], requires_grad=True)
    y = tynx.Tensor([3.0, 4.0, 5.0], requires_grad=True)

    x_grid, y_grid = tynx.meshgrid(x, y, indexing="ij")
    (x_grid + y_grid).sum().backward()

    assert x.grad is not None
    assert x.grad.tolist() == [3.0, 3.0]
    assert y.grad is not None
    assert y.grad.tolist() == [2.0, 2.0, 2.0]


def test_compile_replays_meshgrid() -> None:
    calls = 0

    @tynx.compile(fullgraph=True)
    def grids(x: tynx.Tensor, y: tynx.Tensor) -> tuple[tynx.Tensor, tynx.Tensor]:
        nonlocal calls
        calls += 1
        x_grid, y_grid = tynx.meshgrid(x, y, indexing="ij")
        return x_grid, y_grid

    grids(tynx.Tensor([1.0, 2.0]), tynx.Tensor([3.0, 4.0, 5.0]))
    x_grid, y_grid = grids(tynx.Tensor([6.0, 7.0]), tynx.Tensor([8.0, 9.0, 10.0]))

    assert x_grid.tolist() == [[6.0, 6.0, 6.0], [7.0, 7.0, 7.0]]
    assert y_grid.tolist() == [[8.0, 9.0, 10.0], [8.0, 9.0, 10.0]]
    assert calls == 1
    assert grids.compile_count == 1
    assert grids.fallback_count == 0
    assert grids.replay_count == 1


def test_meshgrid_supports_single_and_bool_inputs() -> None:
    (single,) = tynx.meshgrid(tynx.Tensor([True, False], dtype="bool"), indexing="ij")

    assert single.tolist() == [True, False]
    assert single.dtype == "bool"


def test_meshgrid_validates_inputs_and_indexing() -> None:
    value = tynx.Tensor([1.0])
    with pytest.raises(ValueError, match="non-empty"):
        tynx.meshgrid(indexing="ij")
    with pytest.raises(TypeError, match="inputs"):
        tynx.meshgrid(value, [1.0], indexing="ij")  # type: ignore[list-item]
    with pytest.raises(ValueError, match="rank-1"):
        tynx.meshgrid(tynx.Tensor([[1.0]]), indexing="ij")
    with pytest.raises(TypeError, match="same dtype"):
        tynx.meshgrid(value, tynx.Tensor([1], dtype="int64"), indexing="ij")
    with pytest.raises(ValueError, match="indexing"):
        tynx.meshgrid(value, indexing="yx")  # type: ignore[arg-type]
