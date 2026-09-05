"""PyTorch-shaped eager grid sampling behavior."""

import pytest
import tynx
from tynx.nn import functional as F


def _image(*, requires_grad: bool = False) -> tynx.Tensor:
    return tynx.Tensor(
        [[[[1.0, 2.0], [3.0, 4.0]]]],
        requires_grad=requires_grad,
    )


def test_grid_sample_bilinear_forward_and_backward() -> None:
    input = _image(requires_grad=True)
    grid = tynx.Tensor([[[[0.0, 0.0]]]], requires_grad=True)

    output = F.grid_sample(input, grid, align_corners=True)
    output.sum().backward()

    assert output.tolist() == [[[[2.5]]]]
    input_grad = input.grad
    grid_grad = grid.grad
    assert input_grad is not None
    assert input_grad.tolist() == [[[[0.25, 0.25], [0.25, 0.25]]]]
    assert grid_grad is not None
    assert grid_grad.tolist() == [[[[0.5, 1.0]]]]


def test_grid_sample_default_align_corners_matches_false() -> None:
    top_left = tynx.Tensor([[[[-1.0, -1.0]]]])

    default = F.grid_sample(_image(), top_left)
    explicit = F.grid_sample(_image(), top_left, align_corners=False)

    assert default.item() == pytest.approx(0.25)
    assert default.tolist() == explicit.tolist()


@pytest.mark.parametrize(
    ("padding_mode", "expected"),
    [("zeros", 1.0), ("border", 4.0), ("reflection", 2.5)],
)
def test_grid_sample_padding_modes(padding_mode: str, expected: float) -> None:
    grid = tynx.Tensor([[[[2.0, 2.0]]]])

    output = F.grid_sample(
        _image(),
        grid,
        padding_mode=padding_mode,  # type: ignore[arg-type]
        align_corners=True,
    )

    assert output.item() == pytest.approx(expected)


def test_grid_sample_nearest_bicubic_and_volumetric_inference() -> None:
    center = tynx.Tensor([[[[0.0, 0.0]]]])
    singleton = tynx.Tensor([[[[1.0]]]])
    volume = tynx.Tensor([[[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]]])
    volume_center = tynx.Tensor([[[[[0.0, 0.0, 0.0]]]]])

    with tynx.no_grad():
        nearest = F.grid_sample(_image(), center, mode="nearest", align_corners=True)
        bicubic = F.grid_sample(
            singleton,
            center,
            mode="bicubic",
            padding_mode="border",
            align_corners=True,
        )
        trilinear = F.grid_sample(
            volume,
            volume_center,
            padding_mode="border",
            align_corners=True,
        )

    assert nearest.item() == 1.0
    assert bicubic.item() == 1.0
    assert trilinear.item() == pytest.approx(4.5)


def test_grid_sample_falls_back_from_graph_capture() -> None:
    calls = 0
    grid = tynx.Tensor([[[[0.0, 0.0]]]])

    @tynx.compile
    def sample(input: tynx.Tensor) -> tynx.Tensor:
        nonlocal calls
        calls += 1
        return F.grid_sample(input, grid, align_corners=False)

    with pytest.warns(RuntimeWarning, match="fell back to eager"):
        first = sample(_image())
    second = sample(_image())

    assert first.item() == pytest.approx(2.5)
    assert second.item() == pytest.approx(2.5)
    assert calls == 2
    assert sample.compile_count == 0
    assert sample.fallback_count == 2


def test_grid_sample_rejects_unsupported_autograd_modes() -> None:
    with pytest.raises(NotImplementedError, match="rank-4 bilinear"):
        F.grid_sample(_image(requires_grad=True), tynx.Tensor([[[[0.0, 0.0]]]]), mode="nearest")


@pytest.mark.parametrize(
    ("input", "grid", "message"),
    [
        (tynx.Tensor([1.0]), tynx.Tensor([1.0]), "same rank, either 4 or 5"),
        (_image(), tynx.Tensor([[[[0.0]]]]), "must end in 2"),
        (
            _image(),
            tynx.Tensor([[[[0.0, 0.0]]], [[[0.0, 0.0]]]]),
            "batch dimensions",
        ),
    ],
)
def test_grid_sample_validates_shapes(
    input: tynx.Tensor,
    grid: tynx.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        F.grid_sample(input, grid)


def test_grid_sample_validates_options() -> None:
    grid = tynx.Tensor([[[[0.0, 0.0]]]])
    with pytest.raises(ValueError, match="mode"):
        F.grid_sample(_image(), grid, mode="linear")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="padding_mode"):
        F.grid_sample(_image(), grid, padding_mode="edge")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="align_corners"):
        F.grid_sample(_image(), grid, align_corners=1)  # type: ignore[arg-type]
