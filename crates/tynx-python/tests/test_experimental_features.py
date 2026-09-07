import math
from collections.abc import Callable
from typing import cast

import pytest
import tynx


def test_fourier_features_have_declared_shape_and_values() -> None:
    layer = tynx.experimental.nn.FourierFeatures(2, frequencies=2, include_input=True)
    input = tynx.Tensor([[0.0, 0.5]])

    result = layer(input)

    assert layer.out_features == 10
    assert result.shape == (1, 10)
    assert result.flatten().tolist() == pytest.approx(
        [
            0.0,
            0.5,
            0.0,
            math.sin(math.pi * 0.5),
            1.0,
            math.cos(math.pi * 0.5),
            0.0,
            math.sin(2.0 * math.pi * 0.5),
            1.0,
            math.cos(2.0 * math.pi * 0.5),
        ],
        abs=1e-6,
    )


def test_fourier_features_capture_and_backpropagate() -> None:
    layer = tynx.experimental.nn.FourierFeatures(3, frequencies=4, include_input=True)
    compiled = tynx.compile(layer.forward, fullgraph=True)
    input = tynx.Tensor([[0.1, 0.2, 0.3]], requires_grad=True)

    output = compiled(input)
    output.sum().backward()
    replay = compiled(tynx.Tensor([[0.2, 0.3, 0.4]]))

    assert output.shape == (1, 27)
    assert replay.shape == (1, 27)
    assert input.grad is not None
    assert all(math.isfinite(cast(float, value)) for value in input.grad.flatten().tolist())
    assert compiled.compile_count == 1
    assert compiled.replay_count == 1


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: tynx.experimental.nn.FourierFeatures(0, frequencies=1), "in_features"),
        (lambda: tynx.experimental.nn.FourierFeatures(2, frequencies=0), "frequencies"),
        (
            lambda: tynx.experimental.nn.FourierFeatures(
                2,
                frequencies=1,
                include_input=1,  # type: ignore[arg-type]
            ),
            "include_input",
        ),
    ],
)
def test_fourier_features_reject_invalid_configuration(
    factory: Callable[[], object], message: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()


def test_fourier_features_reject_wrong_shape_and_dtype() -> None:
    layer = tynx.experimental.nn.FourierFeatures(3, frequencies=1)
    with pytest.raises(ValueError, match="final dimension 3"):
        layer(tynx.Tensor([[1.0, 2.0]]))
    with pytest.raises(TypeError, match="float32"):
        layer(tynx.Tensor([[1, 2, 3]], dtype="int64"))


def test_fourier_features_are_not_exposed_as_stable_nn_api() -> None:
    assert not hasattr(tynx.nn, "FourierFeatures")
