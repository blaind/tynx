from __future__ import annotations

import pytest
import tynx


def test_dropout_training_uses_advancing_seeded_masks_and_backward_mask() -> None:
    layer = tynx.nn.Dropout(0.5)
    input = tynx.Tensor([1.0] * 128, requires_grad=True)

    tynx.manual_seed(91)
    first = layer(input)
    second = layer(input)
    tynx.manual_seed(91)
    replay = layer(input)

    assert first.tolist() == replay.tolist()
    assert first.tolist() != second.tolist()
    assert set(first.tolist()) <= {0.0, 2.0}

    first.sum().backward()
    assert input.grad is not None
    assert input.grad.tolist() == first.tolist()


def test_dropout_eval_and_zero_probability_are_identity_operations() -> None:
    input = tynx.Tensor([1.0, 2.0], requires_grad=True)

    assert tynx.nn.Dropout(0.0)(input) is input
    layer = tynx.nn.Dropout(0.75).eval()
    assert layer(input) is input

    layer(input).sum().backward()
    assert input.grad is not None
    assert input.grad.tolist() == pytest.approx([1.0, 1.0])


def test_dropout_one_probability_returns_finite_zeros_and_zero_gradient() -> None:
    input = tynx.Tensor([1.0, -2.0, 3.0], requires_grad=True)
    output = tynx.nn.Dropout(1.0)(input)

    assert output.tolist() == pytest.approx([0.0, 0.0, 0.0])
    output.sum().backward()
    assert input.grad is not None
    assert input.grad.tolist() == pytest.approx([0.0, 0.0, 0.0])


def test_dropout_validates_configuration_and_float_input() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        tynx.nn.Dropout(-0.1)
    with pytest.raises(ValueError, match="between 0 and 1"):
        tynx.nn.Dropout(1.1)
    with pytest.raises(TypeError, match="real number"):
        tynx.nn.Dropout(True)
    with pytest.raises(TypeError, match="inplace must be bool"):
        tynx.nn.Dropout(inplace=1)  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError, match="in-place"):
        tynx.nn.Dropout(inplace=True)
    with pytest.raises(TypeError, match="float32"):
        tynx.nn.Dropout()(tynx.Tensor([1, 2], dtype="int64"))


def test_dropout_repr_matches_public_configuration() -> None:
    assert repr(tynx.nn.Dropout(0.25)) == "Dropout(p=0.25, inplace=False)"
