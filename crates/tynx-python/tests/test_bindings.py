"""Tests for the native Python bindings."""

import math
from collections.abc import Callable
from pathlib import Path

import pytest
import tynx


def test_module_metadata() -> None:
    assert tynx.__version__
    assert callable(tynx.Session)
    assert callable(tynx.Tensor)
    assert callable(tynx.Parameter)


def test_tensor_metadata_and_host_conversion() -> None:
    tensor = tynx.Tensor([[1.0, 2.0], [3.0, 4.0]])

    assert tensor.shape == (2, 2)
    assert tensor.ndim == 2
    assert tensor.numel == 4
    assert tensor.dtype == "float32"
    assert tensor.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert tynx.Tensor(3.5).shape == (1,)
    assert tynx.Tensor([3.5]).item() == pytest.approx(3.5)


def test_tensor_eager_operators() -> None:
    left = tynx.Tensor([[1.0, 2.0], [3.0, 4.0]])
    right = tynx.Tensor([[2.0, 0.5], [1.0, 2.0]])

    assert (left + right).tolist() == [[3.0, 2.5], [4.0, 6.0]]
    assert (left - right).tolist() == [[-1.0, 1.5], [2.0, 2.0]]
    assert (left * right).tolist() == [[2.0, 1.0], [3.0, 8.0]]
    assert (left / right).tolist() == [[0.5, 4.0], [3.0, 2.0]]
    assert (left @ right).tolist() == [[4.0, 4.5], [10.0, 9.5]]
    assert (-left).tolist() == [[-1.0, -2.0], [-3.0, -4.0]]


def test_tensor_scalar_and_reverse_operators() -> None:
    value = tynx.Tensor([2.0, 4.0])

    assert (value + 2).tolist() == pytest.approx([4.0, 6.0])
    assert (2 + value).tolist() == pytest.approx([4.0, 6.0])
    assert (value - 1.5).tolist() == pytest.approx([0.5, 2.5])
    assert (10 - value).tolist() == pytest.approx([8.0, 6.0])
    assert (value * 3).tolist() == pytest.approx([6.0, 12.0])
    assert (3 * value).tolist() == pytest.approx([6.0, 12.0])
    assert (value / 2).tolist() == pytest.approx([1.0, 2.0])
    assert (8 / value).tolist() == pytest.approx([4.0, 2.0])

    with pytest.raises(TypeError, match="Tensor or a real number"):
        value + "invalid"  # type: ignore[operator]


def test_tensor_reverse_scalar_operators_are_differentiable() -> None:
    value = tynx.Tensor([2.0], requires_grad=True)

    (3 - value * 2 + 8 / value).backward()

    assert value.grad is not None
    assert value.grad.tolist() == pytest.approx([-4.0])


def test_tensor_unary_activations_and_math() -> None:
    signed = tynx.Tensor([-1.0, 0.0, 1.0])
    positive = tynx.Tensor([1.0, 4.0])

    assert signed.relu().tolist() == pytest.approx([0.0, 0.0, 1.0])
    assert signed.sigmoid().tolist() == pytest.approx(
        [1.0 / (1.0 + math.e), 0.5, math.e / (1.0 + math.e)]
    )
    assert signed.tanh().tolist() == pytest.approx([-math.tanh(1.0), 0.0, math.tanh(1.0)])
    assert positive.exp().tolist() == pytest.approx([math.e, math.exp(4.0)])
    assert positive.log().tolist() == pytest.approx([0.0, math.log(4.0)])
    assert positive.sqrt().tolist() == pytest.approx([1.0, 2.0])


@pytest.mark.parametrize(
    ("operation", "point"),
    [
        (lambda value: value.relu(), 2.0),
        (lambda value: value.sigmoid(), 0.5),
        (lambda value: value.tanh(), 0.5),
        (lambda value: value.exp(), 0.5),
        (lambda value: value.log(), 2.0),
        (lambda value: value.sqrt(), 4.0),
    ],
)
def test_tensor_unary_gradients_match_finite_differences(
    operation: Callable[[tynx.Tensor], tynx.Tensor], point: float
) -> None:
    value = tynx.Tensor([point], requires_grad=True)
    operation(value).backward()

    epsilon = 1e-3
    above = operation(tynx.Tensor([point + epsilon])).item()
    below = operation(tynx.Tensor([point - epsilon])).item()
    numerical = (above - below) / (2 * epsilon)

    assert value.grad is not None
    assert value.grad.item() == pytest.approx(numerical, rel=2e-3, abs=2e-3)


def test_tensor_rejects_ragged_or_empty_data() -> None:
    with pytest.raises(ValueError, match="ragged"):
        tynx.Tensor([[1.0], [2.0, 3.0]])

    with pytest.raises(ValueError, match="empty"):
        tynx.Tensor([])


def test_tensor_item_requires_one_element() -> None:
    with pytest.raises(ValueError, match="one-element"):
        tynx.Tensor([1.0, 2.0]).item()


def test_tensor_backward_accumulates_leaf_gradients() -> None:
    value = tynx.Tensor([1.0, 2.0, 3.0], requires_grad=True)

    loss = (value * value).mean()
    assert loss.shape == (1,)
    assert loss.requires_grad
    assert not loss.is_leaf
    assert loss.grad is None
    loss.backward()

    assert value.is_leaf
    assert value.requires_grad
    assert value.grad is not None
    assert value.grad.tolist() == pytest.approx([2.0 / 3.0, 4.0 / 3.0, 2.0])

    (value * value).mean().backward()
    assert value.grad is not None
    assert value.grad.tolist() == pytest.approx([4.0 / 3.0, 8.0 / 3.0, 4.0])

    value.zero_grad()
    assert value.grad is None


def test_tensor_reductions_follow_dim_and_keepdim_shapes() -> None:
    value = tynx.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    assert value.sum().shape == (1,)
    assert value.sum().item() == pytest.approx(21.0)
    assert value.sum(0).shape == (3,)
    assert value.sum(0).tolist() == pytest.approx([5.0, 7.0, 9.0])
    assert value.mean(-1).shape == (2,)
    assert value.mean(-1).tolist() == pytest.approx([2.0, 5.0])
    assert value.mean((0, -1)).shape == (1,)
    assert value.mean((0, -1)).item() == pytest.approx(3.5)
    assert value.sum((0, 1), keepdim=True).shape == (1, 1)
    assert value.sum(1, keepdim=True).shape == (2, 1)
    assert value.sum(1, keepdim=True).tolist() == [[6.0], [15.0]]
    assert value.mean(None, keepdim=True).shape == (1, 1)
    assert value.sum(()).item() == pytest.approx(21.0)


def test_tensor_reduction_gradients_survive_axis_squeezing() -> None:
    value = tynx.Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)

    value.mean(0).sum().backward()

    assert value.grad is not None
    assert value.grad.tolist() == [[0.5, 0.5], [0.5, 0.5]]


def test_tensor_reductions_reject_invalid_dims() -> None:
    value = tynx.Tensor([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(ValueError, match="out of range"):
        value.sum(2)
    with pytest.raises(ValueError, match="more than once"):
        value.mean((0, -2))
    with pytest.raises(TypeError, match="tuple must contain only integers"):
        value.sum((0, "bad"))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="int, a tuple"):
        value.sum(True)


def test_tensor_detach_stops_gradient_tracking() -> None:
    value = tynx.Tensor([2.0], requires_grad=True)
    detached = (value * value).detach()

    assert detached.shape == (1,)
    assert not detached.requires_grad
    assert not detached.is_leaf
    with pytest.raises(ValueError, match="autodiff graph"):
        detached.backward()
    assert value.grad is None


def test_tensor_backward_rejects_non_scalar_output() -> None:
    value = tynx.Tensor([1.0, 2.0], requires_grad=True)

    with pytest.raises(ValueError, match="one-element"):
        (value * value).backward()


def test_tensor_backward_accepts_explicit_gradient() -> None:
    value = tynx.Tensor([1.0, 2.0], requires_grad=True)
    output = value * value

    output.backward(tynx.Tensor([3.0, 4.0]))

    assert value.grad is not None
    assert value.grad.tolist() == pytest.approx([6.0, 16.0])

    with pytest.raises(ValueError, match="gradient shape"):
        (value * value).backward(tynx.Tensor([1.0]))


def test_no_grad_is_nested_and_restores_tracking() -> None:
    value = tynx.Tensor([2.0], requires_grad=True)
    assert tynx.is_grad_enabled()

    with tynx.no_grad():
        assert not tynx.is_grad_enabled()
        first = value * value
        assert not first.requires_grad
        with tynx.no_grad():
            assert not tynx.is_grad_enabled()
            second = -value
            assert not second.requires_grad
        assert not tynx.is_grad_enabled()

    assert tynx.is_grad_enabled()
    assert (value * value).requires_grad


def test_parameter_is_a_named_trainable_tensor() -> None:
    parameter = tynx.Parameter([2.0], name="weight")

    assert isinstance(parameter, tynx.Tensor)
    assert parameter.name == "weight"
    assert parameter.requires_grad
    assert parameter.is_leaf
    assert parameter.tolist() == pytest.approx([2.0])


def test_parameter_accumulates_and_zeros_gradients() -> None:
    parameter = tynx.Parameter([2.0])

    (parameter * parameter).mean().backward()
    assert parameter.grad is not None
    assert parameter.grad.tolist() == pytest.approx([4.0])

    # One target entry still carries the full derivative from both uses of the slot.
    (parameter + parameter).mean().backward()
    assert parameter.grad is not None
    assert parameter.grad.tolist() == pytest.approx([6.0])

    parameter.zero_grad()
    assert parameter.grad is None

    (parameter * 3).backward()
    assert parameter.grad is not None
    assert parameter.grad.tolist() == pytest.approx([3.0])


def test_missing_model_raises_os_error(tmp_path: Path) -> None:
    missing_model = tmp_path / "missing.onnx"

    with pytest.raises(OSError):
        tynx.Session(missing_model)


def test_invalid_model_raises_value_error(tmp_path: Path) -> None:
    invalid_model = tmp_path / "invalid.onnx"
    invalid_model.write_bytes(b"not an ONNX model")

    with pytest.raises(ValueError, match="failed to parse"):
        tynx.Session(invalid_model)
