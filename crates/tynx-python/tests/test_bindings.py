"""Tests for the native Python bindings."""

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


def test_missing_model_raises_os_error(tmp_path: Path) -> None:
    missing_model = tmp_path / "missing.onnx"

    with pytest.raises(OSError):
        tynx.Session(missing_model)


def test_invalid_model_raises_value_error(tmp_path: Path) -> None:
    invalid_model = tmp_path / "invalid.onnx"
    invalid_model.write_bytes(b"not an ONNX model")

    with pytest.raises(ValueError, match="failed to parse"):
        tynx.Session(invalid_model)
