"""Tests for the native Python bindings."""

from pathlib import Path

import pytest
import tynx


def test_module_metadata() -> None:
    assert tynx.__version__
    assert callable(tynx.Session)
    assert callable(tynx.Tensor)


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


def test_missing_model_raises_os_error(tmp_path: Path) -> None:
    missing_model = tmp_path / "missing.onnx"

    with pytest.raises(OSError):
        tynx.Session(missing_model)


def test_invalid_model_raises_value_error(tmp_path: Path) -> None:
    invalid_model = tmp_path / "invalid.onnx"
    invalid_model.write_bytes(b"not an ONNX model")

    with pytest.raises(ValueError, match="failed to parse"):
        tynx.Session(invalid_model)
