"""Tests for the native Python bindings."""

from pathlib import Path

import pytest
import tynx

_SIGN_MODEL = bytes.fromhex(
    "0807120c6261636b656e642d746573743a3b0a0c0a017812017922045369676e1209746573745f73"
    "69676e5a0f0a0178120a0a08080112040a02080b620f0a0179120a0a08080112040a02080b42040a"
    "00100d"
)


def test_module_metadata() -> None:
    assert tynx.__version__
    assert callable(tynx.Session)


def test_loads_valid_model(tmp_path: Path) -> None:
    model_path = tmp_path / "sign.onnx"
    model_path.write_bytes(_SIGN_MODEL)

    session = tynx.Session(model_path)

    assert session.inputs == ["x"]
    assert session.outputs == ["sign1_out1"]
    assert repr(session) == 'Session(inputs=["x"], outputs=["sign1_out1"])'


def test_missing_model_raises_os_error(tmp_path: Path) -> None:
    missing_model = tmp_path / "missing.onnx"

    with pytest.raises(OSError):
        tynx.Session(missing_model)


def test_invalid_model_raises_value_error(tmp_path: Path) -> None:
    invalid_model = tmp_path / "invalid.onnx"
    invalid_model.write_bytes(b"not an ONNX model")

    with pytest.raises(ValueError, match="failed to parse"):
        tynx.Session(invalid_model)
