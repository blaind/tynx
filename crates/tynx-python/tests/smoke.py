"""Smoke test for an installed development build."""

import tempfile

import tynx


assert tynx.__version__
assert tynx.Session

try:
    tynx.Session("does-not-exist.onnx")
except OSError:
    pass
else:
    raise AssertionError("a missing model should raise OSError")

with tempfile.NamedTemporaryFile(suffix=".onnx") as invalid_model:
    invalid_model.write(b"not an ONNX model")
    invalid_model.flush()

    try:
        tynx.Session(invalid_model.name)
    except ValueError as error:
        assert "failed to parse" in str(error)
    else:
        raise AssertionError("invalid ONNX data should fail to parse")
