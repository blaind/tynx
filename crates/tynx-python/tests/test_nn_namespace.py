"""Public neural-network namespace coverage."""

import tynx
from tynx.nn import Buffer


def test_buffer_is_exported_from_nn_namespace() -> None:
    assert Buffer is tynx.Buffer
    assert tynx.nn.Buffer is tynx.Buffer
    assert "Buffer" in tynx.nn.__all__
